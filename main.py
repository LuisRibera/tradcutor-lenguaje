from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from audio import ServicioAudio
from camara import GestorCamara
from clasificador import ClasificadorSenias
from configuracion import (
    ALTO_VENTANA,
    ANCHO_PANEL,
    ANCHO_VENTANA,
    FRAMES_CONFIRMACION,
    FRAMES_FIN_GESTO,
    INDICE_CAMARA,
    LONGITUD_SECUENCIA_MODELO,
    LONGITUD_SECUENCIA_PALABRA,
    MIN_FRAMES_GESTO,
    POSE_MUNIECAS,
    UMBRAL_CONFIANZA,
    UMBRAL_MOVIMIENTO,
    obtener_tipo_etiqueta,
)
from editor_texto import EditorTexto
from extraccion_caracteristicas import ExtractorCaracteristicas
from interfaz import InterfazTraductor
from normalizacion import normalizar_holistic, normalizar_landmarks
from secuencias import redimensionar_secuencia
from seguimiento_holistic import SeguimientoHolistic
from seguimiento_manos import SeguimientoManos
from traductor import ConfirmadorLetra
from ventana import crear_ventana, enfocar_ventana, tamano_pantalla

NOMBRE_VENTANA = "Traductor de Lenguaje"


def esta_dentro_del_roi(caja: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = caja
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    centro_x = (x1 + x2) // 2
    centro_y = (y1 + y2) // 2
    return roi_x1 <= centro_x <= roi_x2 and roi_y1 <= centro_y <= roi_y2


class MedidorMovimiento:
    """Estima la velocidad de un punto (fracción del frame por frame) por su desplazamiento
    reciente. Sirve para el gate: mano/brazo quietos -> letra; en movimiento -> gesto."""

    def __init__(self, ventana: int = 4) -> None:
        self.historial = deque(maxlen=ventana)

    def actualizar(self, punto: tuple[float, float] | None) -> float:
        if punto is None:
            self.historial.clear()
            return 0.0
        self.historial.append(np.array(punto, dtype=np.float32))
        if len(self.historial) < 2:
            return 0.0
        distancias = [
            float(np.linalg.norm(self.historial[i] - self.historial[i - 1]))
            for i in range(1, len(self.historial))
        ]
        return sum(distancias) / len(distancias)


def _centro_caja_normalizado(caja, ancho, alto) -> tuple[float, float]:
    x1, y1, x2, y2 = caja
    return ((x1 + x2) / 2.0 / ancho, (y1 + y2) / 2.0 / alto)


def _punto_brazos(pose) -> tuple[float, float] | None:
    if not pose:
        return None
    izq, der = pose[POSE_MUNIECAS[0]], pose[POSE_MUNIECAS[1]]
    return ((izq[0] + der[0]) / 2.0, (izq[1] + der[1]) / 2.0)


def _clasificar_gesto(clasificador, buffer_letra, buffer_palabra):
    """Clasifica el segmento de movimiento como letra con movimiento o frase (la de mayor
    confianza por encima de su umbral). Devuelve (etiqueta, confianza, fuente) o None."""
    cand_palabra = None
    if len(buffer_palabra) >= max(4, LONGITUD_SECUENCIA_PALABRA // 2):
        seq = redimensionar_secuencia(buffer_palabra, LONGITUD_SECUENCIA_PALABRA)
        cand_palabra = clasificador.predecir_palabra(seq)

    # Un gesto largo con el cuerpo visible es una frase: no dejamos que una letra con
    # movimiento (gestos cortos de una mano) lo pise.
    if cand_palabra is not None and len(buffer_palabra) >= LONGITUD_SECUENCIA_PALABRA:
        return cand_palabra

    candidatos = []
    if cand_palabra is not None:
        candidatos.append(cand_palabra)
    if len(buffer_letra) >= max(4, LONGITUD_SECUENCIA_MODELO // 2):
        seq = redimensionar_secuencia(buffer_letra, LONGITUD_SECUENCIA_MODELO)
        pred = clasificador.predecir_letra_dinamica(seq)
        if pred:
            candidatos.append(pred)
    if not candidatos:
        return None
    return max(candidatos, key=lambda pred: pred[1])


def ejecutar() -> None:
    camara = GestorCamara(INDICE_CAMARA, ANCHO_VENTANA, ALTO_VENTANA)
    seguidor_manos = SeguimientoManos()
    seguidor_holistic = SeguimientoHolistic()
    extractor_estatico = ExtractorCaracteristicas(max_frames_buffer=6)
    clasificador = ClasificadorSenias()
    confirmador = ConfirmadorLetra(umbral_confianza=UMBRAL_CONFIANZA, frames_confirmacion=FRAMES_CONFIRMACION)
    audio = ServicioAudio(habilitado=True)
    interfaz = InterfazTraductor(ancho_panel=ANCHO_PANEL)
    editor = EditorTexto()

    medidor_mano = MedidorMovimiento()
    medidor_brazo = MedidorMovimiento()

    # Acumuladores del segmento de gesto (listas sin tope: se re-muestrean al terminar).
    buffer_letra: list = []
    buffer_palabra: list = []
    grabando_gesto = False
    frames_movimiento = 0
    frames_quieto = 0

    etiqueta_reconocida = ""
    acciones_mouse: list[str] = []

    # Entradas rápidas por teclado que se muestran como una detección normal (efecto de
    # validación + texto en la oración + audio). Los números insertan una frase completa.
    FRASES_TECLA = {
        "1": "Hola",
        "2": "¿cómo estás?",
        "3": "¿puedo ayudarte?",
        "4": "¿dónde está el baño?",
        "5": "mi nombre es",
        "6": "¿quieres comer conmigo?",
    }
    SIM_FRAMES_TOTAL = FRAMES_CONFIRMACION  # dura lo mismo que una validación real
    sim_activa = False
    sim_frames = 0
    sim_texto = ""
    sim_es_palabra = False

    pantalla = tamano_pantalla()
    escala = [1.0]  # factor de escala de la ventana (mutable para el callback del mouse)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            factor = escala[0] or 1.0
            accion = interfaz.boton_en(int(x / factor), int(y / factor))
            if accion:
                acciones_mouse.append(accion)

    def aplicar_accion(accion: str) -> None:
        if accion == "leer":
            audio.reproducir_texto(editor.texto())
        elif accion == "limpiar":
            editor.limpiar()
        elif accion == "espacio":
            editor.espacio()
        elif accion == "borrar":
            editor.retroceso()

    def emitir_texto(texto: str, es_palabra: bool) -> None:
        nonlocal etiqueta_reconocida
        if es_palabra:
            editor.insertar_palabra(texto)
        else:
            editor.insertar_letra(texto)
        audio.reproducir_texto(texto)
        etiqueta_reconocida = texto

    def emitir(etiqueta: str) -> None:
        emitir_texto(etiqueta, obtener_tipo_etiqueta(etiqueta) == "palabra")

    try:
        camara.abrir()
        crear_ventana(NOMBRE_VENTANA)
        cv2.setMouseCallback(NOMBRE_VENTANA, on_mouse)
        enfocado = False

        while True:
            frame = camara.obtener_frame()
            frame = cv2.flip(frame, 1)
            alto, ancho = frame.shape[:2]
            roi = interfaz.calcular_roi(frame)

            manos = seguidor_manos.detectar_manos(frame)
            seguidor_manos.dibujar_landmarks(frame, manos)
            resultado = seguidor_holistic.detectar(frame)
            seguidor_holistic.dibujar(frame, resultado)

            if sim_activa:
                # Entrada por teclado con prioridad: se muestra como una detección normal
                # (barra de validación) y al completarse inserta el texto + audio.
                sim_frames += 1
                progreso_ui = min(1.0, sim_frames / SIM_FRAMES_TOTAL)
                etiqueta_ui = sim_texto
                if sim_frames >= SIM_FRAMES_TOTAL:
                    emitir_texto(sim_texto, sim_es_palabra)
                    sim_activa = False
                    sim_frames = 0
                    extractor_estatico.reiniciar()
                    confirmador.reiniciar()
                    grabando_gesto = False
                    frames_movimiento = 0
                    frames_quieto = 0
                    buffer_letra.clear()
                    buffer_palabra.clear()
            else:
                mano_valida = next((m for m in manos if esta_dentro_del_roi(m.caja, roi)), None)

                # --- Gate de movimiento (mano en ROI y brazos) ---
                punto_mano = _centro_caja_normalizado(mano_valida.caja, ancho, alto) if mano_valida else None
                velocidad = max(
                    medidor_mano.actualizar(punto_mano),
                    medidor_brazo.actualizar(_punto_brazos(resultado.pose)),
                )
                en_movimiento = velocidad > UMBRAL_MOVIMIENTO

                if en_movimiento:
                    # Estamos en un gesto: acumular frames de mano (letra) y de cuerpo (palabra).
                    frames_quieto = 0
                    if not grabando_gesto:
                        grabando_gesto = True
                        frames_movimiento = 0
                        buffer_letra.clear()
                        buffer_palabra.clear()
                    if mano_valida is not None:
                        buffer_letra.append(normalizar_landmarks(mano_valida.landmarks))
                    if resultado.hay_cuerpo:
                        buffer_palabra.append(
                            normalizar_holistic(resultado.pose, resultado.mano_izq, resultado.mano_der)
                        )
                    frames_movimiento += 1
                    extractor_estatico.reiniciar()
                    confirmador.actualizar("", 0.0, False)  # sin letras durante el movimiento
                elif grabando_gesto:
                    # Pausa dentro de un gesto: NO cortamos al primer frame quieto (una micro-pausa
                    # a mitad de una frase la partiría). El gesto termina tras varios quietos seguidos.
                    frames_quieto += 1
                    if frames_quieto >= FRAMES_FIN_GESTO:
                        if frames_movimiento >= MIN_FRAMES_GESTO:
                            emision = _clasificar_gesto(clasificador, buffer_letra, buffer_palabra)
                            if emision is not None:
                                emitir(emision[0])
                        grabando_gesto = False
                        frames_movimiento = 0
                        frames_quieto = 0
                        buffer_letra.clear()
                        buffer_palabra.clear()
                    extractor_estatico.reiniciar()
                    confirmador.actualizar("", 0.0, False)
                else:
                    # Sin gesto en curso: camino de letra estática (mano quieta dentro del ROI).
                    if mano_valida is not None:
                        vector = extractor_estatico.extraer_vector(mano_valida.landmarks)
                        suavizado = extractor_estatico.obtener_vector_suavizado()
                        vector_pred = suavizado if suavizado is not None else vector
                        etiqueta, confianza, _ = clasificador.predecir_estatico(vector_pred, mano_valida.landmarks)
                        emision_letra = confirmador.actualizar(etiqueta, confianza, True)
                        etiqueta_reconocida = etiqueta if not emision_letra else etiqueta_reconocida
                        if emision_letra:
                            emitir(emision_letra)
                    else:
                        extractor_estatico.reiniciar()
                        confirmador.actualizar("", 0.0, False)

                progreso_ui = confirmador.progreso_validacion
                etiqueta_ui = confirmador.etiqueta_en_validacion

            vista = interfaz.combinar_vista(
                frame=frame,
                roi=roi,
                texto_oracion=editor.texto(),
                posicion_cursor=editor.posicion(),
                etiqueta_reconocida=etiqueta_reconocida,
                progreso_validacion=progreso_ui,
                etiqueta_validacion=etiqueta_ui,
            )

            # Escalar para que la ventana (video + panel) quepa en la pantalla del monitor.
            factor = min(1.0, pantalla[0] * 0.92 / vista.shape[1], pantalla[1] * 0.92 / vista.shape[0])
            escala[0] = factor
            if factor < 1.0:
                vista = cv2.resize(vista, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

            cv2.imshow(NOMBRE_VENTANA, vista)
            codigo = cv2.waitKeyEx(1)
            if not enfocado:
                enfocar_ventana(NOMBRE_VENTANA)
                enfocado = True

            while acciones_mouse:
                aplicar_accion(acciones_mouse.pop(0))

            if codigo == -1:
                continue
            if codigo == 27:  # ESC: salir
                break
            if codigo == 13:  # Enter: leer la oración completa
                audio.reproducir_texto(editor.texto())
                continue
            if codigo == 8:  # Retroceso
                editor.retroceso()
                continue
            if codigo in (2424832, 65361):  # flecha izquierda (Windows / Linux)
                editor.mover_izquierda()
                continue
            if codigo in (2555904, 65363):  # flecha derecha
                editor.mover_derecha()
                continue
            bajo = codigo & 0xFF
            caracter = chr(bajo) if 32 <= bajo < 256 else ""
            if caracter == " ":
                editor.espacio()
            elif not sim_activa and caracter in FRASES_TECLA:
                sim_activa, sim_frames, sim_texto, sim_es_palabra = True, 0, FRASES_TECLA[caracter], True
            elif not sim_activa and caracter.isalpha():
                sim_activa, sim_frames, sim_texto, sim_es_palabra = True, 0, caracter.upper(), False
    finally:
        camara.liberar()
        seguidor_manos.cerrar()
        seguidor_holistic.cerrar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ejecutar()
