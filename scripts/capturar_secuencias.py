from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camara import GestorCamara
from configuracion import (
    ALTO_VENTANA,
    ANCHO_VENTANA,
    ARCHIVO_DATASET_DINAMICO,
    FRAMES_CONFIRMACION_STOP,
    INDICE_CAMARA,
    MARGEN_FRAMES_STOP,
    MAX_SEGUNDOS_GRABACION_GESTO,
    SEGUNDOS_CUENTA_REGRESIVA,
    VERDE_DETECCION,
)
from normalizacion import normalizar_holistic
from secuencias import guardar_secuencia_jsonl
from seguimiento_holistic import SeguimientoHolistic, palmas_alzadas_abiertas
from ventana import crear_ventana, enfocar_ventana

TITULO_VENTANA = "Captura de palabras (Holistic)"
ROJO = (0, 80, 220)
MIN_FRAMES_SECUENCIA = 8


def capturar_secuencias(etiqueta: str, total_secuencias: int, ruta_dataset: Path, persona: str) -> None:
    camara = GestorCamara(INDICE_CAMARA, ANCHO_VENTANA, ALTO_VENTANA)
    seguidor = SeguimientoHolistic()

    secuencias_guardadas = 0
    fase = "espera"  # espera -> cuenta -> grabando
    buffer_actual: list[list[float]] = []
    t_referencia = 0.0
    enfocado = False
    frames_stop = 0          # frames seguidos con "dos palmas alzadas"
    indice_onset = 0         # inicio del gesto de parada: se recorta el buffer hasta aquí
    aviso = ""               # mensaje breve tras guardar (p. ej. secuencia muy corta)
    t_aviso = 0.0

    try:
        camara.abrir()
        crear_ventana(TITULO_VENTANA)
        while True:
            frame = camara.obtener_frame()
            frame = cv2.flip(frame, 1)
            resultado = seguidor.detectar(frame)
            seguidor.dibujar(frame, resultado)
            hay_cuerpo = resultado.hay_cuerpo
            ahora = time.time()

            color = (0, 0, 255) if fase == "grabando" else (VERDE_DETECCION if hay_cuerpo else ROJO)
            cv2.rectangle(frame, (8, 8), (frame.shape[1] - 8, frame.shape[0] - 8), color, 2)

            mensaje = f"[{persona}] {etiqueta} | {secuencias_guardadas}/{total_secuencias} | G: iniciar/terminar | Q: salir"
            cv2.putText(frame, mensaje, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if fase == "espera":
                texto = "Listo. Pulsa G para iniciar (con cuenta regresiva)." if hay_cuerpo else "Colócate frente a la cámara (medio cuerpo)."
                cv2.putText(frame, texto, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 180) if hay_cuerpo else (70, 200, 255), 2, cv2.LINE_AA)
                if aviso and ahora - t_aviso < 1.8:
                    cv2.putText(frame, aviso, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 80, 255), 2, cv2.LINE_AA)
            elif fase == "cuenta":
                restante = SEGUNDOS_CUENTA_REGRESIVA - (ahora - t_referencia)
                if restante <= 0:
                    fase = "grabando"
                    buffer_actual = []
                    frames_stop = 0
                    indice_onset = 0
                    t_referencia = ahora
                else:
                    cv2.putText(frame, str(int(restante) + 1), (frame.shape[1] // 2 - 30, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, VERDE_DETECCION, 6, cv2.LINE_AA)
            elif fase == "grabando":
                # "Dos palmas alzadas" = terminar sin teclado; se recorta ese gesto del buffer.
                if palmas_alzadas_abiertas(resultado):
                    if frames_stop == 0:
                        indice_onset = max(0, len(buffer_actual) - MARGEN_FRAMES_STOP)
                    frames_stop += 1
                else:
                    frames_stop = 0
                    if hay_cuerpo:
                        vector = normalizar_holistic(resultado.pose, resultado.mano_izq, resultado.mano_der)
                        buffer_actual.append(vector.tolist())
                transcurrido = ahora - t_referencia
                cv2.putText(frame, f"Grabando: {len(buffer_actual)} frames ({transcurrido:.1f}s) | alza ambas palmas (o G) para terminar", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 80, 255), 2, cv2.LINE_AA)
                if frames_stop >= FRAMES_CONFIRMACION_STOP:
                    buffer_actual = buffer_actual[:indice_onset]
                if frames_stop >= FRAMES_CONFIRMACION_STOP or transcurrido >= MAX_SEGUNDOS_GRABACION_GESTO:
                    guardado = _guardar_si_valida(ruta_dataset, etiqueta, buffer_actual, persona)
                    secuencias_guardadas += guardado
                    aviso = "" if guardado else "Secuencia muy corta, repite."
                    t_aviso = ahora
                    fase = "espera"
                    if secuencias_guardadas >= total_secuencias:
                        break

            cv2.imshow(TITULO_VENTANA, frame)
            tecla = cv2.waitKey(1) & 0xFF
            if not enfocado:
                enfocar_ventana(TITULO_VENTANA)
                enfocado = True
            if tecla in (27, ord("q")):
                break
            if tecla == ord("g"):
                if fase == "espera" and hay_cuerpo:
                    fase = "cuenta"
                    t_referencia = ahora
                elif fase == "grabando":
                    secuencias_guardadas += _guardar_si_valida(ruta_dataset, etiqueta, buffer_actual, persona)
                    fase = "espera"
                    if secuencias_guardadas >= total_secuencias:
                        break
    finally:
        camara.liberar()
        seguidor.cerrar()
        cv2.destroyAllWindows()


def _guardar_si_valida(ruta_dataset: Path, etiqueta: str, buffer_actual: list, persona: str) -> int:
    if len(buffer_actual) >= MIN_FRAMES_SECUENCIA:
        guardar_secuencia_jsonl(ruta_dataset, etiqueta, buffer_actual, persona)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Captura secuencias dinámicas (Holistic) para palabras.")
    parser.add_argument("--etiqueta", required=True, help="Palabra a capturar.")
    parser.add_argument("--muestras", type=int, default=60, help="Cantidad objetivo de secuencias.")
    parser.add_argument("--persona", default="desconocida", help="Quién ejecuta la seña (para generalización).")
    parser.add_argument("--dataset", type=Path, default=ARCHIVO_DATASET_DINAMICO, help="Ruta del JSONL destino.")
    args = parser.parse_args()

    capturar_secuencias(
        etiqueta=args.etiqueta,
        total_secuencias=args.muestras,
        ruta_dataset=args.dataset,
        persona=args.persona,
    )
    print(f"Captura de secuencias finalizada para: {args.etiqueta} (persona: {args.persona})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
