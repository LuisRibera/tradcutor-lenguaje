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
    ANCHO_PANEL,
    ANCHO_VENTANA,
    ARCHIVO_DATASET_LETRAS_DINAMICAS,
    INDICE_CAMARA,
    MAX_SEGUNDOS_GRABACION_GESTO,
    SEGUNDOS_CUENTA_REGRESIVA,
    VERDE_DETECCION,
)
from interfaz import InterfazTraductor
from normalizacion import normalizar_landmarks
from secuencias import guardar_secuencia_jsonl
from seguimiento_manos import SeguimientoManos
from ventana import crear_ventana, enfocar_ventana

TITULO_VENTANA = "Captura de letras con movimiento"
ROJO = (0, 80, 220)
MIN_FRAMES_SECUENCIA = 5


def esta_dentro_del_roi(caja: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = caja
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    centro_x = (x1 + x2) // 2
    centro_y = (y1 + y2) // 2
    return roi_x1 <= centro_x <= roi_x2 and roi_y1 <= centro_y <= roi_y2


def capturar_letras(etiqueta: str, total_secuencias: int, ruta_dataset: Path, persona: str) -> None:
    camara = GestorCamara(INDICE_CAMARA, ANCHO_VENTANA, ALTO_VENTANA)
    seguidor = SeguimientoManos()
    interfaz = InterfazTraductor(ancho_panel=ANCHO_PANEL)

    secuencias_guardadas = 0
    fase = "espera"  # espera -> cuenta -> grabando
    buffer_actual: list[list[float]] = []
    t_referencia = 0.0
    enfocado = False

    try:
        camara.abrir()
        crear_ventana(TITULO_VENTANA)
        while True:
            frame = camara.obtener_frame()
            frame = cv2.flip(frame, 1)
            roi = interfaz.calcular_roi(frame)
            manos = seguidor.detectar_manos(frame)
            seguidor.dibujar_landmarks(frame, manos)
            mano_valida = next((mano for mano in manos if esta_dentro_del_roi(mano.caja, roi)), None)
            ahora = time.time()

            color = ROJO
            if fase == "grabando":
                color = (0, 0, 255)
            elif mano_valida is not None:
                color = VERDE_DETECCION
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            mensaje = f"[{persona}] {etiqueta} | {secuencias_guardadas}/{total_secuencias} | G: iniciar/terminar | Q: salir"
            cv2.putText(frame, mensaje, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if fase == "espera":
                texto = "Listo. Pulsa G para iniciar (con cuenta regresiva)." if mano_valida else "Coloca la mano en el recuadro."
                cv2.putText(frame, texto, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 180) if mano_valida else (70, 200, 255), 2, cv2.LINE_AA)
            elif fase == "cuenta":
                restante = SEGUNDOS_CUENTA_REGRESIVA - (ahora - t_referencia)
                if restante <= 0:
                    fase = "grabando"
                    buffer_actual = []
                    t_referencia = ahora
                else:
                    cv2.putText(frame, str(int(restante) + 1), (x1 + 20, y1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 3.0, VERDE_DETECCION, 6, cv2.LINE_AA)
            elif fase == "grabando":
                if mano_valida is not None:
                    buffer_actual.append(normalizar_landmarks(mano_valida.landmarks).tolist())
                transcurrido = ahora - t_referencia
                cv2.putText(frame, f"Grabando: {len(buffer_actual)} frames ({transcurrido:.1f}s) | G para terminar", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 80, 255), 2, cv2.LINE_AA)
                if transcurrido >= MAX_SEGUNDOS_GRABACION_GESTO:
                    secuencias_guardadas += _guardar_si_valida(ruta_dataset, etiqueta, buffer_actual, persona)
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
                if fase == "espera" and mano_valida is not None:
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
    parser = argparse.ArgumentParser(description="Captura secuencias de una mano para las letras con movimiento (J, K, Q, X, Z, Ñ).")
    parser.add_argument("--etiqueta", required=True, help="Letra con movimiento a capturar.")
    parser.add_argument("--muestras", type=int, default=50, help="Cantidad objetivo de secuencias.")
    parser.add_argument("--persona", default="desconocida", help="Quién ejecuta la seña (para generalización).")
    parser.add_argument("--dataset", type=Path, default=ARCHIVO_DATASET_LETRAS_DINAMICAS, help="Ruta del JSONL destino.")
    args = parser.parse_args()

    capturar_letras(
        etiqueta=args.etiqueta,
        total_secuencias=args.muestras,
        ruta_dataset=args.dataset,
        persona=args.persona,
    )
    print(f"Captura de letras finalizada para: {args.etiqueta} (persona: {args.persona})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
