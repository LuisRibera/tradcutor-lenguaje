from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camara import GestorCamara
from configuracion import ALTO_VENTANA, ANCHO_PANEL, ANCHO_VENTANA, ARCHIVO_DATASET, INDICE_CAMARA, VERDE_DETECCION
from interfaz import InterfazTraductor
from normalizacion import normalizar_landmarks
from seguimiento_manos import SeguimientoManos
from ventana import crear_ventana, enfocar_ventana


CABECERA_BASE = ["etiqueta"] + [f"f_{indice}" for indice in range(63)]
TITULO_VENTANA = "Captura de muestras"


def preparar_dataset(ruta_dataset: Path) -> None:
    ruta_dataset.parent.mkdir(parents=True, exist_ok=True)
    if ruta_dataset.exists():
        return
    with ruta_dataset.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(CABECERA_BASE)


def esta_dentro_del_roi(caja: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = caja
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    centro_x = (x1 + x2) // 2
    centro_y = (y1 + y2) // 2
    return roi_x1 <= centro_x <= roi_x2 and roi_y1 <= centro_y <= roi_y2


def guardar_muestra(ruta_dataset: Path, etiqueta: str, vector) -> None:
    with ruta_dataset.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([etiqueta] + [float(valor) for valor in vector])


def capturar_muestras(etiqueta: str, total_muestras: int, ruta_dataset: Path, intervalo: float) -> None:
    preparar_dataset(ruta_dataset)
    camara = GestorCamara(INDICE_CAMARA, ANCHO_VENTANA, ALTO_VENTANA)
    seguidor = SeguimientoManos()
    interfaz = InterfazTraductor(ancho_panel=ANCHO_PANEL)
    ultimo_guardado = 0.0
    capturadas = 0
    enfocado = False

    try:
        camara.abrir()
        crear_ventana(TITULO_VENTANA)
        while True:
            frame = camara.obtener_frame()
            frame = cv2.flip(frame, 1)
            alto_frame = frame.shape[0]
            roi = interfaz.calcular_roi(frame)
            manos = seguidor.detectar_manos(frame)
            seguidor.dibujar_landmarks(frame, manos)

            mensaje = f"Etiqueta: {etiqueta} | Capturadas: {capturadas}/{total_muestras} | G: guardar | Q: salir"
            mano_valida = None
            for mano in manos:
                if esta_dentro_del_roi(mano.caja, roi):
                    mano_valida = mano
                    break

            color = VERDE_DETECCION if mano_valida else (0, 80, 220)
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, mensaje, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Palma hacia la camara (misma orientacion siempre)", (20, alto_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, VERDE_DETECCION, 2, cv2.LINE_AA)

            if mano_valida is None:
                cv2.putText(frame, "Coloca la mano dentro del recuadro.", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 200, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Mano detectada. Presiona G para guardar muestra.", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 180), 2, cv2.LINE_AA)

            cv2.imshow(TITULO_VENTANA, frame)
            tecla = cv2.waitKey(1) & 0xFF
            if not enfocado:
                enfocar_ventana(TITULO_VENTANA)
                enfocado = True

            if tecla in (27, ord("q")):
                break

            if tecla == ord("g") and mano_valida is not None:
                instante = time.time()
                if instante - ultimo_guardado >= intervalo:
                    vector = normalizar_landmarks(mano_valida.landmarks)
                    guardar_muestra(ruta_dataset, etiqueta, vector)
                    capturadas += 1
                    ultimo_guardado = instante
                    if capturadas >= total_muestras:
                        break
    finally:
        camara.liberar()
        seguidor.cerrar()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Captura muestras estáticas etiquetadas para el traductor de señas.")
    parser.add_argument("--etiqueta", required=True, help="Etiqueta o clase estática a capturar.")
    parser.add_argument("--muestras", type=int, default=100, help="Cantidad objetivo de muestras.")
    parser.add_argument("--dataset", type=Path, default=ARCHIVO_DATASET, help="Ruta del CSV destino.")
    parser.add_argument("--intervalo", type=float, default=0.15, help="Segundos mínimos entre muestras guardadas.")
    args = parser.parse_args()

    capturar_muestras(
        etiqueta=args.etiqueta,
        total_muestras=args.muestras,
        ruta_dataset=args.dataset,
        intervalo=args.intervalo,
    )
    print(f"Captura finalizada para la etiqueta: {args.etiqueta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
