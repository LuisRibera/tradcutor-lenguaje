from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuracion import ARCHIVO_DATASET, CARPETA_DATOS, ETIQUETAS_ESTATICAS, ETIQUETAS_DINAMICAS


CABECERA_BASE = ["etiqueta"] + [f"f_{indice}" for indice in range(63)]


def preparar_archivo_salida(ruta_salida: Path) -> None:
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    if ruta_salida.exists():
        return
    with ruta_salida.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(CABECERA_BASE)


def copiar_dataset_csv(ruta_entrada: Path, ruta_salida: Path) -> None:
    preparar_archivo_salida(ruta_salida)
    with ruta_entrada.open("r", newline="", encoding="utf-8") as archivo_entrada:
        lector = csv.reader(archivo_entrada)
        cabecera = next(lector, None)
        if not cabecera:
            raise ValueError("El archivo CSV de entrada está vacío.")

        with ruta_salida.open("a", newline="", encoding="utf-8") as archivo_salida:
            escritor = csv.writer(archivo_salida)
            for fila in lector:
                if len(fila) < 64:
                    continue
                escritor.writerow(fila[:64])


def crear_dataset_demo(ruta_salida: Path) -> None:
    preparar_archivo_salida(ruta_salida)
    print("Dataset estático base creado. Debes agregar muestras reales para entrenar el modelo.")
    print(f"Etiquetas estáticas objetivo: {len(ETIQUETAS_ESTATICAS)}")
    for etiqueta in ETIQUETAS_ESTATICAS:
        print(f"- {etiqueta}")
    print("")
    print(f"Etiquetas dinámicas objetivo: {len(ETIQUETAS_DINAMICAS)}")
    for etiqueta in ETIQUETAS_DINAMICAS:
        print(f"- {etiqueta}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara un dataset de señas en formato CSV para clases estáticas.")
    parser.add_argument("--entrada", type=Path, help="Ruta a un CSV de entrada compatible.")
    parser.add_argument("--salida", type=Path, default=ARCHIVO_DATASET, help="Ruta del CSV preparado.")
    args = parser.parse_args()

    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)

    if args.entrada:
        copiar_dataset_csv(args.entrada, args.salida)
        print(f"Dataset preparado en: {args.salida}")
    else:
        crear_dataset_demo(args.salida)
        print(f"Archivo base listo en: {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
