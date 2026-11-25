from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuracion import (
    ARCHIVO_DATASET,
    ARCHIVO_DATASET_DINAMICO,
    ARCHIVO_DATASET_LETRAS_DINAMICAS,
    ETIQUETAS_DINAMICAS,
    ETIQUETAS_ESTATICAS,
    LETRAS_DINAMICAS,
    obtener_muestras_objetivo,
    obtener_tipo_etiqueta,
)
from secuencias import cargar_secuencias_jsonl


def cargar_conteos_estaticos(ruta_dataset: Path) -> Counter:
    conteos: Counter[str] = Counter()
    if not ruta_dataset.exists():
        return conteos

    with ruta_dataset.open("r", newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        for fila in lector:
            etiqueta = (fila.get("etiqueta") or "").strip()
            if etiqueta:
                conteos[etiqueta] += 1
    return conteos


def cargar_conteos_dinamicos(ruta_dataset: Path) -> Counter:
    conteos: Counter[str] = Counter()
    for registro in cargar_secuencias_jsonl(ruta_dataset):
        etiqueta = (registro.get("etiqueta") or "").strip()
        if etiqueta:
            conteos[etiqueta] += 1
    return conteos


def imprimir_bloque(titulo: str, etiquetas: list[str], conteos: Counter) -> list[tuple[str, str, int]]:
    faltantes = []
    print(titulo)
    for etiqueta in etiquetas:
        actual = conteos.get(etiqueta, 0)
        objetivo = obtener_muestras_objetivo(etiqueta)
        faltan = max(0, objetivo - actual)
        tipo = obtener_tipo_etiqueta(etiqueta)
        estado = "OK" if faltan == 0 else f"Faltan {faltan}"
        print(f"- [{tipo}] {etiqueta}: {actual}/{objetivo} -> {estado}")
        if faltan > 0:
            faltantes.append((tipo, etiqueta, faltan))
    print("")
    return faltantes


def imprimir_reporte(ruta_estatico: Path, ruta_dinamico: Path, ruta_letras: Path) -> int:
    conteos_estaticos = cargar_conteos_estaticos(ruta_estatico)
    conteos_dinamicos = cargar_conteos_dinamicos(ruta_dinamico)
    conteos_letras = cargar_conteos_dinamicos(ruta_letras)

    print("Auditoría de dataset")
    print(f"Dataset estático: {ruta_estatico}")
    print(f"Dataset palabras: {ruta_dinamico}")
    print(f"Dataset letras con movimiento: {ruta_letras}")
    print(f"Muestras estáticas totales: {sum(conteos_estaticos.values())}")
    print(f"Secuencias de palabras totales: {sum(conteos_dinamicos.values())}")
    print(f"Secuencias de letras con movimiento totales: {sum(conteos_letras.values())}")
    print("")

    faltantes = []
    faltantes.extend(imprimir_bloque("Letras estáticas:", ETIQUETAS_ESTATICAS, conteos_estaticos))
    faltantes.extend(imprimir_bloque("Letras con movimiento:", LETRAS_DINAMICAS, conteos_letras))
    faltantes.extend(imprimir_bloque("Palabras:", ETIQUETAS_DINAMICAS, conteos_dinamicos))

    if faltantes:
        print("Pendientes prioritarios:")
        for tipo, etiqueta, faltan in sorted(faltantes, key=lambda item: (-item[2], item[0], item[1]))[:15]:
            print(f"- {etiqueta} ({tipo}): faltan {faltan}")
        return 1

    print("Todas las clases objetivo alcanzaron su meta de muestras.")
    return 0


if __name__ == "__main__":
    raise SystemExit(imprimir_reporte(ARCHIVO_DATASET, ARCHIVO_DATASET_DINAMICO, ARCHIVO_DATASET_LETRAS_DINAMICAS))
