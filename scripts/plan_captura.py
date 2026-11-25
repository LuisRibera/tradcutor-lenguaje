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
    ETIQUETAS_BASE,
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


def comando_captura(tipo: str, etiqueta: str, faltan: int) -> str:
    if tipo == "palabra":
        return f'python scripts/capturar_secuencias.py --etiqueta "{etiqueta}" --persona <nombre> --muestras {faltan}'
    if tipo == "letra_dinamica":
        return f'python scripts/capturar_letras_dinamicas.py --etiqueta "{etiqueta}" --persona <nombre> --muestras {faltan}'
    return f'python scripts/capturar_muestras.py --etiqueta "{etiqueta}" --muestras {faltan}'


def generar_plan(ruta_estatico: Path, ruta_dinamico: Path, ruta_letras: Path) -> int:
    conteos_estaticos = cargar_conteos_estaticos(ruta_estatico)
    conteos_dinamicos = cargar_conteos_dinamicos(ruta_dinamico)
    conteos_letras = cargar_conteos_dinamicos(ruta_letras)
    conteos_por_tipo = {"palabra": conteos_dinamicos, "letra_dinamica": conteos_letras}

    pendientes = []
    for etiqueta in ETIQUETAS_BASE:
        tipo = obtener_tipo_etiqueta(etiqueta)
        conteos = conteos_por_tipo.get(tipo, conteos_estaticos)
        actual = conteos.get(etiqueta, 0)
        faltan = max(0, obtener_muestras_objetivo(etiqueta) - actual)
        if faltan > 0:
            pendientes.append((tipo, etiqueta, faltan))

    if not pendientes:
        print("No hay capturas pendientes para las clases objetivo.")
        return 0

    prioridades = {"palabra": 0, "letra_dinamica": 1, "letra": 2, "desconocida": 3}
    pendientes.sort(key=lambda item: (prioridades.get(item[0], 9), -item[2], item[1]))

    print("Plan de captura recomendado")
    print("")
    print("Orden sugerido:")
    for tipo, etiqueta, faltan in pendientes:
        print(f"- {etiqueta} ({tipo}): capturar {faltan}")

    print("")
    print("Comandos sugeridos:")
    for tipo, etiqueta, faltan in pendientes[:12]:
        print(comando_captura(tipo, etiqueta, faltan))
    return 0


if __name__ == "__main__":
    raise SystemExit(generar_plan(ARCHIVO_DATASET, ARCHIVO_DATASET_DINAMICO, ARCHIVO_DATASET_LETRAS_DINAMICAS))
