from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def redimensionar_secuencia(vectores: list[list[float]] | list[np.ndarray], longitud_objetivo: int) -> np.ndarray:
    if not vectores:
        raise ValueError("La secuencia no puede estar vacía.")

    matriz = np.array(vectores, dtype=np.float32)
    if matriz.ndim != 2:
        raise ValueError("La secuencia debe ser una matriz de frames por características.")

    if len(matriz) == longitud_objetivo:
        return matriz.flatten()

    indices_origen = np.linspace(0, len(matriz) - 1, num=longitud_objetivo)
    indices_redondeados = np.clip(np.round(indices_origen).astype(int), 0, len(matriz) - 1)
    matriz_redimensionada = matriz[indices_redondeados]
    return matriz_redimensionada.flatten()


def guardar_secuencia_jsonl(
    ruta_archivo: Path,
    etiqueta: str,
    vectores: list[list[float]],
    persona: str = "desconocida",
) -> None:
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
    registro = {
        "etiqueta": etiqueta,
        "persona": persona,
        "longitud": len(vectores),
        "frames": vectores,
    }
    with ruta_archivo.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")


def cargar_secuencias_jsonl(ruta_archivo: Path) -> list[dict]:
    if not ruta_archivo.exists():
        return []

    secuencias = []
    with ruta_archivo.open("r", encoding="utf-8") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea:
                continue
            secuencias.append(json.loads(linea))
    return secuencias
