"""Lanzador guiado de captura: menú interactivo para capturar sin teclear comandos largos.

Pregunta una vez quién ejecuta las señas (persona) y, en bucle, permite capturar letras estáticas,
letras con movimiento o palabras, llamando al script de captura correcto. Solo captura (no entrena).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capturar_letras_dinamicas import capturar_letras
from capturar_muestras import capturar_muestras
from capturar_secuencias import capturar_secuencias
from configuracion import (
    ARCHIVO_DATASET,
    ARCHIVO_DATASET_DINAMICO,
    ARCHIVO_DATASET_LETRAS_DINAMICAS,
    LETRAS_DINAMICAS,
    LETRAS_ESTATICAS,
    MUESTRAS_OBJETIVO_LETRA,
    MUESTRAS_OBJETIVO_LETRA_DINAMICA,
    MUESTRAS_OBJETIVO_PALABRA,
    PALABRAS,
)


def _elegir(titulo: str, opciones: list[str]) -> str | None:
    print(f"\n{titulo}")
    for indice, opcion in enumerate(opciones, start=1):
        print(f"  {indice}. {opcion}")
    respuesta = input("Elige un número (o escribe la etiqueta, Enter para cancelar): ").strip()
    if not respuesta:
        return None
    if respuesta.isdigit() and 1 <= int(respuesta) <= len(opciones):
        return opciones[int(respuesta) - 1]
    return respuesta


def _pedir_entero(mensaje: str, defecto: int) -> int:
    respuesta = input(f"{mensaje} [{defecto}]: ").strip()
    return int(respuesta) if respuesta.isdigit() else defecto


def ejecutar() -> int:
    print("Lanzador guiado de captura")
    persona = input("¿Quién ejecuta las señas? (nombre): ").strip() or "desconocida"

    acciones = {
        "1": "Letra estática",
        "2": "Letra con movimiento (J, K, Q, X, Z, Ñ)",
        "3": "Palabra",
    }
    while True:
        print(f"\n=== Persona: {persona} ===")
        for clave, texto in acciones.items():
            print(f"  {clave}. {texto}")
        print("  q. Salir")
        opcion = input("Opción: ").strip().lower()

        if opcion == "q":
            print("Fin de la captura.")
            return 0

        if opcion == "1":
            etiqueta = _elegir("Letras estáticas:", LETRAS_ESTATICAS)
            if etiqueta:
                muestras = _pedir_entero("Muestras", MUESTRAS_OBJETIVO_LETRA)
                capturar_muestras(etiqueta, muestras, ARCHIVO_DATASET, 0.15)
        elif opcion == "2":
            etiqueta = _elegir("Letras con movimiento:", LETRAS_DINAMICAS)
            if etiqueta:
                muestras = _pedir_entero("Muestras", MUESTRAS_OBJETIVO_LETRA_DINAMICA)
                capturar_letras(etiqueta, muestras, ARCHIVO_DATASET_LETRAS_DINAMICAS, persona)
        elif opcion == "3":
            etiqueta = _elegir("Palabras:", PALABRAS)
            if etiqueta:
                muestras = _pedir_entero("Muestras", MUESTRAS_OBJETIVO_PALABRA)
                capturar_secuencias(etiqueta, muestras, ARCHIVO_DATASET_DINAMICO, persona)
        else:
            print("Opción no reconocida.")


if __name__ == "__main__":
    raise SystemExit(ejecutar())
