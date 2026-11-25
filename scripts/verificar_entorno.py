from __future__ import annotations

import sys
from pathlib import Path

import cv2


def verificar_version_python() -> bool:
    version = sys.version_info
    compatible = (version.major, version.minor) == (3, 11)
    print(f"Python detectado: {version.major}.{version.minor}.{version.micro}")
    print("Compatible con el proyecto: SI" if compatible else "Compatible con el proyecto: NO")
    return compatible


def verificar_imports() -> bool:
    modulos = ["numpy", "cv2", "mediapipe", "sklearn", "joblib", "pyttsx3"]
    exito = True
    for nombre in modulos:
        try:
            __import__(nombre)
            print(f"Import correcto: {nombre}")
        except Exception as error:
            exito = False
            print(f"Error importando {nombre}: {error}")
    return exito


def verificar_camara() -> bool:
    captura = cv2.VideoCapture(0)
    disponible = bool(captura and captura.isOpened())
    if disponible:
        exito, _ = captura.read()
        disponible = disponible and exito
    captura.release()
    print("Camara disponible: SI" if disponible else "Camara disponible: NO")
    return disponible


def verificar_carpetas() -> bool:
    rutas = [Path("data"), Path("models"), Path("scripts")]
    exito = True
    for ruta in rutas:
        if ruta.exists():
            print(f"Carpeta OK: {ruta}")
        else:
            exito = False
            print(f"Falta carpeta: {ruta}")
    return exito


def main() -> int:
    resultados = [
        verificar_version_python(),
        verificar_imports(),
        verificar_camara(),
        verificar_carpetas(),
    ]
    return 0 if all(resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
