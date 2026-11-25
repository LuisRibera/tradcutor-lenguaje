"""Utilidades de ventana para la captura.

Abre la ventana de OpenCV y, en Windows, le da el foco del teclado. Sin esto, si la
captura se lanza desde el menú guiado (que pide datos con input() en la terminal), el
foco se queda en la terminal y la tecla G se escribe ahí en vez de guardar la muestra.
"""

from __future__ import annotations

import cv2

try:
    import ctypes

    _user32 = ctypes.windll.user32
except (ImportError, AttributeError, OSError):  # no es Windows o no hay user32
    _user32 = None


def crear_ventana(nombre: str) -> None:
    cv2.namedWindow(nombre)


def enfocar_ventana(nombre: str) -> None:
    """Trae la ventana al frente y le da el foco del teclado (solo Windows).

    En otros sistemas no hace nada; ahí la ventana ya toma el foco al abrirse.
    """
    if _user32 is None:
        return
    hwnd = _user32.FindWindowW(None, nombre)
    if hwnd:
        _user32.SetForegroundWindow(hwnd)


def tamano_pantalla() -> tuple[int, int]:
    """Resolución de la pantalla (ancho, alto) en Windows; si no se puede, un valor seguro
    para laptops. Sirve para escalar la ventana y que no se salga del monitor."""
    if _user32 is not None:
        try:
            ancho = int(_user32.GetSystemMetrics(0))
            alto = int(_user32.GetSystemMetrics(1))
            if ancho > 0 and alto > 0:
                return ancho, alto
        except (AttributeError, OSError):
            pass
    return 1366, 768
