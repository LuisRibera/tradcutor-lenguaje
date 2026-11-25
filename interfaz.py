from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from configuracion import MARGEN_ROI_X, MARGEN_ROI_Y, ROI_ALTO, ROI_ANCHO, VERDE_DETECCION

# Pocos colores: verde de detección para lo reconocido, gris para etiquetas, blanco para texto.
GRIS = (170, 170, 170)
BLANCO = (245, 245, 245)
FONDO_PANEL = 28
COLOR_BOTON = (70, 70, 70)
COLOR_BOTON_TEXTO = (245, 245, 245)

# Botones del panel (todos del mismo color, texto contrastante, sin íconos).
BOTONES = [("Leer", "leer"), ("Espacio", "espacio"), ("Borrar", "borrar"), ("Limpiar", "limpiar")]

# Fuente TrueType para texto con acentos/ñ/¿? (cv2.putText no los dibuja: salen como "?").
_RUTA_FUENTE = r"C:\Windows\Fonts\arial.ttf"
_cache_fuentes: dict[int, ImageFont.FreeTypeFont] = {}


def _fuente(px: int) -> ImageFont.FreeTypeFont:
    if px not in _cache_fuentes:
        try:
            _cache_fuentes[px] = ImageFont.truetype(_RUTA_FUENTE, px)
        except OSError:
            _cache_fuentes[px] = ImageFont.load_default()
    return _cache_fuentes[px]


def _texto_unicode(imagen_bgr, texto, x, y, escala, color_bgr) -> None:
    """Dibuja texto Unicode (ñ, acentos, ¿?) sobre una imagen BGR usando Pillow. `y` es la
    línea base, igual que cv2.putText, para conservar la maquetación."""
    if not texto:
        return
    px = max(12, int(round(escala * 33)))
    lienzo = Image.fromarray(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(lienzo).text((x, y), texto, font=_fuente(px), fill=(color_bgr[2], color_bgr[1], color_bgr[0]), anchor="ls")
    imagen_bgr[:, :] = cv2.cvtColor(np.asarray(lienzo), cv2.COLOR_RGB2BGR)


class InterfazTraductor:
    def __init__(self, ancho_panel: int) -> None:
        self.ancho_panel = ancho_panel
        # Rectángulos de los botones en coordenadas de la imagen combinada (video + panel).
        self.botones: dict[str, tuple[int, int, int, int]] = {}

    def calcular_roi(self, frame) -> tuple[int, int, int, int]:
        """Recuadro de reconocimiento: cuadrado centrado y amplio."""
        alto, ancho = frame.shape[:2]
        lado = min(ROI_ANCHO, ROI_ALTO, alto - 2 * MARGEN_ROI_Y, ancho - 2 * MARGEN_ROI_X)
        lado = max(280, lado)
        x1 = (ancho - lado) // 2
        y1 = (alto - lado) // 2
        return x1, y1, x1 + lado, y1 + lado

    def combinar_vista(
        self,
        frame,
        roi: tuple[int, int, int, int] | None,
        texto_oracion: str,
        posicion_cursor: int,
        etiqueta_reconocida: str,
        progreso_validacion: float = 0.0,
        etiqueta_validacion: str = "",
    ):
        alto, ancho = frame.shape[:2]
        panel = np.full((alto, self.ancho_panel, 3), FONDO_PANEL, dtype=np.uint8)

        if roi is not None:
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), VERDE_DETECCION, 3)
            cv2.putText(frame, "Zona de reconocimiento", (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, VERDE_DETECCION, 2, cv2.LINE_AA)

        # --- Reconocimiento actual ---
        self._texto(panel, "Reconociendo", 28, 40, 0.7, GRIS)
        _texto_unicode(panel, etiqueta_reconocida or "-", 28, 82, 1.0, VERDE_DETECCION)

        # --- Barra de validación (sin porcentaje ni texto de cooldown) ---
        if progreso_validacion > 0.0:
            _texto_unicode(panel, f"Validando {etiqueta_validacion}", 28, 118, 0.6, VERDE_DETECCION)
            self._barra(panel, 28, 128, self.ancho_panel - 56, 12, progreso_validacion, VERDE_DETECCION)

        # --- Oración editable con cursor ---
        self._texto(panel, "Oracion", 28, 185, 0.7, GRIS)
        self._texto_oracion(panel, texto_oracion, posicion_cursor, 28, 218, 22)

        # --- Botones (mismo color, texto contrastante) ---
        self.botones = {}
        alto_boton, gap = 46, 12
        total = len(BOTONES) * alto_boton + (len(BOTONES) - 1) * gap
        y_boton = alto - 70 - total
        for texto, accion in BOTONES:
            self._boton(panel, texto, accion, 28, y_boton, self.ancho_panel - 56, alto_boton, ancho)
            y_boton += alto_boton + gap

        return np.hstack((frame, panel))

    def boton_en(self, x: int, y: int) -> str | None:
        """Acción del botón que contiene el punto (x, y) de la imagen combinada, o None."""
        for accion, (bx1, by1, bx2, by2) in self.botones.items():
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                return accion
        return None

    def _boton(self, panel, texto, accion, x, y, ancho_b, alto_b, desplazamiento_x) -> None:
        cv2.rectangle(panel, (x, y), (x + ancho_b, y + alto_b), COLOR_BOTON, -1)
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx = x + (ancho_b - tw) // 2
        ty = y + (alto_b + th) // 2
        cv2.putText(panel, texto, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BOTON_TEXTO, 2, cv2.LINE_AA)
        # Guardar el rectángulo desplazado al espacio de la imagen combinada (video a la izquierda).
        self.botones[accion] = (desplazamiento_x + x, y, desplazamiento_x + x + ancho_b, y + alto_b)

    def _barra(self, imagen, x, y, ancho_b, alto_b, fraccion, color) -> None:
        cv2.rectangle(imagen, (x, y), (x + ancho_b, y + alto_b), (70, 70, 70), 1)
        relleno = int(ancho_b * max(0.0, min(1.0, fraccion)))
        if relleno > 0:
            cv2.rectangle(imagen, (x, y), (x + relleno, y + alto_b), color, -1)

    def _texto(self, imagen, texto, x, y, escala, color) -> None:
        cv2.putText(imagen, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2, cv2.LINE_AA)

    def _texto_oracion(self, imagen, texto, cursor, x, y, max_chars) -> None:
        """Dibuja la oración de izquierda a derecha, ajustando por ancho, con un cursor visible."""
        visible = texto[:cursor] + "|" + texto[cursor:]
        cursor_y = y
        for inicio in range(0, len(visible), max_chars):
            linea = visible[inicio : inicio + max_chars]
            _texto_unicode(imagen, linea, x, cursor_y, 0.7, BLANCO)
            cursor_y += 30
        if not texto:
            _texto_unicode(imagen, "|", x, y, 0.7, BLANCO)
