from __future__ import annotations

from collections import deque

import numpy as np

from normalizacion import normalizar_landmarks
from secuencias import redimensionar_secuencia


class ExtractorCaracteristicas:
    def __init__(self, max_frames_buffer: int = 24) -> None:
        self.buffer_vectores = deque(maxlen=max_frames_buffer)

    def reiniciar(self) -> None:
        self.buffer_vectores.clear()

    def extraer_vector(self, landmarks: list[list[float]]) -> np.ndarray:
        vector = normalizar_landmarks(landmarks)
        self.buffer_vectores.append(vector)
        return vector

    def agregar_vector(self, vector: np.ndarray) -> None:
        """Encola un vector ya calculado (p. ej. Holistic) para el buffer dinámico."""
        self.buffer_vectores.append(np.asarray(vector, dtype=np.float32))

    def obtener_vector_suavizado(self) -> np.ndarray | None:
        if not self.buffer_vectores:
            return None
        return np.mean(np.array(self.buffer_vectores), axis=0)

    def obtener_secuencia(self) -> list[list[float]]:
        return [vector.tolist() for vector in self.buffer_vectores]

    def obtener_secuencia_redimensionada(self, longitud_objetivo: int) -> np.ndarray | None:
        if len(self.buffer_vectores) < max(4, longitud_objetivo // 2):
            return None
        return redimensionar_secuencia(self.obtener_secuencia(), longitud_objetivo)
