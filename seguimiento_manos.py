from dataclasses import dataclass
from typing import Any, List, Optional

import cv2
import mediapipe as mp

from configuracion import MANO_CANONICA
from normalizacion import espejar_mano


@dataclass
class ResultadoMano:
    landmarks: List[List[float]]
    caja: tuple[int, int, int, int]
    mano_dominante: str
    landmarks_mp: Any = None  # proto crudo de MediaPipe, para dibujar el esqueleto


class SeguimientoManos:
    def __init__(self, max_num_manos: int = 1, confianza_minima: float = 0.5) -> None:
        self.mp_manos = mp.solutions.hands
        self.mp_dibujo = mp.solutions.drawing_utils
        self.mp_estilos = mp.solutions.drawing_styles
        self.detector = self.mp_manos.Hands(
            static_image_mode=False,
            max_num_hands=max_num_manos,
            min_detection_confidence=confianza_minima,
            min_tracking_confidence=confianza_minima,
        )

    def detectar_manos(self, frame_bgr) -> list[ResultadoMano]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resultados = self.detector.process(frame_rgb)
        if not resultados.multi_hand_landmarks:
            return []

        alto, ancho = frame_bgr.shape[:2]
        manos_detectadas: list[ResultadoMano] = []

        for indice, hand_landmarks in enumerate(resultados.multi_hand_landmarks):
            puntos = []
            xs = []
            ys = []
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                z = landmark.z
                puntos.append([x, y, z])
                xs.append(int(x * ancho))
                ys.append(int(y * alto))

            etiqueta_mano = "desconocida"
            if resultados.multi_handedness and indice < len(resultados.multi_handedness):
                etiqueta_mano = resultados.multi_handedness[indice].classification[0].label.lower()

            # Canonicalización de quiralidad: si la mano no es la canónica, se espejan los
            # landmarks que alimentan al modelo (x -> 1-x). La caja y el proto de dibujo se
            # dejan en su posición real (el recuadro-gate y el esqueleto siguen bien ubicados).
            if etiqueta_mano != MANO_CANONICA.lower():
                puntos = espejar_mano(puntos)

            caja = (min(xs), min(ys), max(xs), max(ys))
            manos_detectadas.append(
                ResultadoMano(
                    landmarks=puntos,
                    caja=caja,
                    mano_dominante=etiqueta_mano,
                    landmarks_mp=hand_landmarks,
                )
            )

        return manos_detectadas

    def dibujar_landmarks(self, frame_bgr, resultados_manos: list[ResultadoMano]) -> None:
        """Dibuja el esqueleto de la mano (líneas + puntos) sobre el frame."""
        for resultado in resultados_manos:
            if resultado.landmarks_mp is None:
                continue
            self.mp_dibujo.draw_landmarks(
                frame_bgr,
                resultado.landmarks_mp,
                self.mp_manos.HAND_CONNECTIONS,
                self.mp_estilos.get_default_hand_landmarks_style(),
                self.mp_estilos.get_default_hand_connections_style(),
            )

    def cerrar(self) -> None:
        self.detector.close()
