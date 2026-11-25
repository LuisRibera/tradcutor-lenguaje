from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional

import cv2
import mediapipe as mp

from configuracion import (
    MARGEN_ALTURA_STOP,
    MIN_DEDOS_EXTENDIDOS_STOP,
    POSE_HOMBROS,
    VERDE_DETECCION,
)

# Pares (punta, PIP) de los cuatro dedos largos en MediaPipe Hands. Un dedo está extendido
# si su punta queda más lejos de la muñeca (landmark 0) que su articulación PIP.
_DEDOS_PUNTA_PIP = ((8, 6), (12, 10), (16, 14), (20, 18))


@dataclass
class ResultadoHolistic:
    pose: Optional[List[List[float]]]
    mano_izq: Optional[List[List[float]]]
    mano_der: Optional[List[List[float]]]
    # Protos crudos de MediaPipe, solo para dibujar el esqueleto (líneas + puntos).
    pose_mp: Any = None
    mano_izq_mp: Any = None
    mano_der_mp: Any = None

    @property
    def hay_cuerpo(self) -> bool:
        return self.pose is not None

    @property
    def hay_manos(self) -> bool:
        return self.mano_izq is not None or self.mano_der is not None


class SeguimientoHolistic:
    """Envuelve MediaPipe Holistic para el reconocimiento dinámico de frases.

    Devuelve pose (torso + brazos) y ambas manos por frame. La cara se ignora a
    propósito: no aporta a estas señas y añade ruido y cómputo.
    """

    def __init__(self, complejidad: int = 1, confianza_minima: float = 0.5) -> None:
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "holistic"):
            raise RuntimeError(
                "Este mediapipe no incluye la API 'solutions.holistic'. Usa el entorno "
                "documentado: Python 3.11 + mediapipe==0.10.14 (ver requirements.txt). "
                "Las versiones recientes (>=0.10.30) eliminaron las soluciones legacy."
            )
        self.mp_holistic = mp.solutions.holistic
        self.mp_dibujo = mp.solutions.drawing_utils
        self.mp_estilos = mp.solutions.drawing_styles
        # Conexiones de pose SIN la cara: solo torso, brazos y piernas (índices > 10). Los puntos
        # de ojos/orejas/boca (1-10) no se usan como features; solo ensucian visualmente.
        self._pose_conexiones_cuerpo = frozenset(
            (a, b) for (a, b) in self.mp_holistic.POSE_CONNECTIONS if a > 10 and b > 10
        )
        self.detector = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=complejidad,
            refine_face_landmarks=False,
            min_detection_confidence=confianza_minima,
            min_tracking_confidence=confianza_minima,
        )

    @staticmethod
    def _a_lista(landmarks) -> Optional[List[List[float]]]:
        if landmarks is None:
            return None
        return [[punto.x, punto.y, punto.z] for punto in landmarks.landmark]

    def detectar(self, frame_bgr) -> ResultadoHolistic:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resultados = self.detector.process(frame_rgb)
        return ResultadoHolistic(
            pose=self._a_lista(resultados.pose_landmarks),
            mano_izq=self._a_lista(resultados.left_hand_landmarks),
            mano_der=self._a_lista(resultados.right_hand_landmarks),
            pose_mp=resultados.pose_landmarks,
            mano_izq_mp=resultados.left_hand_landmarks,
            mano_der_mp=resultados.right_hand_landmarks,
        )

    def dibujar(self, frame_bgr, resultado: ResultadoHolistic) -> None:
        """Dibuja el esqueleto (torso + brazos + ambas manos) sobre el frame, sin los puntos de
        la cara: solo se dibujan las conexiones del cuerpo (líneas), no los puntos de pose."""
        if resultado.pose_mp is not None:
            self.mp_dibujo.draw_landmarks(
                frame_bgr,
                resultado.pose_mp,
                self._pose_conexiones_cuerpo,
                landmark_drawing_spec=None,  # sin puntos de pose (quita los nodos de ojos/boca)
                connection_drawing_spec=self.mp_dibujo.DrawingSpec(color=VERDE_DETECCION, thickness=2),
            )
        for mano_mp in (resultado.mano_izq_mp, resultado.mano_der_mp):
            if mano_mp is not None:
                self.mp_dibujo.draw_landmarks(
                    frame_bgr,
                    mano_mp,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_estilos.get_default_hand_landmarks_style(),
                    self.mp_estilos.get_default_hand_connections_style(),
                )

    def cerrar(self) -> None:
        self.detector.close()


def _distancia_xy(a: List[float], b: List[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dedos_extendidos(mano: List[List[float]]) -> int:
    """Cuenta los cuatro dedos largos extendidos: punta más lejos de la muñeca que el PIP."""
    muñeca = mano[0]
    extendidos = 0
    for punta, pip in _DEDOS_PUNTA_PIP:
        if _distancia_xy(mano[punta], muñeca) > _distancia_xy(mano[pip], muñeca):
            extendidos += 1
    return extendidos


def palmas_alzadas_abiertas(resultado: ResultadoHolistic) -> bool:
    """¿El firmante alzó las dos palmas abiertas por encima de los hombros?

    Se usa como señal de parada sin teclado en la captura de palabras: es un gesto
    deliberado (ambas manos, abiertas y en alto) que no se confunde con las señas del
    vocabulario, que se hacen a la altura del pecho/cara y muchas con una sola mano.
    """
    if resultado.pose is None or resultado.mano_izq is None or resultado.mano_der is None:
        return False

    izq, der = POSE_HOMBROS
    altura_hombros = (resultado.pose[izq][1] + resultado.pose[der][1]) / 2
    umbral_altura = altura_hombros - MARGEN_ALTURA_STOP  # "arriba" = y menor

    for mano in (resultado.mano_izq, resultado.mano_der):
        if mano[0][1] > umbral_altura:  # muñeca no está por encima de los hombros
            return False
        if _dedos_extendidos(mano) < MIN_DEDOS_EXTENDIDOS_STOP:
            return False
    return True
