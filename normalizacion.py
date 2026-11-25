from __future__ import annotations

import numpy as np

from configuracion import NUM_FEATURES_DINAMICO, POSE_INDICES_TORSO


def normalizar_landmarks(landmarks: list[list[float]]) -> np.ndarray:
    matriz = np.array(landmarks, dtype=np.float32)
    if matriz.shape != (21, 3):
        raise ValueError("Se esperaban 21 landmarks con coordenadas x, y, z.")

    punto_base = matriz[0].copy()
    matriz[:, :2] -= punto_base[:2]

    distancia_referencia = np.linalg.norm(matriz[9, :2] - matriz[0, :2])
    if distancia_referencia > 0:
        matriz[:, :2] /= distancia_referencia

    return matriz.flatten()


def normalizar_holistic(
    pose: list[list[float]] | None,
    mano_izq: list[list[float]] | None,
    mano_der: list[list[float]] | None,
) -> np.ndarray:
    """Vector dinámico invariante a la persona: torso superior + ambas manos.

    Centra en el punto medio de los hombros y escala por el ancho de hombros, de modo
    que el tamaño y la posición de cada persona se cancelan pero se preserva la posición
    relativa de las manos respecto al cuerpo. Sin pose no hay marco de referencia, así que
    el frame se descarta como ceros.
    """
    if pose is None:
        return np.zeros(NUM_FEATURES_DINAMICO, dtype=np.float32)

    pose = np.array(pose, dtype=np.float32)
    hombro_izq = pose[11]
    hombro_der = pose[12]
    centro = (hombro_izq + hombro_der) / 2.0
    escala = float(np.linalg.norm(hombro_izq[:2] - hombro_der[:2]))
    if escala <= 1e-6:
        escala = 1.0

    def _normalizar(puntos) -> np.ndarray:
        return (np.array(puntos, dtype=np.float32) - centro) / escala

    partes = [_normalizar(pose[POSE_INDICES_TORSO])]
    for mano in (mano_izq, mano_der):
        if mano is None:
            partes.append(np.zeros((21, 3), dtype=np.float32))
        else:
            partes.append(_normalizar(mano))

    return np.concatenate([parte.flatten() for parte in partes]).astype(np.float32)


def espejar_mano(landmarks: list[list[float]]) -> list[list[float]]:
    """Refleja una mano en el eje horizontal (coordenadas de imagen en [0, 1]).

    Se usa para canonicalizar la quiralidad: toda mano se lleva a una sola versión, de
    forma que el modelo ve la misma orientación sin importar con qué mano firme el usuario.
    """
    arr = np.array(landmarks, dtype=np.float32)
    arr[:, 0] = 1.0 - arr[:, 0]
    return arr.tolist()


def secuencia_con_velocidad(secuencia_plana: np.ndarray, features_frame: int) -> np.ndarray:
    """Enriquece una secuencia aplanada (T*F) con la velocidad (delta entre frames).

    Devuelve una secuencia aplanada T*(2F): a cada frame se le concatena su desplazamiento
    respecto al anterior. La velocidad es lo que más ayuda a distinguir movimientos
    parecidos y es barata de calcular. Se aplica igual en entrenamiento y en predicción.
    """
    frames = np.asarray(secuencia_plana, dtype=np.float32).reshape(-1, features_frame)
    velocidad = np.zeros_like(frames)
    velocidad[1:] = frames[1:] - frames[:-1]
    return np.concatenate([frames, velocidad], axis=1).flatten()
