from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from configuracion import (
    ARCHIVO_ETIQUETAS,
    ARCHIVO_ETIQUETAS_DINAMICAS,
    ARCHIVO_ETIQUETAS_LETRAS_DINAMICAS,
    ARCHIVO_MODELO,
    ARCHIVO_MODELO_DINAMICO,
    ARCHIVO_MODELO_LETRAS_DINAMICAS,
    NUM_FEATURES_DINAMICO,
    NUM_FEATURES_MANO,
    UMBRAL_CONFIANZA_DINAMICA,
    UMBRAL_CONFIANZA_LETRA_DINAMICA,
)
from normalizacion import secuencia_con_velocidad


def _cargar_modelo(ruta_modelo: Path, ruta_etiquetas: Path):
    """Devuelve (modelo, etiquetas) si ambos archivos existen, o (None, []) si no."""
    if ruta_modelo.exists() and ruta_etiquetas.exists():
        modelo = joblib.load(ruta_modelo)
        with ruta_etiquetas.open("r", encoding="utf-8") as archivo:
            return modelo, json.load(archivo)
    return None, []


class ClasificadorSenias:
    def __init__(
        self,
        ruta_modelo: Path = ARCHIVO_MODELO,
        ruta_etiquetas: Path = ARCHIVO_ETIQUETAS,
        ruta_modelo_dinamico: Path = ARCHIVO_MODELO_DINAMICO,
        ruta_etiquetas_dinamicas: Path = ARCHIVO_ETIQUETAS_DINAMICAS,
        ruta_modelo_letras: Path = ARCHIVO_MODELO_LETRAS_DINAMICAS,
        ruta_etiquetas_letras: Path = ARCHIVO_ETIQUETAS_LETRAS_DINAMICAS,
    ) -> None:
        self.ruta_modelo = ruta_modelo
        self.ruta_etiquetas = ruta_etiquetas
        self.ruta_modelo_dinamico = ruta_modelo_dinamico
        self.ruta_etiquetas_dinamicas = ruta_etiquetas_dinamicas
        self.ruta_modelo_letras = ruta_modelo_letras
        self.ruta_etiquetas_letras = ruta_etiquetas_letras
        self._cargar_si_existe()

    def _cargar_si_existe(self) -> None:
        self.modelo, self.etiquetas = _cargar_modelo(self.ruta_modelo, self.ruta_etiquetas)
        self.modelo_dinamico, self.etiquetas_dinamicas = _cargar_modelo(
            self.ruta_modelo_dinamico, self.ruta_etiquetas_dinamicas
        )
        self.modelo_letras, self.etiquetas_letras = _cargar_modelo(
            self.ruta_modelo_letras, self.ruta_etiquetas_letras
        )
        self.usando_modelo_entrenado = self.modelo is not None
        self.usando_modelo_dinamico = self.modelo_dinamico is not None
        self.usando_modelo_letras = self.modelo_letras is not None

    def recargar(self) -> None:
        self._cargar_si_existe()

    def _predecir_secuencia(self, modelo, etiquetas, secuencia, umbral, fuente, features_frame):
        if modelo is None or secuencia is None:
            return None
        # Mismo enriquecimiento (posición + velocidad) que en el entrenamiento.
        entrada = secuencia_con_velocidad(secuencia, features_frame)
        probabilidades = modelo.predict_proba([entrada])[0]
        indice = int(np.argmax(probabilidades))
        confianza = float(probabilidades[indice])
        if confianza < umbral:
            return None
        return etiquetas[indice], confianza, fuente

    def predecir_estatico(self, vector_caracteristicas: np.ndarray, landmarks: list[list[float]]):
        if self.modelo is not None:
            probabilidades = self.modelo.predict_proba([vector_caracteristicas])[0]
            indice = int(np.argmax(probabilidades))
            return self.etiquetas[indice], float(probabilidades[indice]), "modelo_estatico"

        return self._predecir_por_reglas(landmarks)

    def predecir_palabra(self, secuencia_redimensionada: np.ndarray | None):
        return self._predecir_secuencia(
            self.modelo_dinamico, self.etiquetas_dinamicas,
            secuencia_redimensionada, UMBRAL_CONFIANZA_DINAMICA, "modelo_palabra",
            NUM_FEATURES_DINAMICO,
        )

    def predecir_letra_dinamica(self, secuencia_redimensionada: np.ndarray | None):
        return self._predecir_secuencia(
            self.modelo_letras, self.etiquetas_letras,
            secuencia_redimensionada, UMBRAL_CONFIANZA_LETRA_DINAMICA, "modelo_letra_dinamica",
            NUM_FEATURES_MANO,
        )

    def combinar_alfabeto(self, prediccion_estatica, prediccion_letra_dinamica):
        """Alfabeto: gana la predicción de mayor confianza. Los dos modelos tienen etiquetas
        disjuntas (letras de pose fija vs. letras con movimiento)."""
        if prediccion_letra_dinamica is None:
            return prediccion_estatica
        if prediccion_estatica is None:
            return prediccion_letra_dinamica
        return max(prediccion_estatica, prediccion_letra_dinamica, key=lambda pred: pred[1])

    def _predecir_por_reglas(self, landmarks: list[list[float]]):
        if not landmarks or len(landmarks) < 21:
            return "Sin mano", 0.0, "reglas"

        dedos_extendidos = self._contar_dedos_extendidos(landmarks)
        pulgar_abierto = landmarks[4][0] < landmarks[3][0]
        munieca_y = landmarks[0][1]
        mano_arriba = landmarks[12][1] < munieca_y

        if dedos_extendidos >= 4 and mano_arriba:
            return "hola", 0.75, "reglas"
        if dedos_extendidos == 0:
            return "A", 0.65, "reglas"
        if dedos_extendidos == 1 and pulgar_abierto:
            return "L", 0.60, "reglas"
        if dedos_extendidos == 2:
            return "V", 0.60, "reglas"
        if dedos_extendidos == 5:
            return "gracias", 0.55, "reglas"

        return "Sin coincidencia", 0.25, "reglas"

    def _contar_dedos_extendidos(self, landmarks: list[list[float]]) -> int:
        puntas = [8, 12, 16, 20]
        articulaciones = [6, 10, 14, 18]
        conteo = 0
        for punta, articulacion in zip(puntas, articulaciones):
            if landmarks[punta][1] < landmarks[articulacion][1]:
                conteo += 1
        return conteo
