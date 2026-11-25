from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from configuracion import MUESTRAS_MINIMAS_POR_CLASE, obtener_tipo_etiqueta
from normalizacion import secuencia_con_velocidad
from secuencias import cargar_secuencias_jsonl, redimensionar_secuencia


def cargar_dataset_estatico(ruta_dataset: Path):
    if not ruta_dataset.exists():
        raise FileNotFoundError(f"No existe el dataset: {ruta_dataset}")

    data = pd.read_csv(ruta_dataset)
    if "etiqueta" not in data.columns:
        raise ValueError("El dataset debe incluir una columna 'etiqueta'.")

    if len(data) < 10:
        raise ValueError("Se requieren al menos 10 muestras para entrenar un modelo mínimo.")

    x = data.drop(columns=["etiqueta"])
    y = data["etiqueta"]
    return x, y


def cargar_dataset_dinamico(ruta_dataset: Path, longitud_secuencia: int):
    registros = cargar_secuencias_jsonl(ruta_dataset)
    if not registros:
        raise FileNotFoundError(f"No existe dataset dinámico utilizable en: {ruta_dataset}")

    filas = []
    etiquetas = []
    grupos = []
    for registro in registros:
        etiqueta = registro.get("etiqueta")
        frames = registro.get("frames") or []
        if not etiqueta or not frames:
            continue
        features_frame = len(frames[0])
        secuencia = redimensionar_secuencia(frames, longitud_secuencia)
        # Enriquecer con velocidad (posición + delta): clave para discriminar movimientos.
        filas.append(secuencia_con_velocidad(secuencia, features_frame))
        etiquetas.append(etiqueta)
        grupos.append(registro.get("persona", "desconocida"))

    if len(filas) < 10:
        raise ValueError("Se requieren al menos 10 secuencias para entrenar un modelo dinámico mínimo.")

    x = pd.DataFrame(filas)
    y = pd.Series(etiquetas, name="etiqueta")
    grupos = pd.Series(grupos, name="persona")
    return x, y, grupos


def dividir_entrenamiento_prueba(x: pd.DataFrame, y: pd.Series, grupos: pd.Series | None = None,
                                 test_size: float = 0.2, semilla: int = 42):
    """Divide en entrenamiento/prueba. Si hay >=2 personas identificadas, usa un split por
    persona (GroupShuffleSplit) para estimar el desempeño ante gente no vista. Devuelve además
    un indicador de si la división fue por persona."""
    personas = set(grupos) - {"desconocida"} if grupos is not None else set()
    if grupos is not None and len(personas) >= 2:
        divisor = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=semilla)
        idx_entrenamiento, idx_prueba = next(divisor.split(x, y, groups=grupos))
        return (
            x.iloc[idx_entrenamiento], x.iloc[idx_prueba],
            y.iloc[idx_entrenamiento], y.iloc[idx_prueba],
            True,
        )
    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(
        x, y, test_size=test_size, random_state=semilla, stratify=y,
    )
    return x_entrenamiento, x_prueba, y_entrenamiento, y_prueba, False




def validar_balance_clases(y: pd.Series, minimo_por_clase: int = MUESTRAS_MINIMAS_POR_CLASE) -> None:
    conteos = y.value_counts()
    insuficientes = conteos[conteos < minimo_por_clase]
    if not insuficientes.empty:
        detalle = ", ".join(f"{etiqueta}: {cantidad}" for etiqueta, cantidad in insuficientes.items())
        raise ValueError(
            "El dataset tiene clases con muy pocas muestras para un modelo robusto. "
            f"Mínimo esperado por clase: {minimo_por_clase}. "
            f"Clases insuficientes: {detalle}"
        )


def construir_modelo() -> Pipeline:
    bosque = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    return Pipeline(
        steps=[
            ("escalador", StandardScaler()),
            ("clasificador", CalibratedClassifierCV(estimator=bosque, cv=3, method="sigmoid")),
        ]
    )


def construir_reporte_evaluacion(y_real, y_pred, etiquetas: list[str]) -> dict:
    reporte = classification_report(y_real, y_pred, labels=etiquetas, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_real, y_pred, labels=etiquetas)

    confusiones = []
    for fila_indice, etiqueta_real in enumerate(etiquetas):
        for col_indice, etiqueta_predicha in enumerate(etiquetas):
            if fila_indice == col_indice:
                continue
            cantidad = int(matriz[fila_indice][col_indice])
            if cantidad > 0:
                confusiones.append(
                    {
                        "real": etiqueta_real,
                        "predicha": etiqueta_predicha,
                        "cantidad": cantidad,
                        "tipo_real": obtener_tipo_etiqueta(etiqueta_real),
                    }
                )

    confusiones.sort(key=lambda item: (-item["cantidad"], item["real"], item["predicha"]))

    clases_debiles = []
    for etiqueta in etiquetas:
        metricas = reporte.get(etiqueta, {})
        clases_debiles.append(
            {
                "etiqueta": etiqueta,
                "tipo": obtener_tipo_etiqueta(etiqueta),
                "precision": float(metricas.get("precision", 0.0)),
                "recall": float(metricas.get("recall", 0.0)),
                "f1": float(metricas.get("f1-score", 0.0)),
                "soporte": int(metricas.get("support", 0)),
            }
        )
    clases_debiles.sort(key=lambda item: (item["f1"], item["recall"], item["precision"], item["etiqueta"]))

    return {
        "metricas_por_clase": reporte,
        "matriz_confusion": matriz.tolist(),
        "top_confusiones": confusiones[:20],
        "clases_mas_debiles": clases_debiles[:15],
    }
