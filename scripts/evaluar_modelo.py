from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aprendizaje import (
    cargar_dataset_estatico,
    construir_modelo,
    construir_reporte_evaluacion,
    dividir_entrenamiento_prueba,
    validar_balance_clases,
)
from configuracion import (
    ARCHIVO_DATASET,
    ARCHIVO_MATRIZ_CONFUSION,
    ARCHIVO_PRIORIDADES_REFUERZO,
    ARCHIVO_REPORTE_EVALUACION,
    CARPETA_MODELOS,
)


def guardar_matriz_confusion(ruta_csv: Path, etiquetas: list[str], matriz: list[list[int]]) -> None:
    with ruta_csv.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["real/predicha"] + etiquetas)
        for etiqueta, fila in zip(etiquetas, matriz):
            escritor.writerow([etiqueta] + fila)


def guardar_prioridades_refuerzo(ruta_json: Path, clases_mas_debiles: list[dict], top_confusiones: list[dict]) -> None:
    prioridades = []
    mapa_confusiones: dict[str, list[dict]] = {}
    for confusion in top_confusiones:
        mapa_confusiones.setdefault(confusion["real"], []).append(confusion)

    for clase in clases_mas_debiles:
        etiqueta = clase["etiqueta"]
        prioridades.append(
            {
                "etiqueta": etiqueta,
                "tipo": clase["tipo"],
                "f1": clase["f1"],
                "recall": clase["recall"],
                "precision": clase["precision"],
                "confusiones_frecuentes": mapa_confusiones.get(etiqueta, [])[:3],
            }
        )

    with ruta_json.open("w", encoding="utf-8") as archivo:
        json.dump(prioridades, archivo, ensure_ascii=False, indent=2)


def evaluar_y_guardar(ruta_dataset: Path) -> None:
    x, y = cargar_dataset_estatico(ruta_dataset)
    validar_balance_clases(y)

    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba, _ = dividir_entrenamiento_prueba(x, y)

    modelo = construir_modelo()
    modelo.fit(x_entrenamiento, y_entrenamiento)
    predicciones = modelo.predict(x_prueba)

    clasificador_calibrado = modelo.named_steps["clasificador"]
    etiquetas = list(clasificador_calibrado.classes_)
    evaluacion = construir_reporte_evaluacion(y_prueba, predicciones, etiquetas)
    evaluacion["tipo_modelo"] = "estatico"

    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    with ARCHIVO_REPORTE_EVALUACION.open("w", encoding="utf-8") as archivo:
        json.dump(evaluacion, archivo, ensure_ascii=False, indent=2)
    guardar_matriz_confusion(ARCHIVO_MATRIZ_CONFUSION, etiquetas, evaluacion["matriz_confusion"])
    guardar_prioridades_refuerzo(
        ARCHIVO_PRIORIDADES_REFUERZO,
        evaluacion["clases_mas_debiles"],
        evaluacion["top_confusiones"],
    )

    print("Evaluación estática completada.")
    print(f"Reporte detallado: {ARCHIVO_REPORTE_EVALUACION}")
    print(f"Matriz de confusión: {ARCHIVO_MATRIZ_CONFUSION}")
    print(f"Prioridades de refuerzo: {ARCHIVO_PRIORIDADES_REFUERZO}")


if __name__ == "__main__":
    evaluar_y_guardar(ARCHIVO_DATASET)
