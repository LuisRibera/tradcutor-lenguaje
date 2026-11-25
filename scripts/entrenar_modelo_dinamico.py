from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aprendizaje import (
    cargar_dataset_dinamico,
    construir_modelo,
    construir_reporte_evaluacion,
    dividir_entrenamiento_prueba,
    validar_balance_clases,
)
from configuracion import CARPETA_MODELOS, perfil_dinamico


def entrenar_y_guardar(perfil: dict) -> None:
    x, y, grupos = cargar_dataset_dinamico(perfil["dataset"], perfil["longitud"])
    validar_balance_clases(y, perfil["minimo"])

    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba, por_persona = dividir_entrenamiento_prueba(x, y, grupos)
    print(f"División {'por persona (held-out)' if por_persona else 'estratificada'}.")

    modelo = construir_modelo()
    modelo.fit(x_entrenamiento, y_entrenamiento)
    predicciones = modelo.predict(x_prueba)
    probabilidades = modelo.predict_proba(x_prueba)
    confianza_promedio = float(probabilidades.max(axis=1).mean())

    clasificador_calibrado = modelo.named_steps["clasificador"]
    etiquetas = list(clasificador_calibrado.classes_)
    evaluacion = construir_reporte_evaluacion(y_prueba, predicciones, etiquetas)

    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, perfil["modelo"])
    with perfil["etiquetas"].open("w", encoding="utf-8") as archivo:
        json.dump(etiquetas, archivo, ensure_ascii=False, indent=2)

    reporte = {
        "tipo_modelo": "dinamico",
        "perfil": perfil["nombre"],
        "muestras_totales": int(len(y)),
        "clases": etiquetas,
        "muestras_por_clase": {str(clase): int(cantidad) for clase, cantidad in y.value_counts().items()},
        "confianza_promedio_validacion": confianza_promedio,
        "macro_f1": float(evaluacion["metricas_por_clase"].get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(evaluacion["metricas_por_clase"].get("weighted avg", {}).get("f1-score", 0.0)),
        "clases_mas_debiles": evaluacion["clases_mas_debiles"],
        "top_confusiones": evaluacion["top_confusiones"],
        "longitud_secuencia_modelo": perfil["longitud"],
    }
    with perfil["reporte_entrenamiento"].open("w", encoding="utf-8") as archivo:
        json.dump(reporte, archivo, ensure_ascii=False, indent=2)

    print(f"Entrenamiento dinámico ({perfil['nombre']}) completado.")
    print(f"Modelo guardado en: {perfil['modelo']}")
    print(f"Etiquetas guardadas en: {perfil['etiquetas']}")
    print(f"Reporte guardado en: {perfil['reporte_entrenamiento']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena un modelo dinámico: frases (por defecto) o letras con movimiento.")
    parser.add_argument("--letras", action="store_true", help="Entrenar el modelo de letras con movimiento en vez de las frases.")
    args = parser.parse_args()
    entrenar_y_guardar(perfil_dinamico(args.letras))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
