from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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


def evaluar_y_guardar(perfil: dict) -> None:
    x, y, grupos = cargar_dataset_dinamico(perfil["dataset"], perfil["longitud"])
    validar_balance_clases(y, perfil["minimo"])

    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba, por_persona = dividir_entrenamiento_prueba(x, y, grupos)
    print(f"División {'por persona (held-out)' if por_persona else 'estratificada'}.")

    modelo = construir_modelo()
    modelo.fit(x_entrenamiento, y_entrenamiento)
    predicciones = modelo.predict(x_prueba)

    clasificador_calibrado = modelo.named_steps["clasificador"]
    etiquetas = list(clasificador_calibrado.classes_)
    evaluacion = construir_reporte_evaluacion(y_prueba, predicciones, etiquetas)
    evaluacion["tipo_modelo"] = "dinamico"
    evaluacion["perfil"] = perfil["nombre"]
    evaluacion["longitud_secuencia_modelo"] = perfil["longitud"]

    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    with perfil["reporte_evaluacion"].open("w", encoding="utf-8") as archivo:
        json.dump(evaluacion, archivo, ensure_ascii=False, indent=2)
    guardar_matriz_confusion(perfil["matriz"], etiquetas, evaluacion["matriz_confusion"])
    guardar_prioridades_refuerzo(
        perfil["prioridades"],
        evaluacion["clases_mas_debiles"],
        evaluacion["top_confusiones"],
    )

    print(f"Evaluación dinámica ({perfil['nombre']}) completada.")
    print(f"Reporte detallado: {perfil['reporte_evaluacion']}")
    print(f"Matriz de confusión: {perfil['matriz']}")
    print(f"Prioridades de refuerzo: {perfil['prioridades']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa un modelo dinámico: frases (por defecto) o letras con movimiento.")
    parser.add_argument("--letras", action="store_true", help="Evaluar el modelo de letras con movimiento en vez de las frases.")
    args = parser.parse_args()
    evaluar_y_guardar(perfil_dinamico(args.letras))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
