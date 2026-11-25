"""Prueba de humo: confirma que la cadena datos -> entrenar -> cargar -> predecir corre en esta
máquina, sin cámara y sin tocar los archivos reales de data/ ni models/.

Genera datasets sintéticos en un directorio temporal, entrena los tres modelos (estático, letras
con movimiento y frases), los carga con ClasificadorSenias y ejecuta una predicción de cada tipo.
Al terminar borra el temporal. NO cubre la captura por cámara (para eso usa verificar_entorno.py).
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aprendizaje import (
    cargar_dataset_dinamico,
    cargar_dataset_estatico,
    construir_modelo,
    dividir_entrenamiento_prueba,
    validar_balance_clases,
)
from clasificador import ClasificadorSenias
from configuracion import (
    LONGITUD_SECUENCIA_MODELO,
    LONGITUD_SECUENCIA_PALABRA,
    MUESTRAS_MINIMAS_POR_LETRA_DINAMICA,
    MUESTRAS_MINIMAS_POR_PALABRA,
    NUM_FEATURES_DINAMICO,
)
from secuencias import guardar_secuencia_jsonl


def _generar_estatico(ruta: Path, etiquetas: list[str], por_clase: int, rng) -> None:
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["etiqueta"] + [f"f_{i}" for i in range(63)])
        for indice, etiqueta in enumerate(etiquetas):
            for _ in range(por_clase):
                escritor.writerow([etiqueta] + (indice + rng.normal(0, 0.05, 63)).tolist())


def _generar_dinamico(ruta: Path, etiquetas: list[str], ancho: int, rng) -> None:
    for indice, etiqueta in enumerate(etiquetas):
        for persona in ("persona1", "persona2"):
            for _ in range(12):
                frames = (indice + rng.normal(0, 0.05, (15, ancho))).tolist()
                guardar_secuencia_jsonl(ruta, etiqueta, frames, persona)


def _entrenar_estatico(ruta_dataset: Path, ruta_modelo: Path, ruta_etiquetas: Path) -> None:
    x, y = cargar_dataset_estatico(ruta_dataset)
    validar_balance_clases(y)
    x_tr, _, y_tr, _, _ = dividir_entrenamiento_prueba(x, y)
    _ajustar_y_guardar(x_tr, y_tr, ruta_modelo, ruta_etiquetas)


def _entrenar_dinamico(ruta_dataset: Path, ruta_modelo: Path, ruta_etiquetas: Path, minimo: int, longitud: int) -> None:
    x, y, grupos = cargar_dataset_dinamico(ruta_dataset, longitud)
    validar_balance_clases(y, minimo)
    x_tr, _, y_tr, _, _ = dividir_entrenamiento_prueba(x, y, grupos)
    _ajustar_y_guardar(x_tr, y_tr, ruta_modelo, ruta_etiquetas)


def _ajustar_y_guardar(x_tr, y_tr, ruta_modelo: Path, ruta_etiquetas: Path) -> None:
    modelo = construir_modelo()
    modelo.fit(x_tr, y_tr)
    joblib.dump(modelo, ruta_modelo)
    etiquetas = list(modelo.named_steps["clasificador"].classes_)
    with ruta_etiquetas.open("w", encoding="utf-8") as archivo:
        json.dump(etiquetas, archivo, ensure_ascii=False, indent=2)


def ejecutar() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="prueba_humo_"))
    rng = np.random.default_rng(0)
    print(f"Prueba de humo en carpeta temporal: {tmp}")
    try:
        # 1) Datos sintéticos separables.
        ds_estatico = tmp / "estatico.csv"
        ds_letras = tmp / "letras.jsonl"
        ds_palabras = tmp / "palabras.jsonl"
        _generar_estatico(ds_estatico, ["A", "B", "C"], por_clase=20, rng=rng)
        _generar_dinamico(ds_letras, ["J", "Z"], ancho=63, rng=rng)
        _generar_dinamico(ds_palabras, ["Hola", "cómo"], ancho=NUM_FEATURES_DINAMICO, rng=rng)
        print("[1/4] Datasets sintéticos generados.")

        # 2) Entrenar los tres modelos.
        m_est, e_est = tmp / "m_estatico.joblib", tmp / "e_estatico.json"
        m_let, e_let = tmp / "m_letras.joblib", tmp / "e_letras.json"
        m_pal, e_pal = tmp / "m_palabras.joblib", tmp / "e_palabras.json"
        _entrenar_estatico(ds_estatico, m_est, e_est)
        _entrenar_dinamico(ds_letras, m_let, e_let, MUESTRAS_MINIMAS_POR_LETRA_DINAMICA, LONGITUD_SECUENCIA_MODELO)
        _entrenar_dinamico(ds_palabras, m_pal, e_pal, MUESTRAS_MINIMAS_POR_PALABRA, LONGITUD_SECUENCIA_PALABRA)
        print("[2/4] Modelos entrenados (estático, letras, palabras).")

        # 3) Cargar en la app tal como lo hace main.py.
        clasificador = ClasificadorSenias(
            ruta_modelo=m_est, ruta_etiquetas=e_est,
            ruta_modelo_dinamico=m_pal, ruta_etiquetas_dinamicas=e_pal,
            ruta_modelo_letras=m_let, ruta_etiquetas_letras=e_let,
        )
        assert clasificador.usando_modelo_entrenado, "no cargó el modelo estático"
        assert clasificador.usando_modelo_dinamico, "no cargó el modelo de palabras"
        assert clasificador.usando_modelo_letras, "no cargó el modelo de letras"
        print("[3/4] ClasificadorSenias cargó los tres modelos.")

        # 4) Predecir un ejemplo de cada tipo (vectores cercanos a una clase -> confianza alta).
        landmarks = np.zeros((21, 3)).tolist()
        pred_est = clasificador.predecir_estatico(np.full(63, 1.0, dtype=np.float32), landmarks)
        pred_let = clasificador.predecir_letra_dinamica(np.full(63 * LONGITUD_SECUENCIA_MODELO, 1.0, dtype=np.float32))
        pred_pal = clasificador.predecir_palabra(np.full(NUM_FEATURES_DINAMICO * LONGITUD_SECUENCIA_PALABRA, 1.0, dtype=np.float32))
        alfabeto = clasificador.combinar_alfabeto(pred_est, pred_let)
        print(f"[4/4] Predicciones -> estático: {pred_est}")
        print(f"                     letra:    {pred_let}")
        print(f"                     palabra:  {pred_pal}")
        print(f"                     alfabeto (combinado): {alfabeto}")

        assert pred_est is not None and pred_pal is not None, "alguna predicción falló"
        print("\nPRUEBA DE HUMO: PASS. La cadena datos -> entrenar -> cargar -> predecir funciona.")
        print("Nota: no cubre la cámara. Para eso: python scripts/verificar_entorno.py")
        return 0
    except Exception as error:  # noqa: BLE001 - la prueba de humo reporta cualquier fallo
        print(f"\nPRUEBA DE HUMO: FAIL -> {type(error).__name__}: {error}")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(ejecutar())
