from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parent
CARPETA_DATOS = RAIZ_PROYECTO / "data"
CARPETA_MODELOS = RAIZ_PROYECTO / "models"
CARPETA_SCRIPTS = RAIZ_PROYECTO / "scripts"

ARCHIVO_MODELO = CARPETA_MODELOS / "modelo_senias.joblib"
ARCHIVO_ETIQUETAS = CARPETA_MODELOS / "etiquetas.json"
ARCHIVO_REPORTE_ENTRENAMIENTO = CARPETA_MODELOS / "reporte_entrenamiento.json"
ARCHIVO_REPORTE_EVALUACION = CARPETA_MODELOS / "reporte_evaluacion.json"
ARCHIVO_MATRIZ_CONFUSION = CARPETA_MODELOS / "matriz_confusion.csv"
ARCHIVO_PRIORIDADES_REFUERZO = CARPETA_MODELOS / "prioridades_refuerzo.json"

# Modelo dinámico de frases/señas (Holistic: pose + ambas manos). Cada frase es una clase
# capturada en una sola toma; al reconocerla se inserta su texto completo (ver PALABRAS).
ARCHIVO_MODELO_DINAMICO = CARPETA_MODELOS / "modelo_palabras.joblib"
ARCHIVO_ETIQUETAS_DINAMICAS = CARPETA_MODELOS / "etiquetas_palabras.json"
ARCHIVO_REPORTE_DINAMICO = CARPETA_MODELOS / "reporte_entrenamiento_palabras.json"
ARCHIVO_REPORTE_EVALUACION_DINAMICA = CARPETA_MODELOS / "reporte_evaluacion_palabras.json"
ARCHIVO_MATRIZ_CONFUSION_DINAMICA = CARPETA_MODELOS / "matriz_confusion_palabras.csv"
ARCHIVO_PRIORIDADES_REFUERZO_DINAMICAS = CARPETA_MODELOS / "prioridades_refuerzo_palabras.json"

# Letras con movimiento (J, K, Q, X, Z, Ñ): modelo dinámico de una mano.
ARCHIVO_MODELO_LETRAS_DINAMICAS = CARPETA_MODELOS / "modelo_letras_dinamicas.joblib"
ARCHIVO_ETIQUETAS_LETRAS_DINAMICAS = CARPETA_MODELOS / "etiquetas_letras_dinamicas.json"
ARCHIVO_REPORTE_LETRAS_DINAMICAS = CARPETA_MODELOS / "reporte_entrenamiento_letras_dinamicas.json"
ARCHIVO_REPORTE_EVALUACION_LETRAS_DINAMICAS = CARPETA_MODELOS / "reporte_evaluacion_letras_dinamicas.json"
ARCHIVO_MATRIZ_CONFUSION_LETRAS_DINAMICAS = CARPETA_MODELOS / "matriz_confusion_letras_dinamicas.csv"
ARCHIVO_PRIORIDADES_REFUERZO_LETRAS_DINAMICAS = CARPETA_MODELOS / "prioridades_refuerzo_letras_dinamicas.json"

ARCHIVO_DATASET = CARPETA_DATOS / "dataset_senias.csv"
ARCHIVO_DATASET_DINAMICO = CARPETA_DATOS / "dataset_palabras.jsonl"
ARCHIVO_DATASET_LETRAS_DINAMICAS = CARPETA_DATOS / "dataset_letras_dinamicas.jsonl"

INDICE_CAMARA = 0
ANCHO_VENTANA = 1280
ALTO_VENTANA = 720
ANCHO_PANEL = 360

# Verde característico de la detección (usado en el recuadro y acentos de la interfaz).
VERDE_DETECCION = (0, 255, 0)

# Área de reconocimiento (recuadro de la mano): centrada y amplia. Solo delimita dónde
# debe estar la mano; MediaPipe detecta desde el frame completo, así que agrandarla no
# perjudica la calidad.
ROI_ANCHO = 640
ROI_ALTO = 640
MARGEN_ROI_X = 20
MARGEN_ROI_Y = 20

UMBRAL_CONFIANZA = 0.60
UMBRAL_CONFIANZA_DINAMICA = 0.55  # umbral del modelo de frases (permisivo para uso en vivo)
UMBRAL_CONFIANZA_LETRA_DINAMICA = 0.70
FRAMES_CONFIRMACION = 8
COOLDOWN_AUDIO_SEGUNDOS = 2.5
MAX_HISTORIAL = 6
MAX_FRAMES_BUFFER = 40
LONGITUD_SECUENCIA_MODELO = 20      # letras con movimiento (una mano)
LONGITUD_SECUENCIA_PALABRA = 30     # palabras (Holistic): trayectorias más largas
MUESTRAS_MINIMAS_POR_CLASE = 15
MUESTRAS_MINIMAS_POR_PALABRA = 20
MUESTRAS_MINIMAS_POR_LETRA_DINAMICA = 20
MUESTRAS_OBJETIVO_LETRA = 120
MUESTRAS_OBJETIVO_LETRA_DINAMICA = 50
MUESTRAS_OBJETIVO_PALABRA = 60

# --- Captura de gestos (letras con movimiento y palabras) ---
SEGUNDOS_CUENTA_REGRESIVA = 3       # 3-2-1 antes de empezar a grabar
MAX_SEGUNDOS_GRABACION_GESTO = 6.0  # tope de seguridad; se graba hasta pulsar G otra vez

# --- Parada sin teclado por "dos palmas alzadas" (solo captura de palabras, Holistic) ---
# Alzar ambas palmas abiertas por encima de los hombros equivale a pulsar G para terminar,
# así no hay que ir al teclado ni se graba ese trayecto. Ver seguimiento_holistic.palmas_alzadas_abiertas.
MIN_DEDOS_EXTENDIDOS_STOP = 4       # dedos extendidos por mano para considerarla "abierta"
MARGEN_ALTURA_STOP = 0.05          # la muñeca debe superar al hombro por esta fracción del frame
FRAMES_CONFIRMACION_STOP = 4       # frames seguidos con la pose para confirmar (evita parpadeos)
MARGEN_FRAMES_STOP = 6             # frames a descartar antes del onset (transición de alzar los brazos)

# --- Arbitraje por movimiento (gate) en la app en vivo ---
# Desplazamiento normalizado (fracción del frame) de muñeca por frame para considerarse
# "en movimiento". Mano quieta -> letra; en movimiento -> gesto (letra con mov. / palabra).
UMBRAL_MOVIMIENTO = 0.020
MIN_FRAMES_GESTO = 8                 # frames mínimos de movimiento para tratarlo como gesto
# Frames quietos seguidos para dar por TERMINADO un gesto. >1 evita que una micro-pausa a
# mitad de una frase la parta en pedazos (era la causa de que las frases no se reconocieran).
FRAMES_FIN_GESTO = 6

# Mano canónica: toda detección se lleva a esta quiralidad (espejando si hace falta) para
# que el modelo vea una sola versión, sin importar con qué mano deletree el usuario.
MANO_CANONICA = "Right"

# --- Representación dinámica con MediaPipe Holistic ---
# Subconjunto de pose de torso superior: nariz, hombros, codos, muñecas y caderas.
POSE_INDICES_TORSO = [0, 11, 12, 13, 14, 15, 16, 23, 24]
# Muñecas dentro del subconjunto de pose (para medir el movimiento de brazos en palabras).
POSE_MUNIECAS = [15, 16]
# Hombros (pose completa de MediaPipe): referencia de altura para la parada por palmas alzadas.
POSE_HOMBROS = [11, 12]
# Puntos por frame dinámico: pose de torso (9) + mano izquierda (21) + mano derecha (21) = 51.
# Cada punto aporta 3 coordenadas (x, y, z) -> 51 * 3 = 153 features por frame.
NUM_FEATURES_DINAMICO = (len(POSE_INDICES_TORSO) + 21 + 21) * 3
# Features de una mano (21 puntos * 3): base de las secuencias de letras con movimiento.
NUM_FEATURES_MANO = 21 * 3

TEXTO_ESTADO_SIN_MANO = "Coloca la mano dentro del recuadro"
TEXTO_ESTADO_SIN_MODELO = "Sin modelo estático entrenado. Usando reglas base."

# Vocabulario de frases/señas (Holistic: pose + ambas manos), cada una capturada en UNA sola
# toma (una seña -> se inserta su texto completo). "Hola" + "cómo estás" forman la frase 1;
# el resto son frases completas. El nombre de "mi nombre es" se deletrea con el abecedario.
PALABRAS = [
    "Hola",
    "cómo estás",
    "puedo ayudarte",
    "dónde está el baño",
    "mi nombre es",
    "quieres comer conmigo",
]

# Alfabeto (dactilología, una mano). En LSM estas letras se hacen con movimiento,
# así que se reconocen con un modelo dinámico de secuencia, no con una pose fija.
LETRAS_DINAMICAS = ["J", "K", "Q", "X", "Z", "Ñ"]
# El resto del alfabeto son poses fijas (modelo estático).
LETRAS_ESTATICAS = [chr(codigo) for codigo in range(ord("A"), ord("Z") + 1) if chr(codigo) not in LETRAS_DINAMICAS]

ETIQUETAS_ESTATICAS = LETRAS_ESTATICAS
ETIQUETAS_DINAMICAS = PALABRAS
ETIQUETAS_BASE = LETRAS_ESTATICAS + LETRAS_DINAMICAS + PALABRAS


def obtener_tipo_etiqueta(etiqueta: str) -> str:
    if etiqueta in LETRAS_ESTATICAS:
        return "letra"
    if etiqueta in LETRAS_DINAMICAS:
        return "letra_dinamica"
    if etiqueta in PALABRAS:
        return "palabra"
    return "desconocida"


def es_etiqueta_dinamica(etiqueta: str) -> bool:
    return etiqueta in ETIQUETAS_DINAMICAS


def obtener_muestras_objetivo(etiqueta: str) -> int:
    tipo = obtener_tipo_etiqueta(etiqueta)
    if tipo == "letra":
        return MUESTRAS_OBJETIVO_LETRA
    if tipo == "letra_dinamica":
        return MUESTRAS_OBJETIVO_LETRA_DINAMICA
    if tipo == "palabra":
        return MUESTRAS_OBJETIVO_PALABRA
    return MUESTRAS_MINIMAS_POR_CLASE


def perfil_dinamico(letras: bool) -> dict:
    """Rutas y parámetros del pipeline dinámico: palabras (por defecto) o letras con
    movimiento. Ambos comparten la misma lógica; cambian archivos y longitud de secuencia."""
    if letras:
        return {
            "nombre": "letras con movimiento",
            "dataset": ARCHIVO_DATASET_LETRAS_DINAMICAS,
            "modelo": ARCHIVO_MODELO_LETRAS_DINAMICAS,
            "etiquetas": ARCHIVO_ETIQUETAS_LETRAS_DINAMICAS,
            "reporte_entrenamiento": ARCHIVO_REPORTE_LETRAS_DINAMICAS,
            "reporte_evaluacion": ARCHIVO_REPORTE_EVALUACION_LETRAS_DINAMICAS,
            "matriz": ARCHIVO_MATRIZ_CONFUSION_LETRAS_DINAMICAS,
            "prioridades": ARCHIVO_PRIORIDADES_REFUERZO_LETRAS_DINAMICAS,
            "minimo": MUESTRAS_MINIMAS_POR_LETRA_DINAMICA,
            "longitud": LONGITUD_SECUENCIA_MODELO,
        }
    return {
        "nombre": "palabras",
        "dataset": ARCHIVO_DATASET_DINAMICO,
        "modelo": ARCHIVO_MODELO_DINAMICO,
        "etiquetas": ARCHIVO_ETIQUETAS_DINAMICAS,
        "reporte_entrenamiento": ARCHIVO_REPORTE_DINAMICO,
        "reporte_evaluacion": ARCHIVO_REPORTE_EVALUACION_DINAMICA,
        "matriz": ARCHIVO_MATRIZ_CONFUSION_DINAMICA,
        "prioridades": ARCHIVO_PRIORIDADES_REFUERZO_DINAMICAS,
        "minimo": MUESTRAS_MINIMAS_POR_PALABRA,
        "longitud": LONGITUD_SECUENCIA_PALABRA,
    }
