import cv2


class GestorCamara:
    def __init__(self, indice_camara: int, ancho: int, alto: int) -> None:
        self.indice_camara = indice_camara
        self.ancho = ancho
        self.alto = alto
        self.captura = None

    def abrir(self) -> None:
        self.captura = cv2.VideoCapture(self.indice_camara)
        if not self.captura or not self.captura.isOpened():
            raise RuntimeError("No se pudo abrir la cámara.")
        self.captura.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
        self.captura.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)

    def obtener_frame(self):
        if self.captura is None:
            raise RuntimeError("La cámara aún no fue inicializada.")
        exito, frame = self.captura.read()
        if not exito:
            raise RuntimeError("No se pudo leer un frame desde la cámara.")
        return frame

    def liberar(self) -> None:
        if self.captura is not None:
            self.captura.release()
            self.captura = None
