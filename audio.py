from __future__ import annotations

import threading

import pyttsx3


class ServicioAudio:
    def __init__(self, habilitado: bool = True) -> None:
        self.habilitado = habilitado
        self.motor = None
        self._inicializar_motor()

    def _inicializar_motor(self) -> None:
        try:
            self.motor = pyttsx3.init()
            self.motor.setProperty("rate", 165)
            self._seleccionar_voz_espanol()
        except Exception:
            self.motor = None
            self.habilitado = False

    def _seleccionar_voz_espanol(self) -> None:
        """Elige una voz en español si el sistema la tiene; si no, avisa y usa la de por defecto."""
        try:
            voces = self.motor.getProperty("voices")
        except Exception:
            return
        for voz in voces or []:
            atributos = f"{getattr(voz, 'id', '')} {getattr(voz, 'name', '')}".lower()
            idiomas = " ".join(str(i) for i in getattr(voz, "languages", []) or []).lower()
            if "spanish" in atributos or "español" in atributos or "es-" in atributos or "es_" in idiomas:
                self.motor.setProperty("voice", voz.id)
                return
        print("[audio] No se encontró una voz en español; usando la de por defecto. "
              "Instala una voz española en Windows (Configuración > Hora e idioma > Voz).")

    def alternar(self) -> bool:
        self.habilitado = not self.habilitado
        return self.habilitado

    def reproducir_texto(self, texto: str) -> None:
        if not self.habilitado or not self.motor or not texto:
            return
        hilo = threading.Thread(target=self._hablar, args=(texto,), daemon=True)
        hilo.start()

    def _hablar(self, texto: str) -> None:
        try:
            self.motor.say(texto)
            self.motor.runAndWait()
        except RuntimeError:
            pass
