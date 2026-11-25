from __future__ import annotations

from collections import deque


class ConfirmadorLetra:
    """Confirma una letra estática por votación, pero SOLO cuando la mano está quieta.

    Reemplaza al cooldown fijo por un gate de movimiento: mientras la mano se mueve no se
    emite ninguna letra (ese movimiento es un gesto: letra con movimiento o palabra). Cuando
    la mano está quieta y hay acuerdo suficiente, se emite la letra **una sola vez por
    episodio de quietud**; para volver a emitir (incluida la misma letra repetida) hay que
    mover la mano y posarla de nuevo. Expone el progreso para la barra de validación.
    """

    def __init__(self, umbral_confianza: float, frames_confirmacion: int) -> None:
        self.umbral_confianza = umbral_confianza
        self.frames_confirmacion = frames_confirmacion
        self.buffer = deque(maxlen=frames_confirmacion)
        self.etiqueta_en_validacion = ""
        self.progreso_validacion = 0.0
        self.emitido_en_episodio = False

    def reiniciar(self) -> None:
        self.buffer.clear()
        self.etiqueta_en_validacion = ""
        self.progreso_validacion = 0.0
        self.emitido_en_episodio = False

    def actualizar(self, etiqueta: str, confianza: float, quieto: bool):
        """Devuelve la etiqueta a emitir (una vez por episodio de quietud) o None."""
        if not quieto:
            self.reiniciar()
            return None

        self.buffer.append((etiqueta, confianza))
        etiquetas = [dato[0] for dato in self.buffer]
        etiqueta_dominante = max(set(etiquetas), key=etiquetas.count)
        repeticiones = etiquetas.count(etiqueta_dominante)
        self.etiqueta_en_validacion = etiqueta_dominante
        self.progreso_validacion = min(1.0, repeticiones / self.frames_confirmacion)

        if self.emitido_en_episodio:
            return None
        if confianza < self.umbral_confianza:
            return None
        if len(self.buffer) < self.frames_confirmacion:
            return None
        if repeticiones < self.frames_confirmacion - 1:
            return None

        self.emitido_en_episodio = True
        self.progreso_validacion = 1.0
        return etiqueta_dominante
