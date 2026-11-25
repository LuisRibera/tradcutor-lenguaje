from __future__ import annotations


class EditorTexto:
    """Oración editable con un cursor de texto.

    Las señas reconocidas se insertan en la posición del cursor (una letra, o una palabra
    seguida de espacio). Con el teclado se mueve el cursor (flechas), se inserta espacio y se
    borra, para corregir la oración antes de leerla completa.
    """

    def __init__(self) -> None:
        self._texto = ""
        self._cursor = 0  # posición del cursor: 0..len(texto)

    def texto(self) -> str:
        return self._texto

    def posicion(self) -> int:
        return self._cursor

    def insertar(self, fragmento: str) -> None:
        if not fragmento:
            return
        self._texto = self._texto[: self._cursor] + fragmento + self._texto[self._cursor :]
        self._cursor += len(fragmento)

    def insertar_letra(self, letra: str) -> None:
        self.insertar(letra)

    def insertar_palabra(self, palabra: str) -> None:
        """Inserta una palabra con un espacio final para separar de la siguiente seña."""
        self.insertar(f"{palabra} ")

    def espacio(self) -> None:
        self.insertar(" ")

    def retroceso(self) -> None:
        if self._cursor > 0:
            self._texto = self._texto[: self._cursor - 1] + self._texto[self._cursor :]
            self._cursor -= 1

    def mover_izquierda(self) -> None:
        if self._cursor > 0:
            self._cursor -= 1

    def mover_derecha(self) -> None:
        if self._cursor < len(self._texto):
            self._cursor += 1

    def limpiar(self) -> None:
        self._texto = ""
        self._cursor = 0
