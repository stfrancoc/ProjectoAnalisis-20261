"""
Pruebas del módulo puro de codificación/normalización de particiones
(`src/funcs/particion.py`).

Se pueden ejecutar con `uv run pytest tests/test_particion.py` o directamente con
`uv run python tests/test_particion.py` (incluye un runner propio para no depender
de que pytest esté instalado).
"""

import os
import sys
from contextlib import contextmanager

# Permite ejecutar el archivo directamente (`python tests/...py`) añadiendo la raíz
# del proyecto (QNodes/) al path; bajo pytest es inocuo (ya está en sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.base import ACTUAL, EFFECT  # noqa: E402
from src.funcs.particion import (  # noqa: E402
    bloques_desde_etiquetas,
    clave_canonica,
    etiquetas_desde_bloques,
    normalizar_etiquetas,
    orden_canonico_nodos,
    validar_particion,
)


@contextmanager
def asume_value_error(descripcion: str):
    """Contexto que exige que el bloque interior lance `ValueError`."""
    try:
        yield
    except ValueError:
        return
    raise AssertionError(f"Se esperaba ValueError y no ocurrió: {descripcion}")


def como_conjunto(bloques):
    """Representa una partición como conjunto de bloques (ignora orden)."""
    return {frozenset((int(t), int(i)) for (t, i) in bloque) for bloque in bloques}


def test_orden_canonico_nodos_presentes_luego_futuros():
    """El orden es presentes (ACTUAL) ascendentes seguidos de futuros (EFFECT)."""
    orden = orden_canonico_nodos([2, 0, 1], [1, 0])
    assert orden == [
        (ACTUAL, 0),
        (ACTUAL, 1),
        (ACTUAL, 2),
        (EFFECT, 0),
        (EFFECT, 1),
    ]


def test_normalizar_etiquetas_colapsa_equivalentes():
    """Vectores que sólo difieren en el nombre de los bloques dan la misma clave."""
    assert normalizar_etiquetas([0, 0, 1, 2]) == (0, 0, 1, 2)
    assert normalizar_etiquetas([1, 1, 0, 2]) == (0, 0, 1, 2)
    assert normalizar_etiquetas([2, 2, 0, 1]) == (0, 0, 1, 2)
    # Las tres versiones colapsan entre sí.
    assert (
        normalizar_etiquetas([0, 0, 1, 2])
        == normalizar_etiquetas([1, 1, 0, 2])
        == normalizar_etiquetas([2, 2, 0, 1])
    )


def test_normalizar_etiquetas_es_idempotente():
    """Normalizar una clave ya canónica no la cambia."""
    vector = [3, 1, 3, 0, 1]
    una_vez = normalizar_etiquetas(vector)
    dos_veces = normalizar_etiquetas(una_vez)
    assert una_vez == dos_veces == (0, 1, 0, 2, 1)


def test_clave_canonica_no_depende_del_nombre_del_bloque():
    """Dos particiones iguales con bloques en distinto orden comparten clave."""
    orden = orden_canonico_nodos([0, 1], [0, 1])
    p1 = [[(ACTUAL, 0), (EFFECT, 0)], [(ACTUAL, 1), (EFFECT, 1)]]
    p2 = [[(ACTUAL, 1), (EFFECT, 1)], [(ACTUAL, 0), (EFFECT, 0)]]
    assert clave_canonica(p1, orden) == clave_canonica(p2, orden)


def test_round_trip_bloques_etiquetas_bloques():
    """bloques -> etiquetas -> bloques preserva la partición (como conjunto)."""
    orden = orden_canonico_nodos([0, 1], [0, 1])
    bloques = [[(ACTUAL, 0), (EFFECT, 1)], [(ACTUAL, 1)], [(EFFECT, 0)]]
    vector = etiquetas_desde_bloques(bloques, orden)
    assert vector == (0, 1, 2, 0)
    reconstruidos = bloques_desde_etiquetas(vector, orden)
    assert como_conjunto(reconstruidos) == como_conjunto(bloques)


def test_validar_particion_acepta_particion_valida():
    """Una partición real con k correcto no lanza error."""
    orden = orden_canonico_nodos([0, 1], [0])
    bloques = [[(ACTUAL, 0), (ACTUAL, 1)], [(EFFECT, 0)]]
    validar_particion(bloques, orden, k=2)  # no debe lanzar


def test_validar_particion_rechaza_nodo_faltante():
    orden = orden_canonico_nodos([0, 1], [0])  # 3 nodos
    bloques = [[(ACTUAL, 0)], [(ACTUAL, 1)]]  # falta (EFFECT, 0)
    with asume_value_error("nodo faltante"):
        validar_particion(bloques, orden)


def test_validar_particion_rechaza_nodo_duplicado():
    orden = orden_canonico_nodos([0, 1], [0])
    bloques = [[(ACTUAL, 0)], [(ACTUAL, 0)], [(ACTUAL, 1)], [(EFFECT, 0)]]
    with asume_value_error("nodo duplicado"):
        validar_particion(bloques, orden)


def test_validar_particion_rechaza_k_fuera_de_rango():
    orden = orden_canonico_nodos([0, 1], [0])  # m = 3
    bloques = [[(ACTUAL, 0), (ACTUAL, 1), (EFFECT, 0)]]  # 1 bloque
    with asume_value_error("k=1 (trivial, fuera de rango)"):
        validar_particion(bloques, orden, k=1)
    bloques_k = [[(ACTUAL, 0)], [(ACTUAL, 1)], [(EFFECT, 0)]]
    with asume_value_error("k=4 > m=3"):
        validar_particion(bloques_k, orden, k=4)


def test_validar_particion_rechaza_k_distinto_de_bloques_no_vacios():
    orden = orden_canonico_nodos([0, 1], [0])
    bloques = [[(ACTUAL, 0)], [(ACTUAL, 1)], [(EFFECT, 0)]]  # 3 bloques no vacíos
    with asume_value_error("k=2 pero hay 3 bloques no vacíos"):
        validar_particion(bloques, orden, k=2)


def _ejecutar_todas():
    """Runner manual para correr sin pytest."""
    pruebas = [
        valor
        for nombre, valor in sorted(globals().items())
        if nombre.startswith("test_") and callable(valor)
    ]
    for prueba in pruebas:
        prueba()
        print(f"OK  {prueba.__name__}")
    print(f"\n{len(pruebas)} pruebas de particion pasaron.")


if __name__ == "__main__":
    _ejecutar_todas()
