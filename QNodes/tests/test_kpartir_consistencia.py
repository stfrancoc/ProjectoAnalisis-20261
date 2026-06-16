"""
Prueba de regresión: `System.kpartir([bloque, complemento])` debe coincidir
EXACTAMENTE con `System.bipartir(alcance, mecanismo)` (la canonicalización ahora
vive en `src/funcs/particion.py`, pero el comportamiento observable no cambia).

Se construye el subsistema completo de N3A leyendo el CSV de muestra por ruta
relativa a este archivo (independiente del directorio de trabajo) y evitando la
cadena de importación pesada (pyttsx3/pyinstrument) de las estrategias.

Ejecutar con `uv run pytest tests/test_kpartir_consistencia.py` o
`uv run python tests/test_kpartir_consistencia.py`.
"""

import os
import sys
from itertools import chain, combinations
from pathlib import Path

import numpy as np

# Permite ejecutar el archivo directamente (`python tests/...py`) añadiendo la raíz
# del proyecto (QNodes/) al path; bajo pytest es inocuo (ya está en sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants.base import ACTUAL, COLON_DELIM, EFFECT  # noqa: E402
from src.models.core.system import System  # noqa: E402

RUTA_N3A = Path(__file__).resolve().parent.parent / "src" / ".samples" / "N3A.csv"


def _subsistema_completo_n3a() -> System:
    """Carga N3A y devuelve su subsistema completo (sin condicionar ni substraer)."""
    tpm = np.genfromtxt(RUTA_N3A, delimiter=COLON_DELIM)
    estado = np.array([1, 0, 0], dtype=np.int8)
    completo = System(tpm, estado)
    candidato = completo.condicionar(np.array([], dtype=np.int8))
    return candidato.substraer(np.array([], dtype=np.int8), np.array([], dtype=np.int8))


def _subconjuntos(arr):
    return chain.from_iterable(combinations(arr, r) for r in range(len(arr) + 1))


def test_kpartir_coincide_con_bipartir_en_todos_los_casos_de_n3a():
    """Para todos los (alcance, mecanismo) posibles, kpartir == bipartir."""
    sub = _subsistema_completo_n3a()
    idxs = sub.indices_ncubos.tolist()
    dims = sub.dims_ncubos.tolist()

    comparados = 0
    for alcance in _subconjuntos(idxs):
        for mecanismo in _subconjuntos(dims):
            esperado = sub.bipartir(
                np.array(alcance, dtype=np.int8),
                np.array(mecanismo, dtype=np.int8),
            ).distribucion_marginal()

            bloque = [(EFFECT, int(i)) for i in alcance] + [
                (ACTUAL, int(j)) for j in mecanismo
            ]
            complemento = [
                (EFFECT, int(i)) for i in idxs if i not in alcance
            ] + [(ACTUAL, int(j)) for j in dims if j not in mecanismo]

            obtenido = sub.kpartir([bloque, complemento]).distribucion_marginal()

            assert np.allclose(esperado, obtenido), (
                f"Discrepancia en alcance={alcance}, mecanismo={mecanismo}: "
                f"{esperado} != {obtenido}"
            )
            comparados += 1

    # N3A completo: 2^3 alcances * 2^3 mecanismos = 64 casos.
    assert comparados == 64


if __name__ == "__main__":
    test_kpartir_coincide_con_bipartir_en_todos_los_casos_de_n3a()
    print("OK  test_kpartir_coincide_con_bipartir_en_todos_los_casos_de_n3a")
    print("\n1 prueba de consistencia kpartir/bipartir pasó.")
