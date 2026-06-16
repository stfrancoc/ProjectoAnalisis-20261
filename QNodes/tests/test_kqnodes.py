"""
Pruebas de la estrategia voraz aglomerativa `KQNodes` (k-particiones).

Cubre, sobre la red de muestra N3A:
- Ejecución con k=2 y k=3 (imprime la partición y el δ obtenidos).
- Que la partición devuelta es VÁLIDA (cobertura/unicidad), tiene exactamente k
  bloques y cubre todos los nodos.
- Determinismo: dos corridas independientes dan la misma partición y el mismo δ.
- Sanidad k=2 contra el óptimo exacto de `BruteForce`: ambas recorren el MISMO
  espacio bipartito (todas las 2-particiones en dos bloques no vacíos), por lo que
  el δ voraz debe ser >= el δ óptimo exacto (con tolerancia de punto flotante); se
  reporta además si coinciden exactamente.

Ejecutar con `uv run pytest tests/test_kqnodes.py -s` o directamente con
`uv run python tests/test_kqnodes.py`.
"""

import os
import sys
from pathlib import Path

import numpy as np

# Permite ejecutar el archivo directamente (`python tests/...py`) añadiendo la raíz
# del proyecto (QNodes/) al path; bajo pytest es inocuo (ya está en sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Desactiva el profiler ANTES de importar las estrategias, para que construir
# `BruteForce` no genere directorios ni ficheros HTML de perfilado en las pruebas.
from src.middlewares.profile import gestor_perfilado  # noqa: E402

gestor_perfilado.enabled = False

from src.constants.base import COLON_DELIM  # noqa: E402
from src.funcs.particion import (  # noqa: E402
    orden_canonico_nodos,
    validar_particion,
)
from src.strategies.force import BruteForce  # noqa: E402
from src.strategies.k_qnodes import KQNodes  # noqa: E402

RUTA_N3A = Path(__file__).resolve().parent.parent / "src" / ".samples" / "N3A.csv"

# Subsistema completo de N3A: 3 nodos presentes + 3 futuros = 6 nodos particionables.
ESTADO_INICIAL = "100"
CONDICION = "111"
ALCANCE = "111"
MECANISMO = "111"

TOLERANCIA = 1e-9


def _imprimir(texto: str) -> None:
    """
    Imprime de forma segura aunque la consola no soporte Unicode (cp1252 en
    Windows): si falla la codificación, reemplaza los caracteres no representables
    (p. ej. `δ`, `ó` o los caracteres de caja de la partición) en lugar de romper.
    """
    try:
        print(texto)
    except UnicodeEncodeError:
        codificacion = sys.stdout.encoding or "ascii"
        print(texto.encode(codificacion, errors="replace").decode(codificacion))


def _cargar_tpm_n3a() -> np.ndarray:
    """Carga la TPM de N3A por ruta relativa a este archivo (independiente del CWD)."""
    return np.genfromtxt(RUTA_N3A, delimiter=COLON_DELIM)


def _ejecutar_kqnodes(k: int) -> tuple[KQNodes, object]:
    """Construye y ejecuta KQNodes sobre N3A para un `k` dado."""
    estrategia = KQNodes(_cargar_tpm_n3a())
    solucion = estrategia.aplicar_estrategia(
        ESTADO_INICIAL, CONDICION, ALCANCE, MECANISMO, k
    )
    return estrategia, solucion


def test_kqnodes_k2_y_k3_imprime_particion_y_delta():
    """KQNodes corre sobre N3A con k=2 y k=3 e imprime la partición y el δ."""
    for k in (2, 3):
        estrategia, solucion = _ejecutar_kqnodes(k)
        _imprimir(f"\n[N3A] KQNodes k={k}  ->  delta = {solucion.perdida:.6f}")
        _imprimir(solucion.particion)
        assert solucion.perdida >= -TOLERANCIA  # un delta (EMD) nunca es negativo


def test_kqnodes_devuelve_particion_valida_con_k_bloques():
    """La partición final es válida, tiene exactamente k bloques y cubre todo."""
    for k in (2, 3):
        estrategia, _ = _ejecutar_kqnodes(k)
        orden_nodos = orden_canonico_nodos(
            estrategia.sia_subsistema.dims_ncubos,
            estrategia.sia_subsistema.indices_ncubos,
        )
        # No debe lanzar: cobertura exacta, sin duplicados, y exactamente k bloques.
        validar_particion(estrategia.particion_final, orden_nodos, k=k)
        assert len(estrategia.particion_final) == k

        nodos_cubiertos = {
            (int(t), int(i)) for bloque in estrategia.particion_final for (t, i) in bloque
        }
        assert nodos_cubiertos == {(int(t), int(i)) for (t, i) in orden_nodos}


def test_kqnodes_es_determinista():
    """Dos corridas independientes dan la misma partición y el mismo δ."""
    for k in (2, 3):
        est_a, sol_a = _ejecutar_kqnodes(k)
        est_b, sol_b = _ejecutar_kqnodes(k)
        assert est_a.particion_final == est_b.particion_final
        assert sol_a.particion == sol_b.particion
        assert abs(sol_a.perdida - sol_b.perdida) <= TOLERANCIA


def test_kqnodes_k2_no_supera_al_optimo_de_fuerza_bruta():
    """
    Sanidad k=2: KQNodes y BruteForce recorren el MISMO espacio bipartito (todas
    las 2-particiones en dos bloques no vacíos; `kpartir` con 2 bloques == `bipartir`,
    ya verificado en test_kpartir_consistencia). Como BruteForce es exhaustivo y
    KQNodes voraz, debe cumplirse δ_voraz >= δ_óptimo (con tolerancia).
    """
    _, sol_kq = _ejecutar_kqnodes(2)

    analizador_bf = BruteForce(_cargar_tpm_n3a())
    sol_bf = analizador_bf.aplicar_estrategia(
        ESTADO_INICIAL, CONDICION, ALCANCE, MECANISMO
    )

    _imprimir(
        f"\n[N3A] k=2  KQNodes delta = {sol_kq.perdida:.6f}  |  "
        f"BruteForce delta optimo = {sol_bf.perdida:.6f}"
    )

    assert sol_kq.perdida >= sol_bf.perdida - TOLERANCIA, (
        f"El voraz quedo por DEBAJO del optimo exacto: "
        f"{sol_kq.perdida} < {sol_bf.perdida}"
    )

    coinciden = abs(sol_kq.perdida - sol_bf.perdida) <= TOLERANCIA
    _imprimir(
        "[N3A] k=2  KQNodes "
        + ("COINCIDE con" if coinciden else "queda por encima de")
        + " el optimo de BruteForce."
    )


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
    print(f"\n{len(pruebas)} pruebas de KQNodes pasaron.")


if __name__ == "__main__":
    _ejecutar_todas()
