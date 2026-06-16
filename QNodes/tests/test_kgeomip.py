"""
Pruebas de la estrategia geométrica k-vía `KGeoMIP` (Fase 2).

Cubre, sobre la red de muestra N3A:
- Consistencia k=2: como `BruteForce` es exhaustivo sobre el mismo espacio
  bipartito y `KGeoMIP` solo evalúa unas pocas candidatas geométricas, su δ debe
  ser >= el δ óptimo exacto de `BruteForce` (se reporta si coinciden).
- Validez estructural para k=2,3,4: la partición devuelta pasa `validar_particion`
  con el k pedido, tiene exactamente k bloques no vacíos y cubre todos los nodos.
- Comparación `KGeoMIP` vs `KQNodes` para k=2,3,4: imprime ambas pérdidas, exige
  solo que sean >= 0 y reporta cuál gana en cada caso (son heurísticas distintas).
- Determinismo: dos corridas sobre N3A con k=3 dan la misma partición y δ.

Ejecutar con `uv run pytest tests/test_kgeomip.py -s` o directamente con
`uv run python tests/test_kgeomip.py`.
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
from src.strategies.k_geomip import KGeoMIP  # noqa: E402
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


def _ejecutar_kgeomip(k: int) -> tuple[KGeoMIP, object]:
    """Construye y ejecuta KGeoMIP sobre N3A para un `k` dado."""
    estrategia = KGeoMIP(_cargar_tpm_n3a())
    solucion = estrategia.aplicar_estrategia(
        ESTADO_INICIAL, CONDICION, ALCANCE, MECANISMO, k
    )
    return estrategia, solucion


def _ejecutar_kqnodes(k: int) -> tuple[KQNodes, object]:
    """Construye y ejecuta KQNodes sobre N3A para un `k` dado."""
    estrategia = KQNodes(_cargar_tpm_n3a())
    solucion = estrategia.aplicar_estrategia(
        ESTADO_INICIAL, CONDICION, ALCANCE, MECANISMO, k
    )
    return estrategia, solucion


def test_kgeomip_k2_no_supera_al_optimo_de_fuerza_bruta():
    """
    Consistencia k=2: KGeoMIP y BruteForce recorren el MISMO espacio bipartito
    (`kpartir` con 2 bloques == `bipartir`, ya verificado en test_kpartir_consistencia).
    Como BruteForce es exhaustivo y KGeoMIP solo evalúa candidatas geométricas, debe
    cumplirse δ_geo >= δ_óptimo (con tolerancia). Se reporta si coinciden.
    """
    _, sol_kg = _ejecutar_kgeomip(2)

    analizador_bf = BruteForce(_cargar_tpm_n3a())
    sol_bf = analizador_bf.aplicar_estrategia(
        ESTADO_INICIAL, CONDICION, ALCANCE, MECANISMO
    )

    _imprimir(
        f"\n[N3A] k=2  KGeoMIP delta = {sol_kg.perdida:.6f}  |  "
        f"BruteForce delta optimo = {sol_bf.perdida:.6f}"
    )

    assert sol_kg.perdida >= sol_bf.perdida - TOLERANCIA, (
        f"El geometrico quedo por DEBAJO del optimo exacto: "
        f"{sol_kg.perdida} < {sol_bf.perdida}"
    )

    coinciden = abs(sol_kg.perdida - sol_bf.perdida) <= TOLERANCIA
    _imprimir(
        "[N3A] k=2  KGeoMIP "
        + ("COINCIDE con" if coinciden else "queda por encima de")
        + " el optimo de BruteForce."
    )


def test_kgeomip_devuelve_particion_valida_con_k_bloques():
    """La partición final es válida, tiene exactamente k bloques y cubre todo."""
    for k in (2, 3, 4):
        estrategia, solucion = _ejecutar_kgeomip(k)
        _imprimir(f"\n[N3A] KGeoMIP k={k}  ->  delta = {solucion.perdida:.6f}")
        _imprimir(solucion.particion)

        orden_nodos = orden_canonico_nodos(
            estrategia.sia_subsistema.dims_ncubos,
            estrategia.sia_subsistema.indices_ncubos,
        )
        # No debe lanzar: cobertura exacta, sin duplicados y exactamente k bloques.
        validar_particion(estrategia.particion_final, orden_nodos, k=k)
        assert len(estrategia.particion_final) == k

        nodos_cubiertos = {
            (int(t), int(i))
            for bloque in estrategia.particion_final
            for (t, i) in bloque
        }
        assert nodos_cubiertos == {(int(t), int(i)) for (t, i) in orden_nodos}


def test_kgeomip_vs_kqnodes_en_n3a():
    """Compara KGeoMIP vs KQNodes para k=2,3,4; reporta cuál gana en cada caso."""
    for k in (2, 3, 4):
        _, sol_kg = _ejecutar_kgeomip(k)
        _, sol_kq = _ejecutar_kqnodes(k)

        assert sol_kg.perdida >= -TOLERANCIA
        assert sol_kq.perdida >= -TOLERANCIA

        if abs(sol_kg.perdida - sol_kq.perdida) <= TOLERANCIA:
            veredicto = "EMPATE"
        elif sol_kg.perdida < sol_kq.perdida:
            veredicto = "gana KGeoMIP"
        else:
            veredicto = "gana KQNodes"

        _imprimir(
            f"\n[N3A] k={k}  KGeoMIP delta = {sol_kg.perdida:.6f}  |  "
            f"KQNodes delta = {sol_kq.perdida:.6f}  ->  {veredicto}"
        )


def test_kgeomip_es_determinista():
    """Dos corridas independientes sobre N3A k=3 dan la misma partición y δ."""
    est_a, sol_a = _ejecutar_kgeomip(3)
    est_b, sol_b = _ejecutar_kgeomip(3)
    assert est_a.particion_final == est_b.particion_final
    assert sol_a.particion == sol_b.particion
    assert abs(sol_a.perdida - sol_b.perdida) <= TOLERANCIA


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
    print(f"\n{len(pruebas)} pruebas de KGeoMIP pasaron.")


if __name__ == "__main__":
    _ejecutar_todas()
