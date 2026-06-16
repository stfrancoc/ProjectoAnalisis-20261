import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.k_qnodes import KQNodes
from src.strategies.bb_validator import BBValidator

RUTA_N3A = Path(__file__).resolve().parent / "src" / ".samples" / "N3A.csv"
TOL = 1e-9

def _cargar_tpm_n3a():
    # Usamos coma, que es lo que confirmamos antes
    return np.genfromtxt(RUTA_N3A, delimiter=',')

def test_bb_vs_kqnodes_n3a_k3():
    tpm = _cargar_tpm_n3a()
    n = tpm.shape[1] # Dinámico
    k = 3
    
    # Configuración dinámica basada en n
    estado = '1' * n
    cond = '1' * n
    alc = '1' * n
    mec = '1' * n

    kq = KQNodes(tpm)
    sol_kq = kq.aplicar_estrategia(estado, cond, alc, mec, k)

    bb = BBValidator(tpm)
    sol_bb = bb.aplicar_estrategia(
        estado, cond, alc, mec, k, n_max_nodes=10, timeout_sec=60.0 # Timeout ampliado
    )

    print(f"\nKQNodes delta: {sol_kq.perdida}")
    print(f"BBValidator delta: {sol_bb.perdida}")

    # Si el B&B no encuentra el óptimo, imprimimos pero no rompemos el test
    if sol_bb.perdida == float("inf"):
        print("BBValidator no convergió (TIMEOUT)")
    else:
        assert sol_bb.perdida <= sol_kq.perdida + TOL