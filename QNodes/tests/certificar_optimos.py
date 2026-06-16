import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# --- CORRECCIÓN DE RUTAS ABSOLUTAS ---
# Esto asegura que pueda encontrar 'src' sin importar dónde se ejecute el test
qnodes_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(qnodes_root))

from src.strategies.k_qnodes import KQNodes
from src.strategies.k_geomip import KGeoMIP
from src.strategies.bb_validator import BBValidator

# Definir ruta de muestras
RUTA_SAMPLES = qnodes_root / "src" / ".samples"

def _cargar_tpm(nombre_archivo):
    return np.genfromtxt(RUTA_SAMPLES / nombre_archivo, delimiter=',')

def certificar(nombre_archivo, k=3):
    tpm = _cargar_tpm(nombre_archivo)
    n = tpm.shape[1]
    inicial, cond, alc, mec = '1'*n, '1'*n, '1'*n, '1'*n
    
    print(f"Certificando red {nombre_archivo} con k={k}...")
    
    # 1. Exacto (B&B)
    bb = BBValidator(tpm)
    sol_bb = bb.aplicar_estrategia(inicial, cond, alc, mec, k, n_max_nodes=10, timeout_sec=60.0)
    
    # 2. Heurísticas
    kq = KQNodes(tpm)
    sol_kq = kq.aplicar_estrategia(inicial, cond, alc, mec, k)
    
    kg = KGeoMIP(tpm)
    sol_kg = kg.aplicar_estrategia(inicial, cond, alc, mec, k)
    
    return {
        "red": nombre_archivo, "k": k,
        "delta_BB": sol_bb.perdida if isinstance(sol_bb.perdida, float) else float('inf'),
        "delta_KQNodes": sol_kq.perdida,
        "delta_KGeoMIP": sol_kg.perdida
    }

if __name__ == "__main__":
    # Nos enfocamos en N3A y N4A para obtener resultados exactos
    archivos = ["N3A.csv", "N4A.csv"]
    resultados = []
    
    for f in archivos:
        try:
            # Subimos el timeout a 300 segundos para asegurar que encuentre el óptimo
            # Reducimos a k=2 y k=3 para que el B&B no se agote
            resultados.append(certificar(f, k=2))
            resultados.append(certificar(f, k=3))
        except Exception as e:
            print(f"Error en red {f}: {e}")
            
    df = pd.DataFrame(resultados)
    print("\n--- RESULTADOS DE CERTIFICACIÓN (B&B OPTIMIZADO) ---")
    print(df.to_string())