
import sys
import numpy as np
from pathlib import Path
import gc

# Ajustar el path para que los imports `src.*` funcionen desde la raíz del proyecto
ROOT = Path(__file__).resolve().parent
QNODES_ROOT = ROOT / "QNodes"
QNODES_SRC = QNODES_ROOT / "src"
if str(QNODES_ROOT) not in sys.path:
    sys.path.insert(0, str(QNODES_ROOT))
if str(QNODES_SRC) not in sys.path:
    sys.path.insert(0, str(QNODES_SRC))

# Importar el orquestador y las estrategias
from src.controllers.manager import Manager
from src.strategies.k_qnodes import KQNodes
from src.strategies.k_geomip import KGeoMIP
from src.funcs.iit import emd_efecto
from src.funcs.particion import orden_canonico_nodos
from src.constants.base import ACTUAL, EFFECT

def generar_particion_trivial(N, K):
    """
    Genera una k-partición trivial siguiendo la directriz:
    - Aísla los primeros K-1 nodos futuros como singletons.
    - Agrupa todos los nodos restantes en un único bloque.

    Nota: `System.kpartir` espera una lista de bloques de nodos en la forma
    `(tiempo, indice)`, donde el tiempo es `ACTUAL` o `EFFECT`.
    """
    # Construir el orden canónico de nodos para un subsistema completo de N variables.
    orden_nodos = [
        (ACTUAL, i) for i in range(N)
    ] + [
        (EFFECT, i) for i in range(N)
    ]

    # Si k=2, usamos un bloque con todos los presentes y todos los futuros excepto el primero,
    # y un bloque singleton con el primer futuro. Para k>2, separamos los primeros K-1 futuros
    # como singletons y agrupamos el resto junto con todos los presentes.
    bloques = []
    future_nodes = orden_nodos[N:]
    for i in range(K - 1):
        bloques.append([future_nodes[i]])

    ultimo_bloque = [nodo for nodo in orden_nodos[:N]] + future_nodes[K - 1:]
    bloques.append(ultimo_bloque)
    return bloques

def evaluar_caso(red_nombre, k, estrategia):
    # Deducir N del nombre de la red
    N = int(''.join(filter(str.isdigit, red_nombre)))
    
    # Cargar TPM
    ruta_tpm = Path("src") / ".samples" / red_nombre
    if not ruta_tpm.exists():
        ruta_tpm = Path("QNodes") / "src" / ".samples" / red_nombre
        
    tpm = np.genfromtxt(ruta_tpm, delimiter=",", dtype=np.float32)
    manager = Manager(estado_inicial="1" + "0"*(N-1))
    
    # Configurar máscaras de subsistema completo
    condiciones = "1" * N
    alcance = "1" * N
    mecanismo = "1" * N
    
    # 1. Instanciar y ejecutar la estrategia del algoritmo
    if estrategia == "KQNodes":
        est = KQNodes(tpm)
    else:
        est = KGeoMIP(tpm)

    solucion = est.aplicar_estrategia(manager.estado_inicial, condiciones, alcance, mecanismo, k)
    perdida_algoritmo = float(solucion.perdida)

    # 2. Generar y evaluar la partición trivial usando el mismo subsistema de la estrategia
    part_trivial = generar_particion_trivial(N, k)
    subsistema = est.sia_subsistema
    particion_trivial_sistema = subsistema.kpartir(part_trivial)
    perdida_trivial = float(emd_efecto(particion_trivial_sistema.distribucion_marginal(), subsistema.distribucion_marginal()))
    
    # 3. Calcular los Gaps
    gap_absoluto = perdida_trivial - perdida_algoritmo
    
    if perdida_trivial > 0:
        gap_relativo = (gap_absoluto / perdida_trivial) * 100
    else:
        gap_relativo = 0.0
        
    # Limpieza de memoria
    del est
    gc.collect()
    
    return perdida_algoritmo, perdida_trivial, gap_absoluto, gap_relativo, part_trivial, solucion.particion

def main():
    pruebas = [
        ("N6A.csv", 4, "KQNodes"),
        ("N6A.csv", 4, "KGeoMIP"),
        ("N10A.csv", 3, "KQNodes"),
        ("N10A.csv", 3, "KGeoMIP"),
        ("N10A.csv", 5, "KQNodes"),
        ("N10A.csv", 5, "KGeoMIP"),
    ]
    
    print(f"{'Estrategia':<10} | {'Red':<4} | {'K':<2} | {'P. Algoritmo':<12} | {'P. Trivial':<10} | {'Gap Abs.':<8} | {'Gap Rel. %':<10}")
    print("─────────────────────────────────────────────────────────────────────────")
    
    for red, k, est in pruebas:
        try:
            p_alg, p_triv, gap_abs, gap_rel, trivial_p, alg_p = evaluar_caso(red, k, est)
            print(f"{est:<10} | {red[:3]:<4} | {k:<2} | {p_alg:12.6f} | {p_triv:10.6f} | {gap_abs:8.6f} | {gap_rel:9.2f}%")
        except Exception as e:
            print(f"{est:<10} | {red[:3]:<4} | {k:<2} | Error en cálculo: {e}")
            
    

if __name__ == "__main__":
    main()