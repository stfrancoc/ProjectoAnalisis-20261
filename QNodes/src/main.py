from src.controllers.manager import Manager

# 👇 Importación de estrategias 👇 #
from src.strategies.force import BruteForce
from src.strategies.k_qnodes import KQNodes


def iniciar():
    """Punto de entrada"""

    # ABCD #
    estado_inicial = "1000"
    condiciones =    "1110"
    alcance =        "1110"
    mecanismo =      "1110"

    # Selector mínimo de estrategia. Cambia `usar_kqnodes` a True para ejecutar la
    # estrategia voraz de k-particiones (KQNodes) con el `k` indicado; en False se
    # mantiene la ejecución original por fuerza bruta (biparticiones).
    usar_kqnodes = True
    k = 3

    gestor_redes = Manager(estado_inicial)
    mpt = gestor_redes.cargar_red()

    if usar_kqnodes:
        ### Solución mediante la estrategia voraz de k-particiones (KQNodes) ###
        analizador_kq = KQNodes(mpt)
        solucion = analizador_kq.aplicar_estrategia(
            estado_inicial,
            condiciones,
            alcance,
            mecanismo,
            k,
        )
    else:
        ### Ejemplo de solución mediante módulo de fuerza bruta ###
        analizador_bf = BruteForce(mpt)
        solucion = analizador_bf.aplicar_estrategia(
            estado_inicial,
            condiciones,
            alcance,
            mecanismo,
        )

    print(solucion)
