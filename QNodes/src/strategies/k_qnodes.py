import time

import numpy as np

from src.constants.base import INFTY_POS
from src.constants.error import ERROR_PARTICION_K_RANGO
from src.constants.models import KQNODES_STRAREGY_TAG, KQNODES_LABEL
from src.funcs.format import fmt_kparticion
from src.funcs.iit import emd_efecto
from src.funcs.particion import clave_canonica, orden_canonico_nodos
from src.middlewares.slogger import SafeLogger
from src.models.base.sia import SIA
from src.models.core.solution import Solution

# Un nodo es una tupla (tiempo, indice); un bloque es una lista de nodos.
Nodo = tuple[int, int]
Bloque = list[Nodo]


class KQNodes(SIA):
    """
    Estrategia voraz aglomerativa para el problema MIP generalizado a k-particiones.

    A diferencia de `QNodes` (que resuelve biparticiones), `KQNodes` busca la
    k-partición —con `k` fijado por el usuario— que minimice la pérdida de
    información entre la dinámica del subsistema y la dinámica reconstruida al
    recombinar los `k` bloques. La pérdida se mide con la EMD-efecto (ver
    `src/funcs/iit.py`).

    Algoritmo (aglomerativo / bottom-up):
    ------------------------------------
    1. Se parte de tantos grupos como nodos (un singleton por cada nodo presente
       `(ACTUAL, j)` y cada nodo futuro `(EFFECT, i)`).
    2. Mientras haya más de `k` grupos, se evalúan TODAS las fusiones de pares de
       grupos. Como la EMD NO es descomponible por bloques, cada fusión candidata se
       evalúa con el δ GLOBAL de la partición completa resultante
       (`System.kpartir(...).distribucion_marginal()` + `emd_efecto(...)`). Se
       fusiona el par que produzca el menor δ global.
    3. Al quedar exactamente `k` grupos, se reporta esa partición y su δ.

    Determinismo: los grupos se mantienen siempre en orden canónico y las fusiones
    se recorren en ese orden con comparación estricta `<`, de modo que ante empates
    gana el primer par en orden canónico. Así dos corridas dan idéntico resultado.

    Memoización: `memoria_particiones` indexa por la clave canónica de la partición
    (ver `src/funcs/particion.py`), evitando reevaluar particiones equivalentes
    (que sólo difieren en el nombre de los bloques).

    Args:
    ----
        tpm (np.ndarray): Matriz de transición de probabilidad de la red.

    Attributes:
    ----------
        memoria_particiones (dict[tuple[int, ...], tuple[float, np.ndarray]]):
            Caché de `clave_canonica -> (δ, distribución marginal)` de la estrategia.
        logger (SafeLogger): Logger de la estrategia.
    """

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        self.memoria_particiones: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
        # Última k-partición hallada (lista de bloques), útil para inspección y tests.
        self.particion_final: list[Bloque] = []
        self.logger = SafeLogger(KQNODES_STRAREGY_TAG)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int,
    ) -> Solution:
        """
        Ejecuta la búsqueda voraz aglomerativa de la mejor k-partición.

        Args:
        ----
            estado_inicial (str): Estado inicial de la red (cadena de bits).
            condicion (str): Condiciones de fondo (bits en 0 se condicionan).
            alcance (str): Alcance/futuro (bits en 0 se substraen).
            mecanismo (str): Mecanismo/presente (bits en 0 se substraen).
            k (int): Número de bloques deseado, con `2 <= k <= m` (m = nº de nodos
                particionables). No se admite `k=1` (partición trivial, δ=0).

        Returns:
        -------
            Solution: La solución con la k-partición de menor pérdida hallada por el
            algoritmo voraz, su δ, las distribuciones y el tiempo de ejecución. Se
            construye con `quiere_hablar=False` para no disparar la síntesis de voz.

        Raises:
        ------
            ValueError: Si `k` está fuera del rango `[2, m]`.
        """
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        # Universo de nodos en orden canónico (presentes y luego futuros).
        orden_nodos = orden_canonico_nodos(
            self.sia_subsistema.dims_ncubos,
            self.sia_subsistema.indices_ncubos,
        )
        m = len(orden_nodos)
        if not (2 <= k <= m):
            raise ValueError(ERROR_PARTICION_K_RANGO(k, m))

        # Inicio: un grupo (singleton) por cada nodo.
        grupos: list[Bloque] = [[nodo] for nodo in orden_nodos]

        # Bucle aglomerativo: fusiona el par de menor δ global hasta quedar con k.
        while len(grupos) > k:
            mejor_delta = INFTY_POS
            mejor_par: tuple[int, int] = (0, 1)
            for a in range(len(grupos)):
                for b in range(a + 1, len(grupos)):
                    candidata = self._fusionar(grupos, a, b)
                    delta, _ = self._evaluar(candidata, orden_nodos)
                    if delta < mejor_delta:
                        mejor_delta = delta
                        mejor_par = (a, b)
            grupos = self._fusionar(grupos, *mejor_par)

        # Partición final de k grupos: su δ y distribución (memoizados).
        self.particion_final = grupos
        delta_final, dist_final = self._evaluar(grupos, orden_nodos)
        particion_fmt = fmt_kparticion(grupos)

        self.logger.info(
            f"KQNodes(k={k}) δ={delta_final:.6f} sobre {m} nodos.\n{particion_fmt}"
        )

        return Solution(
            estrategia=KQNODES_LABEL,
            perdida=delta_final,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_final,
            particion=particion_fmt,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            quiere_hablar=False,
        )

    def _evaluar(
        self,
        bloques: list[Bloque],
        orden_nodos: list[Nodo],
    ) -> tuple[float, np.ndarray]:
        """
        Calcula (con memoización) el δ global de una partición y su distribución.

        El δ es `emd_efecto(distribucion_particion, sia_dists_marginales)`, donde la
        distribución de la partición sale de `System.kpartir(bloques)`. Se memoiza
        por la clave canónica de la partición para no reevaluar equivalentes.

        Args:
        ----
            bloques (list[Bloque]): La partición a evaluar.
            orden_nodos (list[Nodo]): Orden canónico de los nodos del subsistema.

        Returns:
        -------
            tuple[float, np.ndarray]: El par `(δ, distribución marginal)`.
        """
        clave = clave_canonica(bloques, orden_nodos)
        if clave not in self.memoria_particiones:
            distribucion = self.sia_subsistema.kpartir(bloques).distribucion_marginal()
            delta = emd_efecto(distribucion, self.sia_dists_marginales)
            self.memoria_particiones[clave] = (delta, distribucion)
        return self.memoria_particiones[clave]

    def _fusionar(
        self,
        grupos: list[Bloque],
        a: int,
        b: int,
    ) -> list[Bloque]:
        """
        Devuelve una NUEVA lista de grupos con los grupos `a` y `b` fusionados.

        El resultado se reordena canónicamente (nodos ordenados dentro de cada bloque
        y bloques ordenados entre sí) para que el recorrido de fusiones sea
        determinista y los empates se rompan por orden canónico.

        Args:
        ----
            grupos (list[Bloque]): Lista de grupos actual.
            a (int): Índice del primer grupo a fusionar.
            b (int): Índice del segundo grupo a fusionar.

        Returns:
        -------
            list[Bloque]: La nueva lista de grupos, en orden canónico.
        """
        restantes = [grupo for idx, grupo in enumerate(grupos) if idx not in (a, b)]
        restantes.append(grupos[a] + grupos[b])
        return self._ordenar_grupos(restantes)

    @staticmethod
    def _ordenar_grupos(grupos: list[Bloque]) -> list[Bloque]:
        """
        Ordena canónicamente una lista de grupos para garantizar determinismo.

        Ordena los nodos dentro de cada bloque y luego los bloques entre sí
        lexicográficamente. Como `ACTUAL=0 < EFFECT=1`, dentro de un bloque los
        nodos presentes preceden a los futuros.

        Args:
        ----
            grupos (list[Bloque]): Lista de grupos a ordenar.

        Returns:
        -------
            list[Bloque]: La lista de grupos en orden canónico.
        """
        return sorted([sorted(bloque) for bloque in grupos])
