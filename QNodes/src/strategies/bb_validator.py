import time
from typing import List, Tuple

import numpy as np

from src.constants.base import INFTY_POS
from src.funcs.iit import emd_efecto
from src.funcs.particion import (
    bloques_desde_etiquetas,
    clave_canonica,
    orden_canonico_nodos,
)
from src.middlewares.slogger import SafeLogger
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.strategies.k_qnodes import KQNodes


class BBValidator(SIA):
    """
    Branch & Bound exact validator for k-partitions using canonical restricted
    growth strings (RGS) to avoid symmetric duplicates. For small `m` it
    exhaustively enumerates canonical labelings and returns the optimal δ. If
    the run exceeds `timeout_sec` it returns a non-optimal sentinel partition
    (`"NO_OPTIMO"`).

    Note: To limit search to small networks, caller must pass `n_max_nodes`.
    """

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        self.logger = SafeLogger("BB-Validator")

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int,
        *,
        n_max_nodes: int = 10,
        timeout_sec: float = 300.0,
    ) -> Solution:
        """
        Ejecuta un B&B por backtracking enumerando sólo vectores canónicos (RGS)
        de longitud `m` con exactamente `k` bloques.

        - `n_max_nodes` limita m para evitar ejecuciones en redes grandes.
        - `timeout_sec` hace que la función devuelva una solución con partición
          igual a "NO_OPTIMO" si se supera el tiempo.
        """
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        orden_nodos = orden_canonico_nodos(
            self.sia_subsistema.dims_ncubos, self.sia_subsistema.indices_ncubos
        )
        m = len(orden_nodos)
        if m > n_max_nodes:
            # No correr en redes grandes.
            return Solution(
                estrategia="BB-Validator",
                perdida=INFTY_POS,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=np.array([]),
                particion="NO_OPTIMO",
                tiempo_total=0.0,
                quiere_hablar=False,
            )

        self.sia_tiempo_inicio = time.time()
        deadline = self.sia_tiempo_inicio + timeout_sec

        # Initialize upper bound with KQNodes greedy solution to speed pruning
        try:
            kq = KQNodes(self.tpm)
            sol_kq = kq.aplicar_estrategia(
                estado_inicial, condicion, alcance, mecanismo, k
            )
            best_delta = sol_kq.perdida
        except Exception:
            best_delta = INFTY_POS
        best_part = None

        # Memoiza claves canónicas ya evaluadas.
        seen = set()

        # Backtracking over restricted growth strings (canonical labelings).
        vector: List[int] = [0] * m

        def backtrack(pos: int, max_label: int):
            nonlocal best_delta, best_part
            # Timeout check
            if time.time() > deadline:
                raise TimeoutError()

            if pos == m:
                # Require exactly k labels used (max_label indexed from 0)
                if max_label + 1 != k:
                    return
                bloques = bloques_desde_etiquetas(vector, orden_nodos)
                clave = clave_canonica(bloques, orden_nodos)
                if clave in seen:
                    return
                seen.add(clave)

                dist = self.sia_subsistema.kpartir(bloques).distribucion_marginal()
                delta = emd_efecto(dist, self.sia_dists_marginales)
                if delta < best_delta:
                    best_delta = delta
                    best_part = [b for b in bloques]
                return

            # Prune by impossibility to reach k labels: even if every remaining
            # position introduces a new label, cannot reach k.
            remaining = m - pos
            if (max_label + 1) + remaining < k:
                return

            # Lower bound (optimistic): we cannot guarantee any negative
            # contribution from remaining nodes, so the optimistic lower bound
            # is 0. This is a safe (but weak) lower bound: only prune when
            # 0 >= best_delta which is rare, but the hook is here for future
            # stronger relaxations.
            lower_bound = 0.0
            if lower_bound >= best_delta:
                return

            # Allowed labels: 0..max_label and possibly max_label+1 (if < k)
            for label in range(0, max_label + 2):
                if label >= k:
                    break
                vector[pos] = label
                new_max = max_label
                if label == max_label + 1:
                    new_max = max_label + 1
                backtrack(pos + 1, new_max)

        try:
            backtrack(0, -1)
        except TimeoutError:
            return Solution(
                estrategia="BB-Validator",
                perdida=INFTY_POS,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=np.array([]),
                particion="NO_OPTIMO",
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                quiere_hablar=False,
            )

        if best_part is None:
            return Solution(
                estrategia="BB-Validator",
                perdida=INFTY_POS,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=np.array([]),
                particion="NO_OPTIMO",
                tiempo_total=time.time() - self.sia_tiempo_inicio,
                quiere_hablar=False,
            )

        particion_fmt = str(best_part)
        dist_final = self.sia_subsistema.kpartir(best_part).distribucion_marginal()

        return Solution(
            estrategia="BB-Validator",
            perdida=best_delta,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_final,
            particion=particion_fmt,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            quiere_hablar=False,
        )
