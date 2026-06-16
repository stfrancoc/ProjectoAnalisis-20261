import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from collections import deque
import gc

from src.constants.base import ACTUAL, EFFECT, INFTY_POS, KEY_KGEOMIP
from src.constants.error import ERROR_PARTICION_K_RANGO
from src.constants.models import KGEOMIP_LABEL, KGEOMIP_STRAREGY_TAG
from src.funcs.format import fmt_kparticion
from src.funcs.iit import emd_efecto
from src.funcs.particion import (
    clave_canonica,
    orden_canonico_nodos,
    validar_particion,
)
from src.middlewares.slogger import SafeLogger
from src.models.base.sia import SIA
from src.models.core.solution import Solution

# Un nodo es una tupla (tiempo, indice); un bloque es una lista de nodos.
Nodo = Tuple[int, int]
Bloque = List[Nodo]


class KGeoMIP(SIA):
    """
    Estrategia geométrica para el problema MIP generalizada a k-particiones.

    Porta el núcleo del método geométrico-topológico de GeoMIP (codebase
    `GeoMIP/.../geometric.py`) hacia el codebase de QNodes y lo extiende de
    biparticiones a `k` bloques arbitrarios. La idea central de GeoMIP NO cambia
    entre `k=2` y `k>2`: se construye una sola vez por subsistema una tabla de
    costos de transición `t(i, j)` que recorre, nivel a nivel (BFS por distancia de
    Hamming), el hipercubo de estados entre el estado inicial y su complemento. Ese
    coste por variable futura es la señal geométrica que guía el particionamiento.

    - Para `k=2` se reproduce EXACTAMENTE la generación de candidatas de GeoMIP
      (`identificar_particiones_optimas`): un candidato de "nivel cero" por cada
      variable futura aislada, más el mejor candidato por nivel del BFS.
    - Para `k>2` se extiende geométricamente: se ordenan las variables futuras por
      su costo y se cortan en `k` grupos por los `k-1` saltos de costo más grandes;
      las variables presentes acompañan a su futura homóloga (o, si no la hay, al
      bloque de menor costo promedio). Se añaden variantes de "nivel cero" y un
      candidato general de respaldo (agrupamiento por saltos sobre TODOS los nodos)
      que garantiza al menos una k-partición válida cuando `k` supera el número de
      variables futuras.

    Cada candidata se traduce a bloques `(ACTUAL, j)`/`(EFFECT, i)`, se valida con
    `validar_particion`, se evalúa con `System.kpartir(...)` + `emd_efecto(...)` y se
    memoiza por su clave canónica; se devuelve la de menor pérdida.

    Args:
    ----
        tpm (np.ndarray): Matriz de transición de probabilidad de la red.

    Attributes:
    ----------
        _flat_data (list[np.ndarray]): Datos aplanados de cada n-cubo del subsistema.
        tabla_transiciones (dict): Caché de costos de transición geométricos,
            indexado por `(tuple(estado_ini), tuple(estado_fin))`.
        caminos (dict[int, list]): Estados alcanzados por nivel del BFS de Hamming.
        memoria_particiones (dict): Caché `clave -> (pérdida, distribución)`.
        vertices (set): Conjunto de nodos del subsistema.
        estado_inicial (np.ndarray): Estado inicial restringido a las dims del
            subsistema (NO el string completo del usuario).
        estado_final (np.ndarray): Complemento bit a bit del estado inicial.
        idx_ncubos (list[int]): Posiciones de las variables futuras (0..n_vars-1).
    """

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        self._flat_data: List[np.ndarray] = []
        self.tabla_transiciones: Dict[tuple, list] = {}
        self.caminos: Dict[int, List[list]] = {}
        self.memoria_particiones: Dict[tuple, Tuple[float, np.ndarray]] = {}
        self.vertices: set = set()
        self.estado_inicial: np.ndarray
        self.estado_final: np.ndarray
        self.idx_ncubos: List[int] = []
        # Última k-partición hallada (lista de bloques), útil para inspección y tests.
        self.particion_final: List[Bloque] = []
        self.logger = SafeLogger(KGEOMIP_STRAREGY_TAG)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int,
    ) -> Solution:
        """
        Ejecuta la búsqueda geométrica k-vía de la mejor k-partición.

        Args:
        ----
            estado_inicial (str): Estado inicial de la red (cadena de bits).
            condicion (str): Condiciones de fondo (bits en 0 se condicionan).
            alcance (str): Alcance/futuro (bits en 0 se substraen).
            mecanismo (str): Mecanismo/presente (bits en 0 se substraen).
            k (int): Número de bloques deseado, con `2 <= k <= m` (m = nº de nodos
                particionables). No se admite `k=1` (partición trivial).

        Returns:
        -------
            Solution: La solución con la k-partición de menor pérdida hallada por la
            heurística geométrica, su δ, las distribuciones y el tiempo de ejecución.
            Se construye con `quiere_hablar=False` para no disparar la voz.

        Raises:
        ------
            ValueError: Si `k` está fuera del rango `[2, m]`.
        """
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        futuro = tuple((EFFECT, i) for i in self.sia_subsistema.indices_ncubos)
        presente = tuple((ACTUAL, j) for j in self.sia_subsistema.dims_ncubos)
        self.vertices = set(presente + futuro)

        orden_nodos = orden_canonico_nodos(
            self.sia_subsistema.dims_ncubos,
            self.sia_subsistema.indices_ncubos,
        )
        m = len(orden_nodos)
        if not (2 <= k <= m):
            raise ValueError(ERROR_PARTICION_K_RANGO(k, m))

        # Núcleo geométrico (idéntico para todo k): aplanar datos y construir la
        # tabla de costos de transición recorriendo el hipercubo nivel a nivel.
        self._construir_flat_data()
        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial
        self._construir_tabla_transiciones()

        # Generación, validación, evaluación (kpartir + EMD) y selección.
        bloques_ganadores, perdida, dist = self._mejor_particion(orden_nodos, k)
        self.particion_final = bloques_ganadores
        particion_fmt = fmt_kparticion(bloques_ganadores)

        self.logger.info(
            f"KGeoMIP(k={k}) δ={perdida:.6f} sobre {m} nodos.\n{particion_fmt}"
        )

        return Solution(
            estrategia=KGEOMIP_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            particion=particion_fmt,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            quiere_hablar=False,
        )

    # ------------------------------------------------------------------ #
    # Núcleo geométrico portado de GeoMIP (sin cambios de lógica)         #
    # ------------------------------------------------------------------ #

    def _construir_flat_data(self) -> None:
        """
        Aplana los datos de cada n-cubo del subsistema en `self._flat_data`.

        Cada `ncubo.data` tiene forma `(2, 2, ..., 2)`; `ravel()` lo lleva a un
        vector lineal indexable por el entero del estado (little-endian), de modo
        que `flat[estado_int]` da directamente la probabilidad de transición.
        """
        # Build a single 2D float32 matrix (n_vars x 2^m) to avoid many small
        # Python objects and allow fast vectorized indexing. Use in-place
        # assignments when possible to avoid unnecessary copies.
        n_vars = len(self.sia_subsistema.ncubos)
        if n_vars == 0:
            self._flat_matrix = np.empty((0, 1), dtype=np.float32)
            return
        size = self.sia_subsistema.ncubos[0].data.size
        mat = np.empty((n_vars, size), dtype=np.float32)
        for i, ncubo in enumerate(self.sia_subsistema.ncubos):
            row = ncubo.data.ravel()
            if row.dtype != np.float32:
                row = row.astype(np.float32)
            mat[i, :] = row
        # replace the old list with a single matrix to save memory and speed
        self._flat_matrix = mat

    def _construir_tabla_transiciones(self) -> None:
        """
        Llena `self.tabla_transiciones` recorriendo el hipercubo nivel a nivel.

        Réplica del bloque de `find_mip` (GeoMIP) que inicializa los caminos y la
        tabla y va calculando los costos por niveles de distancia de Hamming desde
        el estado inicial hacia el final.
        """
        estado_inicial = self.estado_inicial
        estado_final = self.estado_final
        n_vars = len(self.sia_subsistema.indices_ncubos)

        self.idx_ncubos = list(range(n_vars))
        self.caminos = {0: [estado_inicial.tolist()]}
        # store arrays as numpy float32 for memory efficiency and fast ops
        self.tabla_transiciones[(tuple(self.caminos[0][0]), tuple(self.caminos[0][0]))] = np.zeros(
            n_vars, dtype=np.float32
        )

        for nivel in range(1, len(estado_inicial) + 1):
            self.calcular_costos_nivel(estado_final, nivel)

        # collect garbage after heavy table construction
        gc.collect()

    def calcular_costos_nivel(self, estado_final: np.ndarray, nivel: int) -> None:
        """
        Calcula los costos del `nivel`-ésimo anillo de Hamming (BFS hacia el final).

        Para cada estado del nivel anterior, voltea un bit hacia `estado_final` para
        generar los estados del nivel actual (sin repetir) y dispara el cálculo de
        su costo de transición desde el estado inicial. Portado de GeoMIP.
        """
        # BFS by levels using an explicit queue (deque) to avoid recursion
        n = len(estado_final)
        visitados: set = set()
        self.caminos[nivel] = []
        dq = deque(self.caminos[nivel - 1])
        while dq:
            estado_anterior = dq.popleft()
            estado_actual = np.array(estado_anterior)
            for i in range(n):
                if estado_actual[i] != estado_final[i]:
                    nuevo_estado = estado_actual.copy()
                    nuevo_estado[i] = estado_final[i]
                    nuevo_state_tuple = tuple(nuevo_estado)
                    if nuevo_state_tuple not in visitados:
                        self.caminos[nivel].append(nuevo_estado.tolist())
                        self.calcular_costo(self.caminos[0][0], nuevo_estado.tolist(), self.idx_ncubos)
                        visitados.add(nuevo_state_tuple)

    def calcular_costo(
        self,
        estado_inicial: list,
        estado_final: list,
        ncubos: List[int],
    ) -> None:
        """
        Calcula el costo de transición del estado inicial al final por variable.

        Aplica la función recursiva `t(i, j) = γ·(|X[i] - X[j]| + Σ t(k, j))`, con
        `γ = 1/2^dH` (`dH` la distancia de Hamming), usando `self.tabla_transiciones`
        como caché. Portado idéntico de GeoMIP.
        """
        key = (tuple(estado_inicial), tuple(estado_final))
        n_vars = len(self.sia_subsistema.indices_ncubos)
        if key not in self.tabla_transiciones:
            self.tabla_transiciones[key] = np.empty(n_vars, dtype=np.float32)

        distancia_hamming = self.hamming(estado_inicial, estado_final)
        factor = np.float32(1.0 / (2 ** distancia_hamming))

        estado_ini_int = int("".join(map(str, estado_inicial[::-1])), 2)
        estado_fin_int = int("".join(map(str, estado_final[::-1])), 2)

        # Vectorized difference using the prebuilt matrix (n_vars x 2^m)
        diffs = np.abs(self._flat_matrix[:, estado_ini_int] - self._flat_matrix[:, estado_fin_int])
        # store float32 vector
        self.tabla_transiciones[key][:] = diffs.astype(np.float32)

        if distancia_hamming > 1:
            # Sum costs from intermediate neighbors (already computed at smaller Hamming)
            for i in range(len(estado_inicial)):
                if estado_inicial[i] != estado_final[i]:
                    nuevo_estado = estado_final.copy()
                    nuevo_estado[i] = estado_inicial[i]
                    temp_key = (tuple(estado_inicial), tuple(nuevo_estado))
                    temp_arr = self.tabla_transiciones.get(temp_key)
                    if temp_arr is not None:
                        # in-place addition to avoid allocations
                        self.tabla_transiciones[key] += temp_arr

        # apply factor in-place
        self.tabla_transiciones[key] *= factor

    def hamming(self, a: List[int], b: List[int]) -> int:
        """Distancia de Hamming entre dos estados binarios. Portado de GeoMIP."""
        return sum(x != y for x, y in zip(a, b))

    # ------------------------------------------------------------------ #
    # Generación de candidatas, evaluación y selección                   #
    # ------------------------------------------------------------------ #

    def _mejor_particion(
        self,
        orden_nodos: List[Nodo],
        k: int,
    ) -> Tuple[List[Bloque], float, np.ndarray]:
        """
        Genera candidatas, las valida/evalúa con `kpartir` y devuelve la mejor.

        Args:
        ----
            orden_nodos (list[Nodo]): Orden canónico de los nodos del subsistema.
            k (int): Número de bloques deseado.

        Returns:
        -------
            tuple[list[Bloque], float, np.ndarray]: La partición ganadora, su pérdida
            δ y su distribución marginal.

        Raises:
        ------
            ValueError: Si ninguna candidata generada resulta una k-partición válida
                (no debería ocurrir: el candidato general de respaldo siempre lo es).
        """
        candidatos = (
            self._candidatos_k2(orden_nodos)
            if k == 2
            else self._candidatos_k(orden_nodos, k)
        )

        mejor_perdida = INFTY_POS
        mejor_bloques: Optional[List[Bloque]] = None
        mejor_dist: Optional[np.ndarray] = None

        for bloques in candidatos:
            try:
                validar_particion(bloques, orden_nodos, k)
            except ValueError:
                continue

            clave = (KEY_KGEOMIP, *clave_canonica(bloques, orden_nodos))
            if clave not in self.memoria_particiones:
                dist = self.sia_subsistema.kpartir(bloques).distribucion_marginal()
                perdida = emd_efecto(dist, self.sia_dists_marginales)
                self.memoria_particiones[clave] = (perdida, dist)

            perdida, dist = self.memoria_particiones[clave]
            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_bloques = bloques
                mejor_dist = dist

        if mejor_bloques is None:
            raise ValueError(
                "KGeoMIP no halló ninguna k-partición válida entre las candidatas."
            )
        return mejor_bloques, mejor_perdida, mejor_dist

    def _candidatos_k2(self, orden_nodos: List[Nodo]) -> List[List[Bloque]]:
        """
        Candidatas para `k=2`: porta `identificar_particiones_optimas` de GeoMIP y
        traduce cada bipartición (posiciones presentes/futuras) a bloques para
        `kpartir` (el bloque generado y su complemento).
        """
        dims = self.sia_subsistema.dims_ncubos
        idxs = self.sia_subsistema.indices_ncubos

        bloques_lista: List[List[Bloque]] = []
        for presentes_pos, futuros_pos in self._identificar_particiones_optimas():
            presentes_labels = dims[np.asarray(presentes_pos, dtype=int)]
            futuros_labels = idxs[np.asarray(futuros_pos, dtype=int)]
            bloque_a: Bloque = [(ACTUAL, int(j)) for j in presentes_labels] + [
                (EFFECT, int(i)) for i in futuros_labels
            ]
            en_a = set(bloque_a)
            bloque_b: Bloque = [nodo for nodo in orden_nodos if nodo not in en_a]
            bloques_lista.append([bloque_a, bloque_b])
        return bloques_lista

    def _identificar_particiones_optimas(self) -> List[List[List[int]]]:
        """
        Genera candidatas de bipartición como pares `[presentes_pos, futuros_pos]`.

        Portado idéntico de `geometric.py`: candidatas de "nivel cero" (cada variable
        futura aislada) más, por cada nivel del BFS hasta la mitad, el estado de menor
        costo de corte. Las posiciones son índices en `dims_ncubos`/`indices_ncubos`.
        """
        key = tuple(self.caminos[0][0]), tuple(self.estado_final)
        costos: list = self.tabla_transiciones[key]

        candidatos: List[List[List[int]]] = []
        n_vars = len(costos)
        for idx in range(n_vars):
            presentes = [i for i in range(len(self.estado_final))]
            futuros = [i for i in range(n_vars) if i != idx]
            candidatos.append([presentes, futuros])

        es_par = len(self.caminos) % 2 == 0
        mitad = len(self.caminos) // 2 if es_par else (len(self.caminos) // 2) + 1
        for nivel in range(1, mitad):
            costo_candidato_nivel = 1e5
            presentes_nivel: List[int] = []
            futuros_nivel: List[int] = []
            for estado in self.caminos[nivel]:
                costo_candidato = 0
                presentes: List[int] = []
                futuros: List[int] = []
                actual = self.tabla_transiciones.get(
                    (tuple(self.caminos[0][0]), tuple(estado)), None
                )
                estado_complementario = (1 - np.array(estado)).tolist()
                complementario = self.tabla_transiciones.get(
                    (tuple(self.caminos[0][0]), tuple(estado_complementario)), None
                )
                for idx, i in enumerate(estado):
                    if i == self.caminos[0][0][idx]:
                        presentes.append(idx)
                for idx, _ in enumerate(self.idx_ncubos):
                    if actual[idx] <= complementario[idx]:
                        futuros.append(idx)
                        costo_candidato += actual[idx]
                    else:
                        costo_candidato += complementario[idx]
                if costo_candidato < costo_candidato_nivel:
                    costo_candidato_nivel = costo_candidato
                    presentes_nivel = presentes
                    futuros_nivel = futuros
            candidatos.append([presentes_nivel, futuros_nivel])
        return candidatos

    def _candidatos_k(self, orden_nodos: List[Nodo], k: int) -> List[List[Bloque]]:
        """
        Candidatas para `k>2`: candidato principal por saltos de costo, variantes de
        nivel cero y el candidato general de respaldo (siempre válido).
        """
        candidatos: List[List[Bloque]] = []

        principal = self._candidato_k_principal(k)
        if principal is not None:
            candidatos.append(principal)

        candidatos.extend(self._candidatos_k_nivel_cero(k))
        candidatos.append(self._candidato_k_general(orden_nodos, k))
        return candidatos

    def _candidato_k_principal(self, k: int) -> Optional[List[Bloque]]:
        """
        Candidato principal k-vía: ordena las variables futuras por su costo total y
        las corta en `k` grupos por los `k-1` saltos de costo mayores; las presentes
        acompañan a su futura homóloga. Devuelve `None` si hay menos futuras que `k`.
        """
        costos = self._costos_totales()
        n_vars = len(costos)
        if n_vars < k:
            return None
        indices_ordenados = sorted(range(n_vars), key=lambda i: costos[i])
        grupos_pos = self._cortar_por_saltos(indices_ordenados, costos, k)
        return self._bloques_desde_grupos_futuros(grupos_pos, costos)

    def _candidatos_k_nivel_cero(self, k: int) -> List[List[Bloque]]:
        """
        Variantes de nivel cero para `k>2`: por cada variable futura, ésta va sola en
        un bloque y el resto se reparte en `k-1` grupos por saltos de costo.
        """
        costos = self._costos_totales()
        n_vars = len(costos)
        candidatos: List[List[Bloque]] = []
        if n_vars < k:
            return candidatos
        for pos_removida in range(n_vars):
            restantes = [p for p in range(n_vars) if p != pos_removida]
            restantes_ordenados = sorted(restantes, key=lambda i: costos[i])
            grupos_resto = self._cortar_por_saltos(restantes_ordenados, costos, k - 1)
            grupos_pos = [[pos_removida]] + grupos_resto
            candidatos.append(self._bloques_desde_grupos_futuros(grupos_pos, costos))
        return candidatos

    def _candidato_k_general(
        self, orden_nodos: List[Nodo], k: int
    ) -> List[Bloque]:
        """
        Candidato general de respaldo: agrupa TODOS los nodos (presentes y futuros)
        por saltos de costo. Cada futura aporta su costo geométrico; cada presente
        hereda el costo de su futura homóloga (o el costo medio si no existe) y se
        ordena justo después de ella, de modo que los pares permanezcan juntos hasta
        que `k` obligue a separarlos. Garantiza una k-partición válida para `k <= m`.
        """
        costos = self._costos_totales()
        idxs = self.sia_subsistema.indices_ncubos
        fut_label_costo = {int(idxs[p]): costos[p] for p in range(len(costos))}
        costo_medio = float(np.mean(costos)) if len(costos) else 0.0

        # (costo, desempate, nodo): el desempate=1 ubica al presente tras su futura.
        nodos_con_costo: List[Tuple[float, int, Nodo]] = []
        for (tiempo, label) in orden_nodos:
            costo = fut_label_costo.get(int(label), costo_medio)
            desempate = 0 if tiempo == EFFECT else 1
            nodos_con_costo.append((costo, desempate, (int(tiempo), int(label))))
        nodos_ordenados = sorted(nodos_con_costo)

        valores = [c for (c, _, _) in nodos_ordenados]
        posiciones = list(range(len(nodos_ordenados)))
        grupos = self._cortar_por_saltos(posiciones, valores, k)
        return [[nodos_ordenados[p][2] for p in grupo] for grupo in grupos]

    def _costos_totales(self) -> list:
        """Vector de costos totales por variable futura (estado inicial -> final)."""
        key = tuple(self.caminos[0][0]), tuple(self.estado_final)
        return self.tabla_transiciones[key]

    def _cortar_por_saltos(
        self,
        posiciones_ordenadas: List[int],
        costos: list,
        num_grupos: int,
    ) -> List[List[int]]:
        """
        Corta una secuencia de posiciones (ya ordenada por costo ascendente) en
        `num_grupos` grupos contiguos por los `num_grupos-1` saltos de costo mayores.

        Args:
        ----
            posiciones_ordenadas (list[int]): Posiciones ordenadas por costo.
            costos (list): Vector de costos indexado por posición.
            num_grupos (int): Número de grupos deseado.

        Returns:
        -------
            list[list[int]]: Los grupos contiguos de posiciones (no vacíos si
            `len(posiciones_ordenadas) >= num_grupos`).
        """
        if num_grupos <= 1 or len(posiciones_ordenadas) <= 1:
            return [list(posiciones_ordenadas)]

        saltos: List[Tuple[float, int]] = []
        for idx in range(1, len(posiciones_ordenadas)):
            anterior = posiciones_ordenadas[idx - 1]
            actual = posiciones_ordenadas[idx]
            saltos.append((costos[actual] - costos[anterior], idx))

        # Mayores saltos primero (desempate determinista por índice ascendente).
        mayores = sorted(saltos, key=lambda s: (-s[0], s[1]))[: num_grupos - 1]
        cortes = sorted(idx for _, idx in mayores)

        grupos: List[List[int]] = []
        inicio = 0
        for corte in cortes:
            grupos.append(posiciones_ordenadas[inicio:corte])
            inicio = corte
        grupos.append(posiciones_ordenadas[inicio:])
        return grupos

    def _bloques_desde_grupos_futuros(
        self,
        grupos_pos: List[List[int]],
        costos: list,
    ) -> List[Bloque]:
        """
        Construye bloques a partir de grupos de POSICIONES de variables futuras.

        Cada grupo se vuelve un bloque con sus nodos futuros `(EFFECT, label)`. Luego
        cada presente `(ACTUAL, j)` se asigna al bloque de su futura homóloga si `j`
        es también una variable futura; si no, al bloque de menor costo promedio.

        Args:
        ----
            grupos_pos (list[list[int]]): Grupos de posiciones de variables futuras.
            costos (list): Vector de costos por posición futura.

        Returns:
        -------
            list[Bloque]: La partición como lista de bloques.
        """
        dims = self.sia_subsistema.dims_ncubos
        idxs = self.sia_subsistema.indices_ncubos

        bloques: List[Bloque] = [[] for _ in grupos_pos]
        fut_label_a_grupo: Dict[int, int] = {}
        costo_promedio: List[float] = []
        for g, grupo in enumerate(grupos_pos):
            suma = 0.0
            for pos in grupo:
                label = int(idxs[pos])
                bloques[g].append((EFFECT, label))
                fut_label_a_grupo[label] = g
                suma += costos[pos]
            costo_promedio.append(suma / len(grupo) if grupo else INFTY_POS)

        grupo_min_costo = int(np.argmin(costo_promedio))
        for j in dims:
            label = int(j)
            grupo = fut_label_a_grupo.get(label, grupo_min_costo)
            bloques[grupo].append((ACTUAL, label))
        return bloques
