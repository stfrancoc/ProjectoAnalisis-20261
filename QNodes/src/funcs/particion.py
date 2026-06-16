"""
Codificación y normalización de k-particiones.

Este módulo es la ÚNICA FUENTE DE VERDAD para la canonicalización de particiones
del subsistema. Es un módulo PURO: no depende de `System` ni de numpy, trabaja
sólo con estructuras de Python (tuplas, listas, ints) recibidas por parámetro, de
forma que sea testeable en aislamiento y reutilizable tanto por `System.kpartir`
como por las estrategias voraz (KQNodes) y exacta (Backtracking + B&B), que
operan en el espacio de vectores de etiquetas.

Convenciones (ver CLAUDE.md §3 y §5.2):
- Un nodo es una tupla `(tiempo, indice)` con `tiempo ∈ {ACTUAL, EFFECT}`.
  Los presentes son `(ACTUAL, j)` por cada dimensión presente `j`; los futuros
  `(EFFECT, i)` por cada n-cubo `i`.
- Una k-partición se codifica como un vector de etiquetas sobre los nodos (en el
  orden canónico): `[0, 0, 1, 2]` significa `{n0, n1} | {n2} | {n3}`.
- Como los bloques NO están etiquetados, dos vectores que sólo difieren en el
  nombre de los bloques (p. ej. `[0,0,1]` y `[1,1,0]`) representan la MISMA
  partición; por eso se normalizan a una forma canónica (restricted growth string)
  antes de usarlos como clave de memoización.
"""

from typing import Optional, Sequence

from src.constants.base import ACTUAL, EFFECT
from src.constants.error import (
    ERROR_PARTICION_COBERTURA,
    ERROR_PARTICION_K_BLOQUES,
    ERROR_PARTICION_K_RANGO,
    ERROR_PARTICION_NODO_DUPLICADO,
)

# Tipos cómodos para anotación (un nodo y una partición como lista de bloques).
Nodo = tuple[int, int]
Bloques = Sequence[Sequence[Nodo]]


def orden_canonico_nodos(
    dims_presentes: Sequence[int],
    indices_futuros: Sequence[int],
) -> list[Nodo]:
    """
    Devuelve el orden fijo y determinista de los nodos del subsistema.

    El orden es: primero los presentes `(ACTUAL, j)` con `j` en orden ascendente,
    y luego los futuros `(EFFECT, i)` con `i` en orden ascendente. Este orden es el
    eje sobre el que se construyen los vectores de etiquetas y, por tanto, las
    claves de caché; debe permanecer estable para no invalidar la memoización.

    Args:
    ----
        dims_presentes (Sequence[int]): Dimensiones presentes del subsistema
            (típicamente `System.dims_ncubos`). Puede ser lista o arreglo numpy.
        indices_futuros (Sequence[int]): Índices de los n-cubos futuros
            (típicamente `System.indices_ncubos`). Puede ser lista o arreglo numpy.

    Returns:
    -------
        list[tuple[int, int]]: La lista de nodos en orden canónico.
    """
    presentes = [(ACTUAL, int(j)) for j in sorted(dims_presentes)]
    futuros = [(EFFECT, int(i)) for i in sorted(indices_futuros)]
    return presentes + futuros


def etiquetas_desde_bloques(
    bloques: Bloques,
    orden_nodos: Sequence[Nodo],
) -> tuple[int, ...]:
    """
    Codifica una partición (lista de bloques) como vector de etiquetas.

    A cada nodo de `orden_nodos` se le asigna el identificador (índice) del bloque
    al que pertenece. El vector resultante respeta el orden de `orden_nodos`.

    Args:
    ----
        bloques (Bloques): Partición como lista de bloques; cada bloque es una
            colección de nodos `(tiempo, indice)`.
        orden_nodos (Sequence[tuple[int, int]]): Orden canónico de los nodos
            (ver `orden_canonico_nodos`).

    Returns:
    -------
        tuple[int, ...]: Vector de etiquetas, una por nodo en `orden_nodos`.

    Raises:
    ------
        KeyError: Si algún nodo de `orden_nodos` no aparece en `bloques` (úsese
            `validar_particion` previamente para un mensaje de error claro).
    """
    nodo_a_bloque: dict[Nodo, int] = {
        (int(tiempo), int(indice)): id_bloque
        for id_bloque, bloque in enumerate(bloques)
        for (tiempo, indice) in bloque
    }
    return tuple(nodo_a_bloque[(int(t), int(i))] for (t, i) in orden_nodos)


def normalizar_etiquetas(vector: Sequence[int]) -> tuple[int, ...]:
    """
    Lleva un vector de etiquetas a su forma canónica (restricted growth string).

    Se reetiqueta según el orden de PRIMERA APARICIÓN: la primera etiqueta vista
    pasa a ser `0`, la siguiente etiqueta nueva a `1`, etc. Así `[2, 2, 0, 1]` se
    normaliza a `[0, 0, 1, 2]`, y vectores equivalentes salvo renombrado de bloques
    (`[0,0,1]` y `[1,1,0]`) colapsan a la misma clave canónica. La operación es
    idempotente.

    Args:
    ----
        vector (Sequence[int]): Vector de etiquetas de bloque, una por nodo.

    Returns:
    -------
        tuple[int, ...]: El vector en forma canónica.
    """
    renombrado: dict[int, int] = {}
    canonica: list[int] = []
    for etiqueta in vector:
        if etiqueta not in renombrado:
            renombrado[etiqueta] = len(renombrado)
        canonica.append(renombrado[etiqueta])
    return tuple(canonica)


def clave_canonica(
    bloques: Bloques,
    orden_nodos: Sequence[Nodo],
) -> tuple[int, ...]:
    """
    Calcula la clave canónica de una partición.

    Es la composición de `etiquetas_desde_bloques` seguida de
    `normalizar_etiquetas`. NO incluye ningún prefijo de caché: el prefijo
    (p. ej. `KEY_KPART`) lo añade el llamador para separar espacios de claves.

    Args:
    ----
        bloques (Bloques): Partición como lista de bloques.
        orden_nodos (Sequence[tuple[int, int]]): Orden canónico de los nodos.

    Returns:
    -------
        tuple[int, ...]: Clave canónica de la partición.
    """
    return normalizar_etiquetas(etiquetas_desde_bloques(bloques, orden_nodos))


def bloques_desde_etiquetas(
    vector: Sequence[int],
    orden_nodos: Sequence[Nodo],
) -> list[list[Nodo]]:
    """
    Reconstruye la partición (lista de bloques) a partir de un vector de etiquetas.

    Es la inversa de `etiquetas_desde_bloques`. Los bloques se devuelven ordenados
    por identificador de etiqueta ascendente, y dentro de cada bloque los nodos
    conservan el orden de `orden_nodos`. La usan las estrategias que trabajan en el
    espacio de vectores de etiquetas (voraz y B&B) para traducir un vector a la
    estructura que consume `System.kpartir`.

    Args:
    ----
        vector (Sequence[int]): Vector de etiquetas, una por nodo en `orden_nodos`.
        orden_nodos (Sequence[tuple[int, int]]): Orden canónico de los nodos.

    Returns:
    -------
        list[list[tuple[int, int]]]: La partición como lista de bloques.
    """
    bloques: dict[int, list[Nodo]] = {}
    for nodo, etiqueta in zip(orden_nodos, vector):
        bloques.setdefault(etiqueta, []).append((int(nodo[0]), int(nodo[1])))
    return [bloques[etiqueta] for etiqueta in sorted(bloques)]


def validar_particion(
    bloques: Bloques,
    orden_nodos: Sequence[Nodo],
    k: Optional[int] = None,
) -> None:
    """
    Valida que `bloques` sea una partición real de los nodos de `orden_nodos`.

    Comprueba que cada nodo aparezca EXACTAMENTE una vez (sin duplicados ni nodos
    en dos bloques) y que TODOS los nodos estén cubiertos (sin faltantes ni nodos
    ajenos). Si se proporciona `k`, exige además que haya exactamente `k` bloques
    no vacíos y que `2 <= k <= m`, con `m = len(orden_nodos)` (no se admite `k=1`,
    la partición trivial; ver CLAUDE.md §3).

    Args:
    ----
        bloques (Bloques): Partición a validar.
        orden_nodos (Sequence[tuple[int, int]]): Orden canónico de los nodos, que
            define el conjunto de nodos esperados.
        k (Optional[int]): Número de bloques exigido. Si es `None`, sólo se valida
            la estructura de partición (sin restricción sobre el número de bloques).

    Raises:
    ------
        ValueError: Si hay nodos duplicados, faltantes o ajenos; si `k` está fuera
            de `[2, m]`; o si el número de bloques no vacíos no coincide con `k`.
    """
    m = len(orden_nodos)
    nodos_esperados = {(int(t), int(i)) for (t, i) in orden_nodos}

    vistos: list[Nodo] = []
    bloques_no_vacios = 0
    for bloque in bloques:
        if len(bloque) > 0:
            bloques_no_vacios += 1
        for (tiempo, indice) in bloque:
            vistos.append((int(tiempo), int(indice)))

    conjunto_vistos = set(vistos)
    if len(vistos) != len(conjunto_vistos):
        raise ValueError(ERROR_PARTICION_NODO_DUPLICADO)

    if conjunto_vistos != nodos_esperados:
        faltantes = nodos_esperados - conjunto_vistos
        ajenos = conjunto_vistos - nodos_esperados
        raise ValueError(ERROR_PARTICION_COBERTURA(faltantes, ajenos))

    if k is not None:
        if not (2 <= k <= m):
            raise ValueError(ERROR_PARTICION_K_RANGO(k, m))
        if bloques_no_vacios != k:
            raise ValueError(ERROR_PARTICION_K_BLOQUES(k, bloques_no_vacios))
