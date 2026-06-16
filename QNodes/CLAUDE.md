# CLAUDE.md — Proyecto K-QGMIP / KQNodes

> Colocación: pon este archivo en `QNodes/CLAUDE.md` y abre la carpeta `QNodes/`
> como workspace en VS Code (o en la raíz del repo si trabajas desde ahí).
> Todas las rutas de abajo son relativas a la raíz del proyecto `QNodes/`.

## 0. Idioma y tono
- Responde y comenta **en español**. Docstrings en español, detallados, como el resto del código.
- No reescribas código existente que ya funciona. Extiende, no reimplementes.
- Antes de tocar un archivo, léelo. No inventes firmas: usa las que ya existen.

## 1. Qué es este proyecto (contexto mínimo)
Análisis y Diseño de Algoritmos, U. de Caldas. El problema es el **MIP** (Minimum
Information Partition) de la Teoría de la Información Integrada (IIT): dado un
sistema de variables binarias descrito por una TPM, se busca la **partición** que
minimice la pérdida de información entre la dinámica del subsistema original y la
dinámica reconstruida al recombinar las partes vía producto tensorial. La pérdida
se mide con **EMD (Earth Mover's Distance)** con métrica de Hamming.

El proyecto original resuelve **biparticiones** (k=2). Esta entrega extiende a
**k-particiones** (k general, fijado por el usuario). La estrategia principal se
llama **KQNodes** (convención del docente: el nombre debe ser consistente en
repositorio, carpeta, clase principal y documentación).

## 2. Decisión de diseño (qué construimos y en qué orden)
Triada, con plazo ajustado (~15 días). Implementar en este orden estricto:

1. **KQNodes voraz/aglomerativo** — estrategia PRINCIPAL. Empieza con cada nodo en
   su propio grupo y fusiona iterativamente los dos grupos cuya unión genere menor
   pérdida, hasta quedar con exactamente `k` grupos.
2. **Backtracking + Branch & Bound** — VALIDADOR exacto para `n` pequeño (≤5).
   Sirve como ground-truth propio para medir la calidad de KQNodes.
3. **KGeoMIP geométrico** — comparación avanzada, SOLO si sobra tiempo. Porta la
   tabla de costos desde `GeoMIP/src/Method2_Dynamic_Programming_Reformulation`.

No empezar la fase N+1 hasta que la fase N esté funcional Y probada.

## 3. Formulación del problema (para que el código sea coherente)
- Nodos particionables del subsistema: presentes `(ACTUAL=0, j)` para cada `j` en
  `dims_ncubos`, y futuros `(EFFECT=1, i)` para cada `i` en `indices_ncubos`.
- Una k-partición `P` asigna cada nodo a uno de `k` bloques `S_1..S_k`.
- Función objetivo a minimizar:
  `δ(V, P) = emd_efecto(dist_subsistema, kpartir(P).distribucion_marginal())`
  donde `dist_subsistema = self.sia_dists_marginales`.
- Restricción: `k` lo fija el usuario, con `2 ≤ k ≤ m` (m = nº de nodos
  particionables). NO permitir `k=1` (partición trivial, δ=0).

## 4. Arquitectura REAL existente (verificada — úsala tal cual)
Patrón Strategy sobre una clase base abstracta. Rutas relativas a `QNodes/`:

- `src/main.py` — punto de entrada `iniciar()`. Usa `Manager(estado_inicial)`,
  `gestor.cargar_red()` devuelve la `tpm`; se instancia la estrategia con la `tpm`
  y se llama `aplicar_estrategia(estado_inicial, condicion, alcance, mecanismo)`.
- `src/models/base/sia.py` — clase base **`SIA(ABC)`**:
  - `__init__(self, tpm: np.ndarray)`
  - `@abstractmethod aplicar_estrategia(...)` — cada estrategia la implementa.
  - `sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)`:
    construye el subsistema y deja listos `self.sia_subsistema: System` y
    `self.sia_dists_marginales: np.ndarray`, además de `self.sia_tiempo_inicio`.
- `src/models/core/system.py` — clase **`System`**:
  - `condicionar(indices)` → System candidato (background conditions).
  - `substraer(alcance_dims, mecanismo_dims)` → subsistema.
  - **`bipartir(alcance, mecanismo)`** → System de la bipartición (tiene caché
    `self.memo`). Lógica: por cada n-cubo futuro, si su índice está en `alcance`
    marginaliza `setdiff1d(cubo.dims, mecanismo)` (conserva el mecanismo del grupo);
    si no, marginaliza `mecanismo` (conserva las dims del otro lado).
  - `distribucion_marginal()` → vector que se compara con EMD.
  - Propiedades: `indices_ncubos`, `dims_ncubos`.
- `src/models/core/ncube.py` — clase **`NCube`**, método clave `marginalizar(dims)`
  (es agnóstico a k; hace todo el trabajo pesado de marginalización).
- `src/funcs/iit.py` — **`emd_efecto(u, v) -> float`** y `ABECEDARY`.
- `src/funcs/format.py` — `fmt_biparticion_q`, `fmt_biparticion_fuerza_bruta`.
- `src/models/core/solution.py` — clase **`Solution`**. Constructor:
  `Solution(estrategia, perdida, distribucion_subsistema, distribucion_particion, particion, tiempo_total=..., quiere_hablar=True)`.
  ⚠️ Por defecto **habla por voz (pyttsx3)**. En experimentos por lotes pasar
  `quiere_hablar=False`.
- `src/strategies/` — estrategias existentes: `q_nodes.py` (clase `QNodes`),
  `force.py` (clase `BruteForce`), `phi.py` (referencia PyPhi).
- `src/funcs/force.py` — generador `biparticiones(...)` (referencia para el
  generador de k-particiones).
- `src/constants/base.py` — constantes (`ACTUAL=0`, `EFFECT=1`, `STR_ZERO`, etc.).
- `src/constants/models.py` — TAGS y LABELS de estrategias (agregar los nuevos aquí).

## 5. Especificación de lo nuevo a implementar

### 5.1 `System.kpartir(bloques)` — generalización de `bipartir`
Nuevo método en `src/models/core/system.py`. Generaliza `bipartir` a k bloques.
- Entrada: la partición como asignación de nodos a bloques (ver 5.2).
- Por cada n-cubo futuro `i` asignado al bloque `b`: conserva solo las dims
  presentes `j` tales que `(ACTUAL, j)` está en `b`; marginaliza el resto, es decir
  `cubo.marginalizar(present_dims_fuera_de_b)`.
- Con k=2 debe dar EXACTAMENTE el mismo resultado que `bipartir` (test de
  consistencia obligatorio).
- Reutilizar/extender el caché `self.memo` con clave canónica de la partición.

### 5.2 Codificación y normalización de particiones
Nuevo módulo `src/funcs/particion.py`.
- Representar una k-partición como un vector de etiquetas sobre los nodos, p. ej.
  `[0,0,1,2]` = `{n0,n1} | {n2} | {n3}`.
- Normalización canónica (relabel por primer aparición) para que `[0,0,1,2]` y
  `[1,1,0,2]` se traten como la MISMA partición → evita duplicados y permite
  memoizar correctamente (los bloques no están etiquetados).

### 5.3 `KQNodes(SIA)` — estrategia principal
Nuevo archivo `src/strategies/k_qnodes.py`, clase `KQNodes(SIA)`.
- Firma: `aplicar_estrategia(self, estado_inicial, condicion, alcance, mecanismo, k)`.
- Llama `self.sia_preparar_subsistema(...)`, arma los nodos `(ACTUAL,j)`/`(EFFECT,i)`.
- Bucle voraz aglomerativo: inicia con singletons; mientras nº grupos > k, evalúa
  fusiones candidatas de pares de grupos con `kpartir(...).distribucion_marginal()`
  + `emd_efecto(...)`; fusiona la de menor pérdida; memoiza.
- Memorias: `memoria_individual`, `memoria_particiones`, `memoria_fusiones`.
- Retorna `Solution(estrategia=KQNODES_LABEL, perdida=δ, distribucion_subsistema=self.sia_dists_marginales, distribucion_particion=..., particion=<formato k-vía>, tiempo_total=time.time()-self.sia_tiempo_inicio)`.

### 5.4 Validador B&B (fase 2) y KGeoMIP (fase 3) — más adelante
- `src/strategies/exacto_bb.py`: backtracking con etiquetado canónico (cada nodo
  va a un bloque existente o abre uno nuevo hasta `k`), poda por cota inferior,
  cota inicial = resultado de KQNodes. Evaluar particiones completas con EMD. Solo
  `n` pequeño.
- `src/strategies/k_geomip.py`: portar la tabla de costos de
  `GeoMIP/src/Method2_Dynamic_Programming_Reformulation/src/controllers/strategies/geometric.py`,
  reemplazar el análisis de complementariedad binaria por clustering k-vía sobre la
  tabla, reusar `kpartir`. (Opcional: acelerar la construcción de la tabla en GPU.)

## 6. Cómo ejecutar
- Python 3.11+, gestor `uv`. Datasets TPM en `src/.samples/` (`N*.csv`) y también
  en `GeoMIP/data/samples/` del repo.
- Ejecutar vía `src/main.py` (`iniciar()`), instanciando la estrategia nueva con la
  `tpm` cargada por el `Manager` y pasando `k`.

## 7. Estilo y convenciones (observadas en el repo)
- snake_case, type hints en todo, numpy en todo.
- Constantes centralizadas en `src/constants/`. No hardcodear strings de tags/labels.
- Logger: `SafeLogger`. Nodos como tuplas `(tiempo, indice)` con `ACTUAL=0`/`EFFECT=1`.
- Docstrings en español, estilo del código existente.

## 8. Requisito del manual (no olvidar)
El manual técnico exige documentar de forma transparente el uso de IA generativa
(qué herramientas, en qué fases, ejemplos de prompts, qué se generó/influyó, y una
reflexión crítica). Mantener un registro de los prompts y decisiones durante el
desarrollo. También exige: definición formal de k-particiones, función objetivo y
restricciones, análisis de complejidad en función de `n` y `k`, y comparación con
fuerza bruta y con las estrategias originales de bipartición.
