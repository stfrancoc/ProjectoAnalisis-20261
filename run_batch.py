#!/usr/bin/env python3
"""Runner para ejecutar batch comparativo KQNodes vs KGeoMIP.

Genera `results/batch_resultados.csv` y un reporte agregado.
Uso: python run_batch.py
"""
from pathlib import Path
import sys
import os
import time
import csv
import traceback
from multiprocessing import Pool
from typing import Any, Dict, Tuple
import inspect
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
# Apuntar directamente a la carpeta QNodes/src
QNODES_SRC = ROOT / "QNodes" / "src"

# Agregar QNodes/src al path
if str(QNODES_SRC) not in sys.path:
    sys.path.insert(0, str(QNODES_SRC))

# Opcional: Esto ayuda si los imports internos de los archivos también tienen problemas
if str(ROOT / "QNodes") not in sys.path:
    sys.path.insert(0, str(ROOT / "QNodes"))

def _import_strategy(module_path: str, class_name: str):
    # Import dinámico para evitar errores de path
    mod = __import__(module_path, fromlist=[class_name])
    return getattr(mod, class_name)


def ejecutar_estrategia(estrategia_cls: Any, tpm: np.ndarray, inicial: str, cond: str, alc: str, mec: str, k: int) -> Tuple[float, float]:
    """Instancia y ejecuta la estrategia, adaptando firmas diferentes.

    Retorna (delta, tiempo_segundos).
    Lanza excepción si falla.
    """
    start = time.time()

    # Instanciación: intentar con tpm en __init__, si falla, sin argumentos
    try:
        sig_init = inspect.signature(estrategia_cls.__init__)
        params = sig_init.parameters
        # excluir 'self'
        accepts_tpm = any('tpm' in p or 'gestor' in p or 'tpm' == p for p in params.keys())
    except Exception:
        accepts_tpm = True

    if accepts_tpm:
        instancia = estrategia_cls(tpm)
    else:
        instancia = estrategia_cls()

    # Preparar llamar aplicar_estrategia: detectar parámetros
    aplicar = getattr(instancia, 'aplicar_estrategia')
    sig = inspect.signature(aplicar)
    kwargs = {}
    # Primer intento: forma (estado_inicial, condicion, alcance, mecanismo, k)
    try:
        if 'tpm' in sig.parameters:
            res = aplicar(inicial, cond, alc, mec, tpm=tpm, k=k)
        else:
            # ordenar por nombres si son posicionales
            res = aplicar(inicial, cond, alc, mec, k)
    except TypeError:
        # último recurso: probar sin k (algunas implementaciones usan otro flujo)
        res = aplicar(inicial, cond, alc, mec)

    end = time.time()
    tiempo = end - start

    # Intentar extraer pérdida (perdida, delta) desde la Solution
    delta = None
    if hasattr(res, 'perdida'):
        delta = float(res.perdida)
    elif isinstance(res, tuple) and len(res) >= 1:
        try:
            delta = float(res[0])
        except Exception:
            delta = None

    return delta, tiempo


def _load_tpm(path: Path) -> np.ndarray:
    # Usa delimitador ',' porque los CSV reales están en formato coma.
    return np.genfromtxt(str(path), delimiter=',')


def worker(job: Dict[str, Any]) -> Dict[str, Any]:
    sample = job['sample']
    k = job['k']
    estrategia_name = job['estrategia_name']
    estrategia_cls = job['estrategia_cls']
    inicial = job['inicial']
    cond = job['cond']
    alc = job['alc']
    mec = job['mec']

    out = {
        'sample': sample.name,
        'k': k,
        'estrategia': estrategia_name,
        'delta': None,
        'tiempo': None,
        'error': None,
    }

    try:
        tpm = _load_tpm(sample)
        delta, tiempo = ejecutar_estrategia(estrategia_cls, tpm, inicial, cond, alc, mec, k)
        out['delta'] = delta
        out['tiempo'] = tiempo
    except Exception as e:
        out['error'] = f"{type(e).__name__}: {e}"
        out['traceback'] = traceback.format_exc()

    return out


def discover_samples() -> list[Path]:
    samples = []
    qnodes_samples = QNODES_SRC / '.samples'
    if qnodes_samples.exists():
        samples.extend(sorted(qnodes_samples.glob('*.csv')))
    # fallback: GeoMIP samples
    geomip_samples = ROOT / 'GeoMIP' / 'data' / 'samples'
    if geomip_samples.exists():
        samples.extend(sorted(geomip_samples.glob('*.csv')))
    return samples


def main():
    samples = discover_samples()
    if not samples:
        print('No se encontraron muestras en QNodes/.samples ni GeoMIP/data/samples')
        return

    # Importar clases de estrategias desde QNodes/src
    KQNodes = _import_strategy('src.strategies.k_qnodes', 'KQNodes')
    KGeoMIP = _import_strategy('src.strategies.k_geomip', 'KGeoMIP')

    estrategias = [
        ('KQNodes', KQNodes),
        ('KGeoMIP', KGeoMIP),
    ]

    ks = [2, 3, 4, 5]

    jobs = []
    for sample in samples:
        tpm = _load_tpm(sample)
        n = int(tpm.shape[1])
        inicial = '1' * n
        cond = '1' * n
        alc = '1' * n
        mec = '1' * n

        for k in ks:
            for name, cls in estrategias:
                jobs.append({
                    'sample': sample,
                    'k': k,
                    'estrategia_name': name,
                    'estrategia_cls': cls,
                    'inicial': inicial,
                    'cond': cond,
                    'alc': alc,
                    'mec': mec,
                })

    # multiprocessing pool (max 4 procesos)
    results_dir = ROOT / 'results'
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / 'batch_resultados.csv'

    with Pool(processes=4) as p:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['sample', 'k', 'estrategia', 'delta', 'tiempo', 'error'])
            writer.writeheader()
            for res in p.imap_unordered(worker, jobs):
                writer.writerow({k: res.get(k) for k in ['sample', 'k', 'estrategia', 'delta', 'tiempo', 'error']})

    # Consolidación con pandas
    df = pd.read_csv(csv_path)
    report = []
    for k in ks:
        row = {'k': k}
        sub = df[df['k'] == k]
        for name, _ in estrategias:
            s = sub[sub['estrategia'] == name]
            row[f'tiempo_prom_{name}'] = s['tiempo'].dropna().astype(float).mean()
            row[f'delta_prom_{name}'] = s['delta'].dropna().astype(float).mean()
        report.append(row)

    report_df = pd.DataFrame(report)
    report_path = results_dir / 'batch_reporte_por_k.csv'
    report_df.to_csv(report_path, index=False)

    # Comparativa simple: contar en cuántos ks KGeoMIP es más rápido que KQNodes
    gana_geo = 0
    gana_q = 0
    for _, r in report_df.iterrows():
        t_geo = r.get('tiempo_prom_KGeoMIP', float('nan'))
        t_q = r.get('tiempo_prom_KQNodes', float('nan'))
        if pd.notna(t_geo) and pd.notna(t_q):
            if t_geo < t_q:
                gana_geo += 1
            elif t_q < t_geo:
                gana_q += 1

    summary_path = results_dir / 'batch_resumen.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'Report saved: {report_path}\n')
        f.write(f'Raw results: {csv_path}\n')
        f.write(f'KGeoMIP faster on {gana_geo} k-values; KQNodes faster on {gana_q} k-values\n')

    print('Batch finished. Outputs:')
    print(f' - Raw CSV: {csv_path}')
    print(f' - Report por k: {report_path}')
    print(f' - Resumen: {summary_path}')


def smoke_test():
    """Prueba de humo local: invoca ambas estrategias en la primera muestra."""
    samples = discover_samples()
    if not samples:
        print('No hay muestras para el test de humo.')
        return
    sample = samples[0]
    print(f'Smoke test usando muestra {sample.name}')
    KQNodes = _import_strategy('src.strategies.k_qnodes', 'KQNodes')
    KGeoMIP = _import_strategy('src.strategies.k_geomip', 'KGeoMIP')
    tpm = _load_tpm(sample)
    n = int(tpm.shape[1])
    inicial = '1' * n
    cond = '1' * n
    alc = '1' * n
    mec = '1' * n

    try:
        print(' - Ejecutando KQNodes k=2 ...')
        d1, t1 = ejecutar_estrategia(KQNodes, tpm, inicial, cond, alc, mec, 2)
        print(f'   OK delta={d1} tiempo={t1:.3f}s')
    except Exception as e:
        print('   KQNodes ERROR', e)

    try:
        print(' - Ejecutando KGeoMIP k=2 ...')
        d2, t2 = ejecutar_estrategia(KGeoMIP, tpm, inicial, cond, alc, mec, 2)
        print(f'   OK delta={d2} tiempo={t2:.3f}s')
    except Exception as e:
        print('   KGeoMIP ERROR', e)


if __name__ == '__main__':
    # Guardar current dir y colocarse en ROOT para imports relativos funcionales
    os.chdir(str(ROOT))
    print('RUN BATCH: working dir set to', ROOT)
    print('Ejecutando smoke-test antes del batch...')
    smoke_test()
    proceed = input('Continuar con batch completo? (y/N): ').strip().lower() == 'y'
    if proceed:
        main()
