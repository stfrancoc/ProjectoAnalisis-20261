#!/usr/bin/env python3
"""Runner para ejecutar batch comparativo KQNodes vs KGeoMIP.

Genera `results/batch_resultados_final.csv`.
"""
from pathlib import Path
import sys
import os
import time
import csv
import traceback
from multiprocessing import Pool
from typing import Any, Dict
import numpy as np
import logging

_WORKER_TPM_CACHE: Dict[str, np.ndarray] = {}

# Desactivar todos los logs globalmente para evitar la recursión de colorama
logging.disable(logging.CRITICAL)

# --- SETUP PATHS ---
ROOT = Path(__file__).resolve().parent
QNODES_SRC = ROOT / "QNodes" / "src"

# Asegurar que el paquete `src` dentro de QNodes sea importable
if str(QNODES_SRC) not in sys.path:
    sys.path.insert(0, str(QNODES_SRC))
if str(ROOT / "QNodes") not in sys.path:
    sys.path.insert(0, str(ROOT / "QNodes"))

import colorama
import colorama.initialise
# Monkeypatch global para evitar que colorama envuelva sys.stderr repetidamente en cada tarea
colorama.init = lambda *args, **kwargs: None
colorama.initialise.init = lambda *args, **kwargs: None

from src.strategies.k_qnodes import KQNodes
from src.strategies.k_geomip import KGeoMIP

# --- DEFINICIÓN DE PRUEBAS ---
PRUEBAS_POR_RED = {
    "N10A.csv": [
        ("ABCDEFGHIJ","ABCDEFGHIJ"),("ABCDEFGHIJ","ABCDEFGHI"),("ABCDEFGHIJ","BCDEFGHIJ"),("ABCDEFGHIJ","BCDEFGHI"),("ABCDEFGHIJ","ABDEGHJ"),("ABCDEFGHIJ","ACEGI"),("ABCDEFGHIJ","BDFHJ"),("ABCDEFGHI","ABCDEFGHIJ"),("ABCDEFGHI","ABCDEFGHI"),("ABCDEFGHI","BCDEFGHIJ"),("ABCDEFGHI","BCDEFGHI"),("ABCDEFGHI","ABDEGHJ"),("ABCDEFGHI","ACEGI"),("ABCDEFGHI","BDFHJ"),("BCDEFGHIJ","ABCDEFGHIJ"),("BCDEFGHIJ","ABCDEFGHI"),("BCDEFGHIJ","BCDEFGHIJ"),("BCDEFGHIJ","BCDEFGHI"),("BCDEFGHIJ","ABDEGHJ"),("BCDEFGHIJ","ACEGI"),("BCDEFGHIJ","BDFHJ"),("BCDEFGHI","ABCDEFGHIJ"),("BCDEFGHI","ABCDEFGHI"),("BCDEFGHI","BCDEFGHIJ"),("BCDEFGHI","BCDEFGHI"),("BCDEFGHI","ABDEGHJ"),("BCDEFGHI","ACEGI"),("BCDEFGHI","BDFHJ"),("ABDEGHJ","ABCDEFGHIJ"),("ABDEGHJ","ABCDEFGHI"),("ABDEGHJ","BCDEFGHIJ"),("ABDEGHJ","BCDEFGHI"),("ABDEGHJ","ABDEGHJ"),("ABDEGHJ","ACEGI"),("ABDEGHJ","BDFHJ"),("ACEGI","ABCDEFGHIJ"),("ACEGI","ABCDEFGHI"),("ACEGI","BCDEFGHIJ"),("ACEGI","BCDEFGHI"),("ACEGI","ABDEGHJ"),("ACEGI","ACEGI"),("ACEGI","BDFHJ"),("BDFHJ","ABCDEFGHIJ"),("BDFHJ","ABCDEFGHI"),("BDFHJ","BCDEFGHIJ"),("BDFHJ","BCDEFGHI"),("BDFHJ","ABDEGHJ"),("BDFHJ","ACEGI"),("BDFHJ","BDFHJ")
    ],
    "N10A.csv": [
        ("ABCDEFGHIJ","ABCDEFGHIJ"),("ABCDEFGHIJ","ABCDEFGHI"),("ABCDEFGHIJ","BCDEFGHIJ"),("ABCDEFGHIJ","BCDEFGHI"),("ABCDEFGHIJ","ABDEGHJ"),("ABCDEFGHIJ","ACEGI"),("ABCDEFGHIJ","BDFHJ"),("ABCDEFGHI","ABCDEFGHIJ"),("ABCDEFGHI","ABCDEFGHI"),("ABCDEFGHI","BCDEFGHIJ"),("ABCDEFGHI","BCDEFGHI"),("ABCDEFGHI","ABDEGHJ"),("ABCDEFGHI","ACEGI"),("ABCDEFGHI","BDFHJ"),("BCDEFGHIJ","ABCDEFGHIJ"),("BCDEFGHIJ","ABCDEFGHI"),("BCDEFGHIJ","BCDEFGHIJ"),("BCDEFGHIJ","BCDEFGHI"),("BCDEFGHIJ","ABDEGHJ"),("BCDEFGHIJ","ACEGI"),("BCDEFGHIJ","BDFHJ"),("BCDEFGHI","ABCDEFGHIJ"),("BCDEFGHI","ABCDEFGHI"),("BCDEFGHI","BCDEFGHIJ"),("BCDEFGHI","BCDEFGHI"),("BCDEFGHI","ABDEGHJ"),("BCDEFGHI","ACEGI"),("BCDEFGHI","BDFHJ"),("ABDEGHJ","ABCDEFGHIJ"),("ABDEGHJ","ABCDEFGHI"),("ABDEGHJ","BCDEFGHIJ"),("ABDEGHJ","BCDEFGHI"),("ABDEGHJ","ABDEGHJ"),("ABDEGHJ","ACEGI"),("ABDEGHJ","BDFHJ"),("ACEGI","ABCDEFGHIJ"),("ACEGI","ABCDEFGHI"),("ACEGI","BCDEFGHIJ"),("ACEGI","BCDEFGHI"),("ACEGI","ABDEGHJ"),("ACEGI","ACEGI"),("ACEGI","BDFHJ"),("BDFHJ","ABCDEFGHIJ"),("BDFHJ","ABCDEFGHI"),("BDFHJ","BCDEFGHIJ"),("BDFHJ","BCDEFGHI"),("BDFHJ","ABDEGHJ"),("BDFHJ","ACEGI"),("BDFHJ","BDFHJ")
    ],
    "N15B.csv": [
        ("ABCDEFGHIJKLMNO","ABCDEFGHIJKLMNO"),("ABCDEFGHIJKLMNO","ABCDEFGHIJKLMN"),("ABCDEFGHIJKLMNO","BCDEFGHIJKLMNO"),("ABCDEFGHIJKLMNO","BCDEFGHIJKLMN"),("ABCDEFGHIJKLMNO","ABDEGHJKMN"),("ABCDEFGHIJKLMNO","ACEGIKMO"),("ABCDEFGHIJKLMNO","BDFHJLN"),("ABCDEFGHIJKLMN","ABCDEFGHIJKLMNO"),("ABCDEFGHIJKLMN","ABCDEFGHIJKLMN"),("ABCDEFGHIJKLMN","BCDEFGHIJKLMNO"),("ABCDEFGHIJKLMN","BCDEFGHIJKLMN"),("ABCDEFGHIJKLMN","ABDEGHJKMN"),("ABCDEFGHIJKLMN","ACEGIKMO"),("ABCDEFGHIJKLMN","BDFHJLN"),("BCDEFGHIJKLMNO","ABCDEFGHIJKLMNO"),("BCDEFGHIJKLMNO","ABCDEFGHIJKLMN"),("BCDEFGHIJKLMNO","BCDEFGHIJKLMNO"),("BCDEFGHIJKLMNO","BCDEFGHIJKLMN"),("BCDEFGHIJKLMNO","ABDEGHJKMN"),("BCDEFGHIJKLMNO","ACEGIKMO"),("BCDEFGHIJKLMNO","BDFHJLN"),("BCDEFGHIJKLMN","ABCDEFGHIJKLMNO"),("BCDEFGHIJKLMN","ABCDEFGHIJKLMN"),("BCDEFGHIJKLMN","BCDEFGHIJKLMNO"),("BCDEFGHIJKLMN","BCDEFGHIJKLMN"),("BCDEFGHIJKLMN","ABDEGHJKMN"),("BCDEFGHIJKLMN","ACEGIKMO"),("BCDEFGHIJKLMN","BDFHJLN"),("ABDEGHJKMN","ABCDEFGHIJKLMNO"),("ABDEGHJKMN","ABCDEFGHIJKLMN"),("ABDEGHJKMN","BCDEFGHIJKLMNO"),("ABDEGHJKMN","BCDEFGHIJKLMN"),("ABDEGHJKMN","ABDEGHJKMN"),("ABDEGHJKMN","ACEGIKMO"),("ABDEGHJKMN","BDFHJLN"),("ACEGIKMO","ABCDEFGHIJKLMNO"),("ACEGIKMO","ABCDEFGHIJKLMN"),("ACEGIKMO","BCDEFGHIJKLMNO"),("ACEGIKMO","BCDEFGHIJKLMN"),("ACEGIKMO","ABDEGHJKMN"),("ACEGIKMO","ACEGIKMO"),("ACEGIKMO","BDFHJLN"),("BDFHJLN","ABCDEFGHIJKLMNO"),("BDFHJLN","ABCDEFGHIJKLMN"),("BDFHJLN","BCDEFGHIJKLMNO"),("BDFHJLN","BCDEFGHIJKLMN"),("BDFHJLN","ABDEGHJKMN"),("BDFHJLN","ACEGIKMO"),("BDFHJLN","BDFHJLN"),("BCDEFGJKLMNO","BCDEFGHIJKLMNO")
    ],
    "N20A.csv": [
        ("ABCDEFGHIJKLMNOPQRST","ABCDEFGHIJKLMNOPQRST"),("ABCDEFGHIJKLMNOPQRST","ABCDEFGHIJKLMNOPQRS"),("ABCDEFGHIJKLMNOPQRST","BCDEFGHIJKLMNOPQRST"),("ABCDEFGHIJKLMNOPQRST","BCDEFGHIJKLMNOPQRS"),("ABCDEFGHIJKLMNOPQRST","ABDEGHJKMNPQST"),("ABCDEFGHIJKLMNOPQRST","ACEGIKMOQS"),("ABCDEFGHIJKLMNOPQRST","BDFHJLNPRT"),("ABCDEFGHIJKLMNOPQRS","ABCDEFGHIJKLMNOPQRST"),("ABCDEFGHIJKLMNOPQRS","ABCDEFGHIJKLMNOPQRS"),("ABCDEFGHIJKLMNOPQRS","BCDEFGHIJKLMNOPQRST"),("ABCDEFGHIJKLMNOPQRS","BCDEFGHIJKLMNOPQRS"),("ABCDEFGHIJKLMNOPQRS","ABDEGHJKMNPQST"),("ABCDEFGHIJKLMNOPQRS","ACEGIKMOQS"),("ABCDEFGHIJKLMNOPQRS","BDFHJLNPRT"),("BCDEFGHIJKLMNOPQRST","ABCDEFGHIJKLMNOPQRST"),("BCDEFGHIJKLMNOPQRST","ABCDEFGHIJKLMNOPQRS"),("BCDEFGHIJKLMNOPQRST","BCDEFGHIJKLMNOPQRST"),("BCDEFGHIJKLMNOPQRST","BCDEFGHIJKLMNOPQRS"),("BCDEFGHIJKLMNOPQRST","ABDEGHJKMNPQST"),("BCDEFGHIJKLMNOPQRST","ACEGIKMOQS"),("BCDEFGHIJKLMNOPQRST","BDFHJLNPRT"),("BCDEFGHIJKLMNOPQRS","ABCDEFGHIJKLMNOPQRST"),("BCDEFGHIJKLMNOPQRS","ABCDEFGHIJKLMNOPQRS"),("BCDEFGHIJKLMNOPQRS","BCDEFGHIJKLMNOPQRST"),("BCDEFGHIJKLMNOPQRS","BCDEFGHIJKLMNOPQRS"),("BCDEFGHIJKLMNOPQRS","ABDEGHJKMNPQST"),("BCDEFGHIJKLMNOPQRS","ACEGIKMOQS"),("BCDEFGHIJKLMNOPQRS","BDFHJLNPRT"),("ABDEGHJKMNPQST","ABCDEFGHIJKLMNOPQRST"),("ABDEGHJKMNPQST","ABCDEFGHIJKLMNOPQRS"),("ABDEGHJKMNPQST","BCDEFGHIJKLMNOPQRST"),("ABDEGHJKMNPQST","BCDEFGHIJKLMNOPQRS"),("ABDEGHJKMNPQST","ABDEGHJKMNPQST"),("ABDEGHJKMNPQST","ACEGIKMOQS"),("ABDEGHJKMNPQST","BDFHJLNPRT"),("ACEGIKMOQS","ABCDEFGHIJKLMNOPQRST"),("ACEGIKMOQS","ABCDEFGHIJKLMNOPQRS"),("ACEGIKMOQS","BCDEFGHIJKLMNOPQRST"),("ACEGIKMOQS","BCDEFGHIJKLMNOPQRS"),("ACEGIKMOQS","ABDEGHJKMNPQST"),("ACEGIKMOQS","ACEGIKMOQS"),("ACEGIKMOQS","BDFHJLNPRT"),("BDFHJLNPRT","ABCDEFGHIJKLMNOPQRST"),("BDFHJLNPRT","ABCDEFGHIJKLMNOPQRS"),("BDFHJLNPRT","BCDEFGHIJKLMNOPQRST"),("BDFHJLNPRT","BCDEFGHIJKLMNOPQRS"),("BDFHJLNPRT","ABDEGHJKMNPQST"),("BDFHJLNPRT","ACEGIKMOQS"),("BDFHJLNPRT","BDFHJLNPRT"),("BCDEFGJKLMNO","BCDEFGHIJKLMNO")
    ],
    "N22A.csv": [
        ("ABCDEFGHIJKLMNOPQRSTUV","ABCDEFGHIJKLMNOPQRSTUV"),("ABCDEFGHIJKLMNOPQRSTUV","ABCDEFGHIJKLMNOPQRSTU"),("ABCDEFGHIJKLMNOPQRSTUV","BCDEFGHIJKLMNOPQRSTUV"),("ABCDEFGHIJKLMNOPQRSTUV","BCDEFGHIJKLMNOPQRSTU"),("ABCDEFGHIJKLMNOPQRSTUV","ABDEGHJKMNPQSTV"),("ABCDEFGHIJKLMNOPQRSTUV","ACEGIKMOQSU"),("ABCDEFGHIJKLMNOPQRSTUV","BDFHJLNPRTV"),("ABCDEFGHIJKLMNOPQRSTU","ABCDEFGHIJKLMNOPQRSTUV"),("ABCDEFGHIJKLMNOPQRSTU","ABCDEFGHIJKLMNOPQRSTU"),("ABCDEFGHIJKLMNOPQRSTU","BCDEFGHIJKLMNOPQRSTUV"),("ABCDEFGHIJKLMNOPQRSTU","BCDEFGHIJKLMNOPQRSTU"),("ABCDEFGHIJKLMNOPQRSTU","ABDEGHJKMNPQSTV"),("ABCDEFGHIJKLMNOPQRSTU","ACEGIKMOQSU"),("ABCDEFGHIJKLMNOPQRSTU","BDFHJLNPRTV"),("BCDEFGHIJKLMNOPQRSTUV","ABCDEFGHIJKLMNOPQRSTUV"),("BCDEFGHIJKLMNOPQRSTUV","ABCDEFGHIJKLMNOPQRSTU"),("BCDEFGHIJKLMNOPQRSTUV","BCDEFGHIJKLMNOPQRSTUV"),("BCDEFGHIJKLMNOPQRSTUV","BCDEFGHIJKLMNOPQRSTU"),("BCDEFGHIJKLMNOPQRSTUV","ABDEGHJKMNPQSTV"),("BCDEFGHIJKLMNOPQRSTUV","ACEGIKMOQSU"),("BCDEFGHIJKLMNOPQRSTUV","BDFHJLNPRTV"),("BCDEFGHIJKLMNOPQRSTU","ABCDEFGHIJKLMNOPQRSTUV"),("BCDEFGHIJKLMNOPQRSTU","ABCDEFGHIJKLMNOPQRSTU"),("BCDEFGHIJKLMNOPQRSTU","BCDEFGHIJKLMNOPQRSTUV"),("BCDEFGHIJKLMNOPQRSTU","BCDEFGHIJKLMNOPQRSTU"),("BCDEFGHIJKLMNOPQRSTU","ABDEGHJKMNPQSTV"),("BCDEFGHIJKLMNOPQRSTU","ACEGIKMOQSU"),("BCDEFGHIJKLMNOPQRSTU","BDFHJLNPRTV"),("ABDEGHJKMNPQSTV","ABCDEFGHIJKLMNOPQRSTUV"),("ABDEGHJKMNPQSTV","ABCDEFGHIJKLMNOPQRSTU"),("ABDEGHJKMNPQSTV","BCDEFGHIJKLMNOPQRSTUV"),("ABDEGHJKMNPQSTV","BCDEFGHIJKLMNOPQRSTU"),("ABDEGHJKMNPQSTV","ABDEGHJKMNPQSTV"),("ABDEGHJKMNPQSTV","ACEGIKMOQSU"),("ABDEGHJKMNPQSTV","BDFHJLNPRTV"),("ACEGIKMOQSU","ABCDEFGHIJKLMNOPQRSTUV"),("ACEGIKMOQSU","ABCDEFGHIJKLMNOPQRSTU"),("ACEGIKMOQSU","BCDEFGHIJKLMNOPQRSTUV"),("ACEGIKMOQSU","BCDEFGHIJKLMNOPQRSTU"),("ACEGIKMOQSU","ABDEGHJKMNPQSTV"),("ACEGIKMOQSU","ACEGIKMOQSU"),("ACEGIKMOQSU","BDFHJLNPRTV"),("BDFHJLNPRTV","ABCDEFGHIJKLMNOPQRSTUV"),("BDFHJLNPRTV","ABCDEFGHIJKLMNOPQRSTU"),("BDFHJLNPRTV","BCDEFGHIJKLMNOPQRSTUV"),("BDFHJLNPRTV","BCDEFGHIJKLMNOPQRSTU"),("BDFHJLNPRTV","ABDEGHJKMNPQSTV"),("BDFHJLNPRTV","ACEGIKMOQSU"),("BDFHJLNPRTV","BDFHJLNPRTV"),("ACDEFGHIJKLMNOPQRST","ACDEFGHIJKLMNOPQRST")
    ],
    "N25A.csv": [
        ("ABCDEFGHIJKLMNOPQRSTUVWXY","ABCDEFGHIJKLMNOPQRSTUVWXY"),("ABCDEFGHIJKLMNOPQRSTUVWXY","ABCDEFGHIJKLMNOPQRSTUVWX"),("ABCDEFGHIJKLMNOPQRSTUVWXY","BCDEFGHIJKLMNOPQRSTUVWXY"),("ABCDEFGHIJKLMNOPQRSTUVWXY","BCDEFGHIJKLMNOPQRSTUVWX"),("ABCDEFGHIJKLMNOPQRSTUVWXY","ABDEGHJKMNPQSTVWY"),("ABCDEFGHIJKLMNOPQRSTUVWXY","ACEGIKMOQSUWY"),("ABCDEFGHIJKLMNOPQRSTUVWXY","BDFHJLNPRTVX"),("ABCDEFGHIJKLMNOPQRSTUVWX","ABCDEFGHIJKLMNOPQRSTUVWXY"),("ABCDEFGHIJKLMNOPQRSTUVWX","ABCDEFGHIJKLMNOPQRSTUVWX"),("ABCDEFGHIJKLMNOPQRSTUVWX","BCDEFGHIJKLMNOPQRSTUVWXY"),("ABCDEFGHIJKLMNOPQRSTUVWX","BCDEFGHIJKLMNOPQRSTUVWX"),("ABCDEFGHIJKLMNOPQRSTUVWX","ABDEGHJKMNPQSTVWY"),("ABCDEFGHIJKLMNOPQRSTUVWX","ACEGIKMOQSUWY"),("ABCDEFGHIJKLMNOPQRSTUVWX","BDFHJLNPRTVX"),("BCDEFGHIJKLMNOPQRSTUVWXY","ABCDEFGHIJKLMNOPQRSTUVWXY"),("BCDEFGHIJKLMNOPQRSTUVWXY","ABCDEFGHIJKLMNOPQRSTUVWX"),("BCDEFGHIJKLMNOPQRSTUVWXY","BCDEFGHIJKLMNOPQRSTUVWXY"),("BCDEFGHIJKLMNOPQRSTUVWXY","BCDEFGHIJKLMNOPQRSTUVWX"),("BCDEFGHIJKLMNOPQRSTUVWXY","ABDEGHJKMNPQSTVWY"),("BCDEFGHIJKLMNOPQRSTUVWXY","ACEGIKMOQSUWY"),("BCDEFGHIJKLMNOPQRSTUVWXY","BDFHJLNPRTVX"),("ABCDEFGHIJKLMNOPQRSTUVWX","ABCDEFGHIJKLMNOPQRSTUVWXY"),("ABCDEFGHIJKLMNOPQRSTUVWX","ABCDEFGHIJKLMNOPQRSTUVWX"),("ABCDEFGHIJKLMNOPQRSTUVWX","BCDEFGHIJKLMNOPQRSTUVWXY"),("ABCDEFGHIJKLMNOPQRSTUVWX","BCDEFGHIJKLMNOPQRSTUVWX"),("ABCDEFGHIJKLMNOPQRSTUVWX","ABDEGHJKMNPQSTVWY"),("ABCDEFGHIJKLMNOPQRSTUVWX","ACEGIKMOQSUWY"),("ABCDEFGHIJKLMNOPQRSTUVWX","BDFHJLNPRTVX"),("ABDEGHJKMNPQSTVWY","ABCDEFGHIJKLMNOPQRSTUVWXY"),("ABDEGHJKMNPQSTVWY","ABCDEFGHIJKLMNOPQRSTUVWX"),("ABDEGHJKMNPQSTVWY","BCDEFGHIJKLMNOPQRSTUVWXY"),("ABDEGHJKMNPQSTVWY","BCDEFGHIJKLMNOPQRSTUVWX"),("ABDEGHJKMNPQSTVWY","ABDEGHJKMNPQSTVWY"),("ABDEGHJKMNPQSTVWY","ACEGIKMOQSUWY"),("ABDEGHJKMNPQSTVWY","BDFHJLNPRTVX"),("ACEGIKMOQSUWY","ABCDEFGHIJKLMNOPQRSTUVWXY"),("ACEGIKMOQSUWY","ABCDEFGHIJKLMNOPQRSTUVWX"),("ACEGIKMOQSUWY","BCDEFGHIJKLMNOPQRSTUVWXY"),("ACEGIKMOQSUWY","BCDEFGHIJKLMNOPQRSTUVWX"),("ACEGIKMOQSUWY","ABDEGHJKMNPQSTVWY"),("ACEGIKMOQSUWY","ACEGIKMOQSUWY"),("ACEGIKMOQSUWY","BDFHJLNPRTVX"),("BDFHJLNPRTVX","ABCDEFGHIJKLMNOPQRSTUVWXY"),("BDFHJLNPRTVX","ABCDEFGHIJKLMNOPQRSTUVWX"),("BDFHJLNPRTVX","BCDEFGHIJKLMNOPQRSTUVWXY"),("BDFHJLNPRTVX","BCDEFGHIJKLMNOPQRSTUVWX"),("BDFHJLNPRTVX","ABDEGHJKMNPQSTVWY"),("BDFHJLNPRTVX","ACEGIKMOQSUWY"),("BDFHJLNPRTVX","BDFHJLNPRTVX"),("ACDEFGHIJKLMNOPQRSTVX","ACDEFGHIJKLMNOPQRSTVX")
    ],
}

def letras_a_bitstring(letras: str, n: int) -> str:
    letras_set = set(letras or "")
    bits = []
    for i in range(n):
        char = chr(ord('A') + i)
        bits.append('1' if char in letras_set else '0')
    return ''.join(bits)

def _load_tpm(path):
    path_key = str(path)
    if path_key in _WORKER_TPM_CACHE:
        # DEVOLVER COPIA para evitar que estrategias muten el caché maestro
        return _WORKER_TPM_CACHE[path_key].copy()

    npy_path = path.with_suffix('.npy')
    if npy_path.exists():
        tpm = np.load(str(npy_path))
        if tpm.dtype != np.float32:
            tpm = np.asarray(tpm, dtype=np.float32)
    else:
        tpm = np.genfromtxt(str(path), delimiter=',')
        tpm = np.asarray(tpm, dtype=np.float32)
        try:
            np.save(str(npy_path), tpm)
        except Exception:
            pass

    # Guardar referencia en caché (maestro)
    _WORKER_TPM_CACHE[path_key] = tpm
    # DEVOLVER COPIA para prevenir mutaciones in-place posteriores
    return tpm.copy()

def ejecutar_estrategia(estrategia_cls, tpm, inicial, cond, alc, mec, k):
    start = time.time()
    # Instanciar fresco en cada tarea para evitar choques de dimensiones
    instancia = estrategia_cls(tpm)

    aplicar = getattr(instancia, 'aplicar_estrategia')
    res = aplicar(inicial, cond, alc, mec, k)
    tiempo = time.time() - start

    delta = None
    particion = None
    if hasattr(res, 'perdida'):
        delta = float(res.perdida)
    if hasattr(res, 'particion'):
        particion = str(res.particion)
    return delta, tiempo, particion

def discover_samples() -> list[Path]:
    samples = []
    qnodes_samples = QNODES_SRC / '.samples'
    if qnodes_samples.exists():
        samples.extend(sorted(qnodes_samples.glob('*.csv')))
    
    geomip_samples = ROOT / 'GeoMIP' / 'data' / 'samples'
    if geomip_samples.exists():
        samples.extend(sorted(geomip_samples.glob('*.csv')))
    return samples

# --- CORRECCIÓN CRÍTICA DE MULTIPROCESSING ---
# _worker debe estar fuera de main() para que Windows pueda serializarlo (pickle)
def _worker(job):
    # Asegurar que los procesos hijos también desactiven logging
    logging.disable(logging.CRITICAL)
    out = {
        'sample': job['sample'].name,
        'k': job['k'],
        'estrategia': job['estrategia_name'],
        'delta': None,
        'tiempo': None,
        'particion': None,
        'alcance': job.get('alc_letras'),
        'mecanismo': job.get('mec_letras'),
    }
    try:
        tpm = _load_tpm(job['sample'])
        delta, tiempo, particion = ejecutar_estrategia(
            job['estrategia_cls'], tpm, job['inicial'], job['cond'], job['alc'], job['mec'], job['k']
        )
        out['delta'] = delta
        out['tiempo'] = tiempo
        out['particion'] = particion
    except Exception as e:
        # Evitar picos de colorama escribiendo directamente a un archivo de texto aislado
        try:
            import traceback as _tb
            with open("worker_errors.txt", "a", encoding="utf-8") as err_file:
                err_file.write(f"\n--- ERROR EN: {job['sample'].name} | k={job['k']} | {job['estrategia_name']} ---\n")
                _tb.print_exc(file=err_file)
        except Exception:
            pass  # Absoluta seguridad de que nada tumbe el proceso trabajador
        
        out['delta'] = None
        out['tiempo'] = None
        out['particion'] = f'ERROR: {type(e).__name__}: {e}'
    return out

def main():
    results_dir = ROOT / 'results'
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / 'batch_resultados_final.csv'

    estrategias = [
        ('KQNodes', KQNodes),
        ('KGeoMIP', KGeoMIP),
    ]

    ks = [2, 3, 4, 5]
    samples = discover_samples()
    
    jobs = []
    max_n = 0
    for red_name, pruebas in PRUEBAS_POR_RED.items():
        candidates = [s for s in samples if s.name == red_name]
        if not candidates:
            print(f'Warning: no se encontró muestra para {red_name}')
            continue
        sample_path = candidates[0]

        tpm = _load_tpm(sample_path)
        n = int(tpm.shape[1])
        max_n = max(max_n, n)
        inicial = '1' * n
        cond = '1' * n

        for alcance_letras, mecanismo_letras in pruebas:
            alc_bits = letras_a_bitstring(alcance_letras, n)
            mec_bits = letras_a_bitstring(mecanismo_letras, n)
            for k in ks:
                for name, cls in estrategias:
                    jobs.append({
                        'sample': sample_path,
                        'k': k,
                        'estrategia_name': name,
                        'estrategia_cls': cls,
                        'inicial': inicial,
                        'cond': cond,
                        'alc': alc_bits,
                        'mec': mec_bits,
                        'alc_letras': alcance_letras,
                        'mec_letras': mecanismo_letras,
                    })

    cpu_count = os.cpu_count() or 1
    env_workers = os.getenv('RUN_BATCH_FINAL_PROCESSES')
    if env_workers:
        worker_count = min(max(int(env_workers), 1), cpu_count)
    elif max_n >= 22:
        worker_count = 1
    elif max_n >= 20:
        worker_count = min(2, cpu_count)
    else:
        worker_count = min(4, cpu_count)

    print(f'Usando {worker_count} procesos para redes con n={max_n}')

    # Ejecutar con Pool de procesos
    with Pool(processes=worker_count) as p, open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['sample', 'k', 'estrategia', 'delta', 'tiempo', 'particion', 'alcance', 'mecanismo']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for res in p.imap_unordered(_worker, jobs, chunksize=1):
            writer.writerow({k: res.get(k) for k in fieldnames})

    print('Batch final completado. Resultados guardados en', csv_path)

if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent
    import os
    os.chdir(str(ROOT))
    print('Ejecutando run_batch_final...')
    main()