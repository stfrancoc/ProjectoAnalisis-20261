"""
Genera de forma determinista los TPM faltantes (N20A, N22A, N25A) usando el
generador existente `Manager.generar_red` (semilla numpy = 73, datos binarios), y
los verifica. NO modifica `generar_red` ni ninguna otra parte del proyecto.

Ejecutar desde el directorio del proyecto con:  `uv run generar_redes.py`

Verificación:
- N20A / N22A: carga completa (`Manager.cargar_red`), confirma shape, reporta
  filas/columnas/tamaño/tiempo de carga.
- N25A: verificación LIGERA (cuenta filas por bloques sin cargar todo, lee solo la
  primera fila para las columnas, cronometra una carga parcial y extrapola el
  tiempo y la memoria de la carga completa).
"""

import builtins
import os
import time
from pathlib import Path

import numpy as np

from src.constants.base import COLON_DELIM, PATH_SAMPLES
from src.controllers.manager import Manager
from src.models.base.application import aplicacion

DIMENSIONES = [20, 22, 25]
FILAS_PARCIAL = 100_000  # filas para la carga parcial de N25A
DIM_LIGERA = 25  # esta dimensión se verifica de forma ligera


def _input_automatico_si(*_args, **_kwargs) -> str:
    """Responde 's' a cualquier prompt durante la generación (parche temporal)."""
    return "s"


def generar(dim: int) -> tuple[Manager, str, float]:
    """
    Genera N{dim}A de forma determinista, parcheando `input` por si surgiera algún
    prompt (no debería para estas dimensiones, todas < 1 GB en memoria).
    """
    gestor = Manager("0" * dim)  # el estado_inicial es solo un placeholder aquí
    original_input = builtins.input
    builtins.input = _input_automatico_si
    inicio = time.time()
    try:
        nombre = gestor.generar_red(dim, datos_deterministas=True)
    finally:
        builtins.input = original_input
    return gestor, nombre, time.time() - inicio


def contar_filas_binario(ruta: Path, chunk: int = 1 << 22) -> int:
    """Cuenta filas (saltos de línea) leyendo en bloques, sin cargar todo el CSV."""
    total = 0
    cola_sin_salto = False
    with open(ruta, "rb") as fichero:
        while True:
            datos = fichero.read(chunk)
            if not datos:
                break
            total += datos.count(b"\n")
            cola_sin_salto = not datos.endswith(b"\n")
    # Si la última línea no termina en '\n', cuéntala igual.
    return total + (1 if cola_sin_salto else 0)


def columnas_primera_fila(ruta: Path) -> int:
    """Lee únicamente la primera línea y cuenta sus columnas."""
    with open(ruta, "r", encoding="utf-8") as fichero:
        primera = fichero.readline().strip()
    return len([valor for valor in primera.split(COLON_DELIM) if valor != ""])


def tamano_mb(ruta: Path) -> float:
    return os.path.getsize(ruta) / (1024**2)


def verificar_completo(
    gestor: Manager, dim: int, ruta: Path
) -> tuple[int, int, float, bool]:
    """Carga completa de N20A/N22A y verifica shape == (2**dim, dim)."""
    inicio = time.time()
    datos = gestor.cargar_red()  # np.genfromtxt(delimiter=',')
    tiempo_carga = time.time() - inicio
    filas, columnas = datos.shape
    ok = filas == (1 << dim) and columnas == dim
    del datos
    return filas, columnas, tiempo_carga, ok


def verificar_ligero(dim: int, ruta: Path) -> dict:
    """Verificación ligera de N25A: filas por bloques + carga parcial extrapolada."""
    print(f"  [ligero] contando filas por bloques de {ruta.name}...")
    filas = contar_filas_binario(ruta)
    columnas = columnas_primera_fila(ruta)

    print(f"  [ligero] carga parcial de {FILAS_PARCIAL:,} filas para extrapolar...")
    inicio = time.time()
    parcial = np.genfromtxt(ruta, delimiter=COLON_DELIM, max_rows=FILAS_PARCIAL)
    tiempo_parcial = time.time() - inicio
    filas_parciales = parcial.shape[0]
    del parcial

    filas_totales = 1 << dim
    factor = filas_totales / filas_parciales
    tiempo_estimado = tiempo_parcial * factor
    memoria_estimada_gb = (filas_totales * columnas * 8) / (1024**3)  # float64

    ok = filas == filas_totales and columnas == dim
    return {
        "filas": filas,
        "columnas": columnas,
        "tiempo_parcial": tiempo_parcial,
        "filas_parciales": filas_parciales,
        "tiempo_estimado": tiempo_estimado,
        "memoria_estimada_gb": memoria_estimada_gb,
        "ok": ok,
    }


def main() -> None:
    resultados: list[dict] = []

    for dim in DIMENSIONES:
        print("=" * 64)
        print(f"GENERANDO N{dim}A  (2**{dim} = {1 << dim:,} filas x {dim} columnas)")
        print("=" * 64)

        gestor, nombre, tiempo_generacion = generar(dim)
        ruta = Path(PATH_SAMPLES) / f"N{dim}A.csv"

        if not ruta.exists():
            print(f"  !! No se encontró {ruta} tras generar (nombre devuelto: {nombre})")
            resultados.append(
                {
                    "red": f"N{dim}A",
                    "dim": dim,
                    "filas": "-",
                    "columnas": "-",
                    "tamano_mb": "-",
                    "t_gen": tiempo_generacion,
                    "t_carga": "-",
                    "verificacion": "NO GENERADO",
                }
            )
            continue

        size_mb = tamano_mb(ruta)
        print(f"  Generado: {nombre}  ({size_mb:,.1f} MB en disco)")
        print(f"  Tiempo de generación+guardado: {tiempo_generacion:.2f} s")

        if dim >= DIM_LIGERA:
            print(f"  Verificación LIGERA de N{dim}A (evita cargar ~6.7 GB):")
            info = verificar_ligero(dim, ruta)
            print(
                f"    filas={info['filas']:,}  columnas={info['columnas']}  "
                f"esperado=({1 << dim:,}, {dim})"
            )
            print(
                f"    carga parcial: {info['filas_parciales']:,} filas en "
                f"{info['tiempo_parcial']:.2f} s"
            )
            print(
                f"    ESTIMADO carga completa (float64): "
                f"~{info['tiempo_estimado']:.0f} s "
                f"(~{info['tiempo_estimado'] / 60:.1f} min), "
                f"~{info['memoria_estimada_gb']:.2f} GB en RAM"
            )
            resultados.append(
                {
                    "red": f"N{dim}A",
                    "dim": dim,
                    "filas": info["filas"],
                    "columnas": info["columnas"],
                    "tamano_mb": size_mb,
                    "t_gen": tiempo_generacion,
                    "t_carga": f"~{info['tiempo_estimado']:.0f}s est. "
                    f"(~{info['memoria_estimada_gb']:.1f}GB)",
                    "verificacion": "OK (ligera)" if info["ok"] else "FALLO",
                }
            )
        else:
            print(f"  Verificación COMPLETA de N{dim}A (cargando todo)...")
            filas, columnas, tiempo_carga, ok = verificar_completo(gestor, dim, ruta)
            print(
                f"    shape=({filas:,}, {columnas})  esperado=({1 << dim:,}, {dim})  "
                f"carga={tiempo_carga:.2f} s"
            )
            resultados.append(
                {
                    "red": f"N{dim}A",
                    "dim": dim,
                    "filas": filas,
                    "columnas": columnas,
                    "tamano_mb": size_mb,
                    "t_gen": tiempo_generacion,
                    "t_carga": f"{tiempo_carga:.1f}s",
                    "verificacion": "OK" if ok else "FALLO",
                }
            )

    # Tabla final.
    print("\n" + "=" * 96)
    print("TABLA RESUMEN")
    print("=" * 96)
    encabezado = (
        f"{'Red':<7} {'Dim':<4} {'Filas':>13} {'Cols':>5} {'Tam_MB':>10} "
        f"{'T_gen':>9} {'T_carga':>26} {'Verif.':<12}"
    )
    print(encabezado)
    print("-" * 96)
    for r in resultados:
        filas = f"{r['filas']:,}" if isinstance(r["filas"], int) else str(r["filas"])
        tam = f"{r['tamano_mb']:,.1f}" if isinstance(r["tamano_mb"], float) else str(
            r["tamano_mb"]
        )
        t_gen = f"{r['t_gen']:.1f}s" if isinstance(r["t_gen"], float) else str(r["t_gen"])
        print(
            f"{r['red']:<7} {r['dim']:<4} {filas:>13} {str(r['columnas']):>5} "
            f"{tam:>10} {t_gen:>9} {str(r['t_carga']):>26} {r['verificacion']:<12}"
        )
    print("=" * 96)


if __name__ == "__main__":
    main()
