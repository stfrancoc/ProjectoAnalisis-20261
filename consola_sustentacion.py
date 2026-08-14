# =====================================================================
# CONSOLA MAESTRA DE SUSTENTACIÓN EN VIVO - K-QGMIP
# =====================================================================
import numpy as np
from pathlib import Path
import gc
import time

from src.controllers.manager import Manager
from src.controllers.strategies.k_nodes import KQNodes
from src.controllers.strategies.k_geomip import KGeoMIP

# =====================================================================
# 1. BLOQUE DE CONFIGURACIÓN DE LA SUSTENTACIÓN (Modificar aquí en vivo)
# =====================================================================

# MODO_CORRIDA: 
#   - "HOJA_DOCENTE" -> Corre una de las 49 pruebas del Excel poniendo su número (1 a 49)
#   - "PERSONALIZADO" -> Corre una prueba manual usando las máscaras de abajo
#   - "RED_ALEATORIA" -> Genera una TPM aleatoria nueva desde cero y la evalúa
MODO_CORRIDA = "HOJA_DOCENTE"  
NUMERO_PRUEBA_EXCEL = 24       # Número de fila en la hoja de N10A (1 a 49)

# CONFIGURACIÓN PARA ESTRATEGIA:
#   - "KQNodes" -> Solo corre el voraz aglomerativo
#   - "KGeoMIP" -> Solo corre cortes geométricos
#   - "AMBAS"   -> Corre ambas estrategias secuencialmente y compara delta y tiempo
ESTRATEGIA_A_EVALUAR = "AMBAS"
K_BLOQUES = 3                 # Valor de k-particiones a buscar (k >= 2)

# CONFIGURACIÓN PARA MODO "PERSONALIZADO"
RED_MANUAL = "N10A.csv"
MASCARA_ALCANCE_MANUAL   = "1111111111"
MASCARA_MECANISMO_MANUAL = "1111111111"

# CONFIGURACIÓN PARA MODO "RED_ALEATORIA" (Generación en vivo)
N_ALEATORIO = 6               # Número de variables para crear la red en vivo

# =====================================================================
# 2. BASE DE DATOS DE PRUEBAS DE LA DOCENTE (Red N10A)
# =====================================================================
SISTEMA_LETRAS = "ABCDEFGHIJ"
PRUEBAS_N10 = [
    ("ABCDEFGHIJ","ABCDEFGHIJ"), ("ABCDEFGHIJ","ABCDEFGHI"),  ("ABCDEFGHIJ","BCDEFGHIJ"),
    ("ABCDEFGHIJ","BCDEFGHI"),   ("ABCDEFGHIJ","ABDEGHJ"),    ("ABCDEFGHIJ","ACEGI"),
    ("ABCDEFGHIJ","BDFHJ"),      ("ABCDEFGHI","ABCDEFGHIJ"),  ("ABCDEFGHI","ABCDEFGHI"),
    ("ABCDEFGHI","BCDEFGHIJ"),   ("ABCDEFGHI","BCDEFGHI"),    ("ABCDEFGHI","ABDEGHJ"),
    ("ABCDEFGHI","ACEGI"),       ("ABCDEFGHI","BDFHJ"),       ("BCDEFGHIJ","ABCDEFGHIJ"),
    ... # Se asume la secuencia completa mapeada
]

# Copiamos la lista completa para tener acceso O(1) por índice
PRUEBAS_EXCEL_MAPEADAS = [
    ("ABCDEFGHIJ", "ABCDEFGHIJ"), ("ABCDEFGHIJ", "ABCDEFGHI"), ("ABCDEFGHIJ", "BCDEFGHIJ"), ("ABCDEFGHIJ", "BCDEFGHI"),
    ("ABCDEFGHIJ", "ABDEGHJ"),    ("ABCDEFGHIJ", "ACEGI"),       ("ABCDEFGHIJ", "BDFHJ"),      ("ABCDEFGHI", "ABCDEFGHIJ"),
    ("ABCDEFGHI", "ABCDEFGHI"),   ("ABCDEFGHI", "BCDEFGHIJ"),    ("ABCDEFGHI", "BCDEFGHI"),    ("ABCDEFGHI", "ABDEGHJ"),
    ("ABCDEFGHI", "ACEGI"),       ("ABCDEFGHI", "BDFHJ"),        ("BCDEFGHIJ", "ABCDEFGHIJ"),  ("BCDEFGHIJ", "ABCDEFGHI"),
    ("BCDEFGHIJ", "BCDEFGHIJ"),   ("BCDEFGHIJ", "BCDEFGHI"),     ("BCDEFGHIJ", "ABDEGHJ"),     ("BCDEFGHIJ", "ACEGI"),
    ("BCDEFGHIJ", "BDFHJ"),        ("BCDEFGHI", "ABCDEFGHIJ"),    ("BCDEFGHI", "ABCDEFGHI"),    ("BCDEFGHI", "BCDEFGHIJ"),
    ("BCDEFGHI", "BCDEFGHI"),     ("BCDEFGHI", "ABDEGHJ"),       ("BCDEFGHI", "ACEGI"),        ("BCDEFGHI", "BDFHJ"),
    ("ABDEGHJ", "ABCDEFGHIJ"),    ("ABDEGHJ", "ABCDEFGHI"),      ("ABDEGHJ", "BCDEFGHIJ"),     ("ABDEGHJ", "BCDEFGHI"),
    ("ABDEGHJ", "ABDEGHJ"),       ("ABDEGHJ", "ACEGI"),          ("ABDEGHJ", "BDFHJ"),         ("ACEGI", "ABCDEFGHIJ"),
    ("ACEGI", "ABCDEFGHI"),       ("ACEGI", "BCDEFGHIJ"),        ("ACEGI", "BCDEFGHI"),        ("ACEGI", "ABDEGHJ"),
    ("ACEGI", "ACEGI"),           ("ACEGI", "BDFHJ"),            ("BDFHJ", "ABCDEFGHIJ"),      ("BDFHJ", "ABCDEFGHI"),
    ("BDFHJ", "BCDEFGHIJ"),       ("BDFHJ", "BCDEFGHI"),         ("BDFHJ", "ABDEGHJ"),         ("BDFHJ", "ACEGI"),
    ("BDFHJ", "BDFHJ")
]

# =====================================================================
# 3. FUNCIONES AUXILIARES DE TRADUCCIÓN Y GENERACIÓN
# =====================================================================
def letras_a_bitstring(letras, sistema_letras):
    mask = ["0"] * len(sistema_letras)
    for char in letras:
        idx = sistema_letras.find(char)
        if idx != -1:
            mask[idx] = "1"
    return "".join(mask)

def generar_tpm_aleatoria(n):
    # Genera una TPM estado-nodo aleatoria estructurada de tamaño 2^N x N
    print(f"-> Generando TPM aleatoria in-memory para N={n} variables...")
    np.random.seed(int(time.time()))
    return np.random.uniform(0.0, 1.0, (2**n, n)).astype(np.float32)

# =====================================================================
# 4. ORQUESTACIÓN DEL PROCESAMIENTO
# =====================================================================
def ejecutar_tarea(tpm, estado_inicial, condiciones, alcance, mecanismo, estrategia):
    manager = Manager(estado_inicial=estado_inicial)
    
    if estrategia == "KQNodes":
        instancia = KQNodes(manager, k=K_BLOQUES)
    else:
        instancia = KGeoMIP(manager, k=K_BLOQUES)
        
    instancia.sia_preparar_subsistema(condiciones, alcance, mecanismo, tpm.copy())
    
    t_start = time.time()
    solucion = instancia.aplicar_estrategia()
    t_end = time.time()
    
    return solucion, t_end - t_start

def main():
    print("=====================================================================")
    print("           CONSOLA DE EJECUCIÓN EN VIVO - PROYECTO K-QGMIP           ")
    print("=====================================================================")
    
    # Resolver Modo de Corrida
    if MODO_CORRIDA == "HOJA_DOCENTE":
        idx = NUMERO_PRUEBA_EXCEL - 1
        if idx < 0 or idx >= len(PRUEBAS_EXCEL_MAPEADAS):
            raise IndexError("Error: El número de prueba debe estar entre 1 y 49.")
        
        alcance_letras, mecanismo_letras = PRUEBAS_EXCEL_MAPEADAS[idx]
        print(f"[MODO HOJA DOCENTE] Procesando Prueba #{NUMERO_PRUEBA_EXCEL}:")
        print(f" -> Alcance (letras): {alcance_letras} | Mecanismo (letras): {mecanismo_letras}")
        
        # Traducir a bitstrings usando la nomenclatura de base N10A
        alcance_mask = letras_a_bitstring(alcance_letras, SISTEMA_LETRAS)
        mecanismo_mask = letras_a_bitstring(mecanismo_letras, SISTEMA_LETRAS)
        condiciones_mask = "1" * 10
        estado_inicial = "1" + "0" * 9
        
        ruta_tpm = Path("src/.samples/N10A.csv")
        tpm_input = np.genfromtxt(ruta_tpm, delimiter=",", dtype=np.float32)
        
    elif MODO_CORRIDA == "PERSONALIZADO":
        print(f"[MODO PERSONALIZADO] Procesando archivo '{RED_MANUAL}':")
        alcance_mask = MASCARA_ALCANCE_MANUAL
        mecanismo_mask = MASCARA_MECANISMO_MANUAL
        condiciones_mask = "1" * len(alcance_mask)
        estado_inicial = "1" + "0" * (len(alcance_mask) - 1)
        
        ruta_tpm = Path("src/.samples") / RED_MANUAL
        tpm_input = np.genfromtxt(ruta_tpm, delimiter=",", dtype=np.float32)
        
    else:  # RED_ALEATORIA
        print(f"[MODO RED ALEATORIA] Creando red en vivo con N={N_ALEATORIO}:")
        alcance_mask = "1" * N_ALEATORIO
        mecanismo_mask = "1" * N_ALEATORIO
        condiciones_mask = "1" * N_ALEATORIO
        estado_inicial = "1" + "0" * (N_ALEATORIO - 1)
        tpm_input = generar_tpm_aleatoria(N_ALEATORIO)

    print(f" -> Alcance (bits):   {alcance_mask}")
    print(f" -> Mecanismo (bits): {mecanismo_mask}")
    print(f" -> Valor de k:       {K_BLOQUES}")
    
    # Resolver Estrategias a correr
    estrategias = []
    if ESTRATEGIA_A_EVALUAR in ["KQNodes", "AMBAS"]:
        estrategias.append("KQNodes")
    if ESTRATEGIA_A_EVALUAR in ["KGeoMIP", "AMBAS"]:
        estrategias.append("KGeoMIP")
        
    for estr in estrategias:
        print(f"\nEjecutando estrategia: {estr}...")
        try:
            sol, tiempo = ejecutar_tarea(tpm_input, estado_inicial, condiciones_mask, alcance_mask, mecanismo_mask, estr)
            print(sol)
            print(f"-> Tiempo de ejecución medido: {tiempo:.4f} segundos.")
        except Exception as e:
            print(f"Error al ejecutar {estr}: {e}")
        gc.collect()

if __name__ == "__main__":
    main()