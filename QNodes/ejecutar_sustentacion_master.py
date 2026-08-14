# =====================================================================
# SCRIPT MAESTRO DE SUSTENTACIÓN EN VIVO - K-QGMIP (TODO EN UNO)
# =====================================================================
import numpy as np
from pathlib import Path
import gc
import time

# Importar el orquestador y las estrategias
from src.controllers.manager import Manager
from src.strategies.k_qnodes import KQNodes
from src.strategies.k_geomip import KGeoMIP

# =====================================================================
# 1. PANEL DE CONTROL DE LA SUSTENTACIÓN (Modifica aquí en vivo)
# =====================================================================

# A. Modo de ejecución: "MANUAL" (caso único) o "LISTA_PRUEBAS" (las del Excel)
MODO = "MANUAL"  

# B. Selección de Estrategia: "KQNodes", "KGeoMIP" o "AMBAS"
ESTRATEGIA = "AMBAS"  

# C. Configuración de red y bloques k
RED_NOMBRE = "N20A.csv"     # "N3A.csv", "N5A.csv", "N10A.csv", "N15B.csv", etc.
K_BLOQUES  = 3             # k >= 2

# D. Si MODO es "MANUAL", define aquí las máscaras de entrada (letras)
# (El script ajustará automáticamente las dimensiones según la red cargada)
ALCANCE_MANUAL     = "ABCDEFGHIJKLMNOPQRST"  # Asegurar que tenga la misma longitud que N
MECANISMO_MANUAL   = "ABCDEFGHIJKLMNOPQRST"  # Asegurar que tenga la misma longitud que N
CONDICIONES_MANUAL = "11111111111111111111"  # Asegurar que tenga la misma longitud que N

# E. Si MODO es "LISTA_PRUEBAS", define qué número de prueba correr (1 a 49)
# Si pones None, el script correrá las 49 pruebas secuencialmente.
PRUEBA_A_EJECUTAR = 1  

# =====================================================================
# 2. BASE DE DATOS DE PRUEBAS DE LA DOCENTE (49 combinaciones para N10A)
# =====================================================================
PRUEBAS_POR_RED = {
    "N10A.csv": [
        ("ABCDEFGHIJ","ABCDEFGHIJ"), ("ABCDEFGHIJ","ABCDEFGHI"), ("ABCDEFGHIJ","BCDEFGHIJ"), ("ABCDEFGHIJ","BCDEFGHI"), ("ABCDEFGHIJ","ABDEGHJ"), ("ABCDEFGHIJ","ACEGI"), ("ABCDEFGHIJ","BDFHJ"),
        ("ABCDEFGHI","ABCDEFGHIJ"), ("ABCDEFGHI","ABCDEFGHI"), ("ABCDEFGHI","BCDEFGHIJ"), ("ABCDEFGHI","BCDEFGHI"), ("ABCDEFGHI","ABDEGHJ"), ("ABCDEFGHI","ACEGI"), ("ABCDEFGHI","BDFHJ"),
        ("BCDEFGHIJ","ABCDEFGHIJ"), ("BCDEFGHIJ","ABCDEFGHI"), ("BCDEFGHIJ","BCDEFGHIJ"), ("BCDEFGHIJ","BCDEFGHI"), ("BCDEFGHIJ","ABDEGHJ"), ("BCDEFGHIJ","ACEGI"), ("BCDEFGHIJ","BDFHJ"),
        ("BCDEFGHI","ABCDEFGHIJ"), ("BCDEFGHI","ABCDEFGHI"), ("BCDEFGHI","BCDEFGHIJ"), ("BCDEFGHI","BCDEFGHI"), ("BCDEFGHI","ABDEGHJ"), ("BCDEFGHI","ACEGI"), ("BCDEFGHI","BDFHJ"),
        ("ABDEGHJ","ABCDEFGHIJ"), ("ABDEGHJ","ABCDEFGHI"), ("ABDEGHJ","BCDEFGHIJ"), ("ABDEGHJ","BCDEFGHI"), ("ABDEGHJ","ABDEGHJ"), ("ABDEGHJ","ACEGI"), ("ABDEGHJ","BDFHJ"),
        ("ACEGI","ABCDEFGHIJ"), ("ACEGI","ABCDEFGHI"), ("ACEGI","BCDEFGHIJ"), ("ACEGI","BCDEFGHI"), ("ACEGI","ABDEGHJ"), ("ACEGI","ACEGI"), ("ACEGI","BDFHJ"),
        ("BDFHJ","ABCDEFGHIJ"), ("BDFHJ","ABCDEFGHI"), ("BDFHJ","BCDEFGHIJ"), ("BDFHJ","BCDEFGHI"), ("BDFHJ","ABDEGHJ"), ("BDFHJ","ACEGI"), ("BDFHJ","BDFHJ")
    ]
}

# =====================================================================
# 3. FUNCIONES AUXILIARES DE CONVERSIÓN Y TRADUCCIÓN
# =====================================================================
def letras_a_bitstring(letras, N):
    """Convierte un conjunto de letras como 'ACEGI' a un bitstring como '1010101000'"""
    if set(letras).issubset({"0", "1"}):
        return letras  # Ya es binario
    mask = ["0"] * N
    for char in letras.upper():
        idx = ord(char) - ord('A')
        if 0 <= idx < N:
            mask[idx] = "1"
    return "".join(mask)

def ejecutar_tarea(manager, condiciones, alcance, mecanismo, tpm, estrategia, k):
    """Ejecuta de manera controlada la estrategia seleccionada con la firma correcta de QNodes"""
    res_particion, res_delta, res_tiempo = "N/A", 0.0, 0.0
    tpm_input = tpm.copy()
    
    if estrategia == "KQNodes":
        try:
            est = KQNodes(tpm_input, k)
        except TypeError:
            try:
                est = KQNodes(tpm_input, k=k)
            except TypeError:
                try:
                    est = KQNodes(tpm_input)
                    est.k = k
                except TypeError:
                    est = KQNodes(manager)
                    est.k = k
                    
        est.manager = manager
        t0 = time.time()
        sol = est.aplicar_estrategia(manager.estado_inicial, condiciones, alcance, mecanismo, k)
        res_tiempo = time.time() - t0
        res_particion, res_delta = sol.particion, sol.perdida
        del est
        
    elif estrategia == "KGeoMIP":
        try:
            est = KGeoMIP(tpm_input, k)
        except TypeError:
            try:
                est = KGeoMIP(tpm_input, k=k)
            except TypeError:
                try:
                    est = KGeoMIP(tpm_input)
                    est.k = k
                except TypeError:
                    est = KGeoMIP(manager)
                    est.k = k
                    
        est.manager = manager
        t0 = time.time()
        sol = est.aplicar_estrategia(manager.estado_inicial, condiciones, alcance, mecanismo, k)
        res_tiempo = time.time() - t0
        res_particion, res_delta = sol.particion, sol.perdida
        del est
        
    gc.collect()
    return res_particion, res_delta, res_tiempo

# =====================================================================
# 4. FLUJO PRINCIPAL DE EJECUCIÓN (UNIFICADO)
# =====================================================================
def main():
    # Obtener el número de variables N deducido del nombre de la red
    N = int(''.join(filter(str.isdigit, RED_NOMBRE)))
    print(f"=== INICIANDO SISTEMA K-QGMIP (N={N} variables, k={K_BLOQUES}) ===")
    
    # Cargar TPM
    ruta_tpm = Path("src") / ".samples" / RED_NOMBRE
    if not ruta_tpm.exists():
        ruta_tpm = Path("QNodes") / "src" / ".samples" / RED_NOMBRE
    
    print(f"[*] Cargando TPM desde: {ruta_tpm}")
    tpm = np.genfromtxt(ruta_tpm, delimiter=",", dtype=np.float32)
    
    # AUTODETECCIÓN DEL ESTADO INICIAL SEGÚN N (Evita ERROR_ESPACIOS_INCOMPATIBLES)
    estado_inicial_correcto = "1" + "0" * (N - 1)
    manager = Manager(estado_inicial=estado_inicial_correcto)
    
    # Resolver tareas a ejecutar
    tareas = []
    if MODO == "MANUAL":
        alc_bit = letras_a_bitstring(ALCANCE_MANUAL, N)
        mec_bit = letras_a_bitstring(MECANISMO_MANUAL, N)
        cond_bit = letras_a_bitstring(CONDICIONES_MANUAL, N)
        tareas.append((1, ALCANCE_MANUAL, MECANISMO_MANUAL, cond_bit, alc_bit, mec_bit))
    else:
        lista_original = PRUEBAS_POR_RED.get(RED_NOMBRE, [])
        if not lista_original:
            print(f"Error: No hay pruebas predefinidas para la red {RED_NOMBRE}")
            return
        
        if PRUEBA_A_EJECUTAR is not None:
            idx = PRUEBA_A_EJECUTAR - 1
            alc, mec = lista_original[idx]
            alc_bit = letras_a_bitstring(alc, N)
            mec_bit = letras_a_bitstring(mec, N)
            cond_bit = "1" * N
            tareas.append((PRUEBA_A_EJECUTAR, alc, mec, cond_bit, alc_bit, mec_bit))
        else:
            for i, (alc, mec) in enumerate(lista_original, start=1):
                alc_bit = letras_a_bitstring(alc, N)
                mec_bit = letras_a_bitstring(mec, N)
                cond_bit = "1" * N
                tareas.append((i, alc, mec, cond_bit, alc_bit, mec_bit))

    # --- PREPARACIÓN DEL ARCHIVO CSV DE SALIDA ---
    Path("results").mkdir(parents=True, exist_ok=True)
    ruta_csv_vivo = Path("results") / "resultados_sustentacion_en_vivo.csv"
    
    if not ruta_csv_vivo.exists():
        with open(ruta_csv_vivo, "w", encoding="utf-8") as f:
            f.write("Prueba,Red,k,Estrategia,Delta,Tiempo_s,Particion\n")

    # --- BUCLE ÚNICO DE EJECUCIÓN ---
    for num, alc_txt, mec_txt, cond, alc, mec in tareas:
        print(f"\n────────────────────────────────────────────────────────────")
        print(f"PRUEBA #{num} | Alcance: {alc_txt} ({alc}) | Mecanismo: {mec_txt} ({mec})")
        print(f"────────────────────────────────────────────────────────────")
        
        with open(ruta_csv_vivo, "a", encoding="utf-8") as f:
            if ESTRATEGIA in ["KQNodes", "AMBAS"]:
                part_kq, delta_kq, t_kq = ejecutar_tarea(manager, cond, alc, mec, tpm, "KQNodes", K_BLOQUES)
                print(f"[KQNodes]  Pérdida: {delta_kq:.6f} | Tiempo: {t_kq:.4f}s")
                print(f"           Partición: {part_kq}")
                part_limpia = str(part_kq).replace("\n", " ").replace(",", ";")
                f.write(f"{num},{RED_NOMBRE},{K_BLOQUES},KQNodes,{delta_kq:.6f},{t_kq:.4f},\"{part_limpia}\"\n")
                
            if ESTRATEGIA in ["KGeoMIP", "AMBAS"]:
                part_geo, delta_geo, t_geo = ejecutar_tarea(manager, cond, alc, mec, tpm, "KGeoMIP", K_BLOQUES)
                print(f"[KGeoMIP]  Pérdida: {delta_geo:.6f} | Tiempo: {t_geo:.4f}s")
                print(f"           Partición: {part_geo}")
                part_limpia = str(part_geo).replace("\n", " ").replace(",", ";")
                f.write(f"{num},{RED_NOMBRE},{K_BLOQUES},KGeoMIP,{delta_geo:.6f},{t_geo:.4f},\"{part_limpia}\"\n")
            
        if ESTRATEGIA == "AMBAS":
            razon_tiempo = t_geo / t_kq if t_kq > 0 else 0
            print(f"--> Comparativa: KGeoMIP fue {razon_tiempo:.2f}x de rápido respecto a KQNodes.")
            
    print(f"\n[*] Resultados de la sustentación registrados en: {ruta_csv_vivo}")

if __name__ == "__main__":
    main()