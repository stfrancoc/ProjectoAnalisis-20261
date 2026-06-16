def ERROR_ESPACIOS_INCOMPATIBLES(espacio: int) -> str:
    return f"Estado inicial debe tener longitud {espacio}"


# Validación de k-particiones (src/funcs/particion.py)
ERROR_PARTICION_NODO_DUPLICADO: str = (
    "Partición inválida: algún nodo aparece más de una vez (en uno o varios bloques)."
)


def ERROR_PARTICION_COBERTURA(faltantes: set, ajenos: set) -> str:
    return (
        "Partición inválida: la cobertura de nodos no es exacta. "
        f"Nodos faltantes: {sorted(faltantes)}; nodos ajenos: {sorted(ajenos)}."
    )


def ERROR_PARTICION_K_RANGO(k: int, m: int) -> str:
    return (
        f"k inválido: se requiere 2 <= k <= m (m={m} nodos particionables), "
        f"pero se recibió k={k}."
    )


def ERROR_PARTICION_K_BLOQUES(k: int, no_vacios: int) -> str:
    return (
        f"Partición inválida: se esperaban exactamente k={k} bloques no vacíos, "
        f"pero hay {no_vacios}."
    )
