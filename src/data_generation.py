import numpy as np
import pandas as pd

def classify_residues(residue):
    if residue < 33.33:
        return 0  # Bajo
    elif residue < 66.66:
        return 1  # Medio
    else:
        return 2  # Alto

def generate_random_data(num_samples):
    # Coordenadas aproximadas para San Francisco de Orellana, Ecuador
    lat_min, lat_max = -0.477, -0.456  # Latitud entre -0.477 y -0.456
    lon_min, lon_max = -76.999, -76.978  # Longitud entre -76.999 y -76.978

    data = {
        'residuos': np.random.uniform(0, 100, num_samples),  # Cantidad de residuos en %
        'dia_semana': np.random.choice(range(1, 8), num_samples),  # Día de la semana (1-7)
        'hora': np.random.choice(range(0, 24), num_samples),  # Hora del día (0-23)
        'temperatura': np.random.uniform(15, 35, num_samples),  # Temperatura en grados Celsius
        'latitud': np.random.uniform(lat_min, lat_max, num_samples),  # Latitud en rango especificado
        'longitud': np.random.uniform(lon_min, lon_max, num_samples),  # Longitud en rango especificado
        'nivel_residuos': np.random.choice([0, 1, 2], num_samples)  # Nivel de residuos (0: bajo, 1: medio, 2: alto)
    }
    return pd.DataFrame(data)
