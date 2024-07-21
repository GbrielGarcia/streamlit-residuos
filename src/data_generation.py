import numpy as np
import pandas as pd

def generate_random_data(num_samples):
    data = {
        'residuos': np.random.uniform(0, 100, num_samples),  # Cantidad de residuos en %
        'dia_semana': np.random.choice(range(1, 8), num_samples),  # Día de la semana (1-7)
        'hora': np.random.choice(range(0, 24), num_samples),  # Hora del día (0-23)
        'temperatura': np.random.uniform(15, 35, num_samples)  # Temperatura en grados Celsius
    }
    return pd.DataFrame(data)

