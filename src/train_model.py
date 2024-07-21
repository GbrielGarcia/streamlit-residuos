import pandas as pd
# Divide el conjunto de datos en dos partes ( entrenamiento y prueba )
from sklearn.model_selection import train_test_split
# Implementa el algoritmo de regresión lineal [ predicion continua en base a una entrada]
from sklearn.linear_model import LinearRegression
# Calcula el error cuadratico medio (MSE) entre las predicciones del modelo y los valores reales.
# [ El MSE mide la diferencia promedio al cuadrado entre las predicciones y los valores verdaderos ]
from sklearn.metrics import mean_squared_error
import pickle

def train_model(data):
    # Preparar los datos
    X = data[['dia_semana', 'hora', 'temperatura']]
    y = data['residuos']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    print(f'Error Cuadrático Medio (MSE): {mse}')

    # Guardar el modelo entrenado
    with open('models/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return model, mse
