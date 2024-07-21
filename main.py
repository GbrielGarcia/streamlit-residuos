# main.py
import time
import streamlit as st
import pandas as pd
from src.data_generation import generate_random_data
from src.train_model import train_model
import io
import os


# Función para generar, guardar datos, entrenar el modelo y mostrar los resultados
def run_app(num_samples):
    # Generar datos aleatorios
    data = generate_random_data(num_samples)

    # Entrenando al modelo
    model, mse = train_model(data)

    # Preparando los datos para las predicciones
    X = data[['dia_semana', 'hora', 'temperatura']]
    y_actual = data['residuos']
    y_pred = model.predict(X)

    # Visualizar los datos generados
    st.subheader('Datos Generados')
    st.dataframe(data)

    # Mostrar el rendimineto del modelo
    st.subheader('Evaluación del Modelo')
    st.write('Error Cuadrático Medio (MSE):', mse)

    # Visualizar el gráfico con predicciones
    st.subheader('Predicciones del Modelo')
    results = pd.DataFrame({'Actual': y_actual, 'Predicción': y_pred})
    st.line_chart(results)

    # Visualizacion del gráfico con de datos generado y resultados en un único DataFrame
    combined_results = data.copy()
    combined_results['Predicción'] = y_pred

    # Butón para guardar los datos y predicciones generados
    buffer = io.StringIO()
    combined_results.to_csv(buffer, index=False)
    st.download_button(
        label="Descargar Información Actual",
        data=buffer.getvalue(),
        file_name='resultados.csv',
        mime='text/csv'
    )

    # Guardar locamente los datos generados
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    combined_results.to_csv(f'data/processed/resultados_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            index=False)

    # Mensaje de éxito
    message = st.empty()
    message.success("Datos guardados exitosamente.")
    time.sleep(5)
    message.empty()


# Función de visualización de los datos guardados
def show_saved_data():
    st.subheader('Datos Guardados')
    files = [f for f in os.listdir('data/processed') if f.endswith('.csv')]
    if not files:
        st.write("No hay datos guardados.")
        return

    for file in files:
        data = pd.read_csv(os.path.join('data/processed', file))
        col1, col2 = st.columns(2)
        with col1:
            st.write(f'**{file}**')
            st.dataframe(data)
        with col2:
            if 'Predicción' in data.columns:
                st.write('Resultados')
                st.line_chart(data[['residuos', 'Predicción']])
            else:
                st.write("La columna 'Predicción' no se encontró en los datos.")


# Titulo
st.title('Optimización de la Gestión de Residuos con Machine Learning')

# Pestañas de la aplicación
tab1, tab2 = st.tabs(["Generar Nueva Información", "Visualizar Datos Guardados"])

with tab1:
    # Control, para el selector de número a generar
    num_samples = st.slider("Seleccione el número de datos a generar", min_value=2, max_value=1000, value=100)

    # Botón para generar nueva información
    if st.button('Generar Nueva Información'):
        run_app(num_samples)

with tab2:
    show_saved_data()

# Makdown Pie de pagina
st.markdown("""
---
Hecho por:
- Roxana Mabel toala torres
- Maria Fernanda Dueñas Murillo
""")