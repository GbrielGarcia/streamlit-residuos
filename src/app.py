import io
import os
import pandas as pd
import streamlit as st
from src.data_generation import generate_random_data
from src.train_model import train_model

def run_app(num_samples):
    # Generar datos aleatorios
    data = generate_random_data(num_samples)

    # Entrenando al modelo
    model, mse = train_model(data)

    # Preparando los datos para las predicciones
    X = data[['dia_semana', 'hora', 'temperatura', 'latitud', 'longitud']]
    y_actual = data['residuos']
    y_pred = model.predict(X)

    # Visualizar los datos generados
    st.subheader('Datos Generados')
    st.dataframe(data)

    # Mostrar el rendimiento del modelo
    st.subheader('Evaluación del Modelo')
    st.write('Error Cuadrático Medio (MSE):', mse)

    # Visualizar el gráfico con predicciones
    st.subheader('Predicciones del Modelo')
    results = pd.DataFrame({'Actual': y_actual, 'Predicción': y_pred})
    st.line_chart(results)

    # Visualización del gráfico con datos generados y resultados en un único DataFrame
    combined_results = data.copy()
    combined_results['Predicción'] = y_pred

    # Guardar datos generados en el estado de Streamlit
    st.session_state.generated_data = combined_results
    st.session_state.data_generated = True

    # Botón para descargar la información actual
    buffer = io.StringIO()
    combined_results.to_csv(buffer, index=False)
    st.download_button(
        label="Descargar Información Actual",
        data=buffer.getvalue(),
        file_name='resultados.csv',
        mime='text/csv'
    )

    # Mensaje de éxito para la descarga
    st.success("Datos generados listos para descargar.")

def save_generated_data():
    if st.session_state.generated_data is not None:
        if not os.path.exists('data/processed'):
            os.makedirs('data/processed')
        file_path = f'data/processed/resultados_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
        st.session_state.generated_data.to_csv(file_path, index=False)
        st.session_state.data_generated = False  # Ocultar el botón después de guardar
        st.success(f"Datos guardados exitosamente en {file_path}.")
    else:
        st.warning("No hay datos generados para guardar.")
