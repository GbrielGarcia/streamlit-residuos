import os
import pandas as pd
import streamlit as st

def show_saved_data():
    # Usar ruta absoluta
    processed_data_dir = os.path.join(os.path.dirname(__file__), '../data/processed')

    if not os.path.exists(processed_data_dir):
        st.write("El directorio 'data/processed' no existe.")
        return

    # Mostrar el encabezado de la sección
    st.subheader('Datos Guardados')

    # Obtener la lista de archivos CSV en el directorio 'data/processed'
    files = [f for f in os.listdir(processed_data_dir) if f.endswith('.csv')]

    # Verificar si hay archivos CSV disponibles
    if not files:
        st.write("No hay datos guardados.")
        return

    # Ordenar los archivos por fecha de modificación (más reciente primero)
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(processed_data_dir, f)), reverse=True)

    # Iterar sobre cada archivo CSV encontrado
    for file in files:
        # Leer el archivo CSV en un DataFrame de pandas
        data = pd.read_csv(os.path.join(processed_data_dir, file))

        # Mostrar el nombre del archivo y el DataFrame en la interfaz de usuario de Streamlit
        st.subheader(f'Datos del archivo: {file}')
        st.write(f'**{file}**')  # Mostrar el nombre del archivo en negrita
        st.dataframe(data)  # Mostrar el DataFrame en la interfaz

        # Mostrar un gráfico, si existe la columna 'Predicción'
        if 'Predicción' in data.columns:
            st.write('Resultados')
            st.line_chart(data[['residuos', 'Predicción']])
        else:
            st.write("La columna 'Predicción' no se encontró en los datos.")
