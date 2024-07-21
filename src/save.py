import os
import pandas as pd
import streamlit as st

def show_saved_data():
    # Mostrar el encabezado de la sección
    st.subheader('Datos Guardados')

    # Obtener la lista de archivos CSV en el directorio 'data/processed'
    files = [f for f in os.listdir('data/processed') if f.endswith('.csv')]

    # Verificar si hay archivos CSV disponibles
    if not files:
        st.write("No hay datos guardados.")
        return

    # Ordenar los archivos por fecha de modificación (más reciente primero)
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join('data/processed', f)), reverse=True)

    # Iterar sobre cada archivo CSV encontrado
    for file in files:
        # Leer el archivo CSV en un DataFrame de pandas
        data = pd.read_csv(os.path.join('data/processed', file))

        # Crear dos columnas en la interfaz de usuario de Streamlit
        col1, col2 = st.columns(2)

        # Mostrar el contenido del archivo CSV en la primera columna
        with col1:
            st.write(f'**{file}**')  # Mostrar el nombre del archivo en negrita
            st.dataframe(data)  # Mostrar el DataFrame en la interfaz

        # Mostrar un gráfico en la segunda columna, si existe la columna 'Predicción'
        with col2:
            if 'Predicción' in data.columns:
                st.write('Resultados')  # Título para la sección de gráficos
                st.line_chart(data[['residuos', 'Predicción']])  # Mostrar gráfico de residuos vs. predicción
            else:
                st.write("La columna 'Predicción' no se encontró en los datos.")  # Mensaje si no hay columna 'Predicción'
