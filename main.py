import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
import pydeck as pdk
from src.app import run_app, save_generated_data
from src.save import show_saved_data

# Inicializar el estado de Streamlit para los datos generados
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = None
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False

# Titulo
st.title('Optimización de la Gestión de Residuos con Machine Learning')

# Pestañas de la aplicación
tab1, tab2, tab3 = st.tabs(["Generar Nueva Información", "Visualizar Datos Guardados", "Análisis de K-Means"])

with tab1:
    # Control, para el selector de número a generar
    num_samples = st.slider("Seleccione el número de datos a generar", min_value=2, max_value=1000, value=100)

    # Botón para generar nueva información
    if st.button('Generar Nueva Información'):
        run_app(num_samples)

    # Botón para guardar los datos generados
    if st.session_state.data_generated:
        if st.button('Guardar Datos Generados'):
            save_generated_data()

with tab2:
    show_saved_data()

with tab3:
    st.subheader("Análisis de K-Means")

    # Cargar archivo CSV
    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

    if uploaded_file:
        # Leer el archivo CSV
        data = pd.read_csv(uploaded_file)
        st.write("Datos cargados:")
        st.dataframe(data)

        # Verificar columnas necesarias
        required_columns = ['latitud', 'longitud']
        if all(col in data.columns for col in required_columns):
            # Normalizar los datos
            features = data[required_columns].values
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Ejecutar K-means
            num_clusters = st.slider("Número de Clústeres", min_value=1, max_value=10, value=3)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)

            # Añadir resultados de clustering a los datos
            data['Cluster'] = clusters

            # Mostrar datos con clusters
            st.write("Datos con Clusters:")
            st.dataframe(data)

            # Graficar resultados
            st.subheader("Gráfico de Clústeres")
            fig, ax = plt.subplots()
            scatter = ax.scatter(data['latitud'], data['longitud'], c=data['Cluster'], cmap='viridis')
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
            plt.xlabel('Latitud')
            plt.ylabel('Longitud')
            st.pyplot(fig)

            # Crear un DataFrame con los colores de los clusters
            cluster_colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_clusters))
            data['color'] = data['Cluster'].apply(lambda x: mcolors.to_hex(cluster_colors[x]))
            data['color_rgb'] = data['color'].apply(lambda x: [int(255 * c) for c in mcolors.hex2color(x)])
            data['size'] = 50  # Incrementar el tamaño de los puntos

            # Mostrar el mapa en Streamlit con pydeck
            st.subheader("Mapa de los Datos y Clústeres")
            layer = pdk.Layer(
                'ScatterplotLayer',
                data,
                get_position='[longitud, latitud]',
                get_fill_color='color_rgb',
                get_radius='size',
                pickable=True,
                auto_highlight=True
            )

            tooltip = {
                "html": "<b>Latitud:</b> {latitud}<br><b>Longitud:</b> {longitud}<br><b>Cluster:</b> {Cluster}",
                "style": {"color": "white"}
            }

            view_state = pdk.ViewState(
                latitude=data['latitud'].mean(),
                longitude=data['longitud'].mean(),
                zoom=13,
                pitch=0,
            )

            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip
            )

            st.pydeck_chart(r)

        else:
            st.write(f"El archivo debe contener las columnas: {', '.join(required_columns)}")

# Markdown Pie de página
st.markdown("""
---
Realizado por:
- Maria Fernanda Dueñas Murillo
- Roxana Mabel Toala Torres
---
Tecnologías Utilizadas:
- Streamlit (UI, Despliegue)
- Regresión Lineal [sklearn] (Modelo Machine Learning)
- Pandas (Manejo y Procesamiento de datos)
- Numpy (Genera datos aleatorios)
- IO (Flujo de entrada y salida)
- OS (Manejo del sistema operativo)
""")