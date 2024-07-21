import streamlit as st
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
tab1, tab2 = st.tabs(["Generar Nueva Información", "Visualizar Datos Guardados"])

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

# Markdown Pie de página
st.markdown("""
---
Hecho por:
- Roxana Mabel Toala Torres
- María Fernanda Dueñas Murillo

Tecnologías Utilizadas:
- Streamlit (UI, Despliegue)
- Regresión Lineal [sklearn] (Modelo Machine Learning)
- Pandas (Manejo y Procesamiento de datos)
- Numpy (Genera datos aleatorios)
- IO (Flujo de entrada y salida)
- OS (Manejo del sistema operativo)
""")
