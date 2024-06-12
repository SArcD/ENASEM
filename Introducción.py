import streamlit as st
import pandas as pd
import gdown

# Enlaces a los archivos CSV en Google Drive
file_urls = {
    "Archivo 1": "https://drive.google.com/uc?id=1OXrglgbqvwA1Oa2aMB5iLh9bMLJNo-uu",
    "Archivo 2": "https://drive.google.com/uc?id=1pn8-1nCeVb8piMgad-7foAI9z1nmfqsO"
}

# Función para descargar y cargar un archivo CSV desde Google Drive
def load_csv_from_drive(url):
    output = "temp.csv"
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# Interfaz de Streamlit
st.title("Carga y Visualización de Archivos CSV desde Google Drive")

# Crear una barra lateral para la selección de pestañas
st.sidebar.title("Navegación")
option = st.sidebar.selectbox("Seleccione una pestaña", ["Cargar Archivo", "Otra Funcionalidad"])

if option == "Cargar Archivo":
    # Menú desplegable para elegir el archivo
    selected_file = st.selectbox("Selecciona un archivo CSV", list(file_urls.keys()))

    if selected_file:
        # Cargar el archivo seleccionado
        data = load_csv_from_drive(file_urls[selected_file])
        
        st.write(f"Archivo seleccionado: {selected_file}")
        st.write(data)
        
        # Lista de verificación para seleccionar columnas
        selected_columns = st.multiselect("Selecciona las columnas para mostrar", data.columns.tolist())
        
        if selected_columns:
            # Crear dataframe reducido
            reduced_data = data[selected_columns]
            
            st.write("Dataframe reducido:")
            st.write(reduced_data)
            
            # Mostrar información del dataframe reducido
            num_rows, num_cols = reduced_data.shape
            st.write(f"Número de filas: {num_rows}")
            st.write(f"Número de columnas: {num_cols}")
            
            # Contar valores NaN por columna
            nan_counts = reduced_data.isna().sum().reset_index()
            nan_counts.columns = ["Clave", "Conteo"]
            
            st.write("Conteo de valores NaN por columna:")
            st.write(nan_counts)

elif option == "Otra Funcionalidad":
    st.write("Aquí puedes agregar otra funcionalidad.")
    # Agrega el código correspondiente para la otra funcionalidad
