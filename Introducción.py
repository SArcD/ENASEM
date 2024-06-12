

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
st.title("Carga de Archivos CSV desde Google Drive")

# Crear pestañas para cada archivo
tabs = st.tabs(["Archivo 1", "Archivo 2"])

# Descargar y mostrar el contenido de cada archivo en su respectiva pestaña
for i, tab in enumerate(tabs):
    with tab:
        file_name = list(file_urls.keys())[i]
        file_url = file_urls[file_name]
        
        # Cargar el archivo seleccionado
        data = load_csv_from_drive(file_url)
        
        st.write(f"Archivo seleccionado: {file_name}")
        st.write(data)

