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

# Menú desplegable para elegir el archivo
selected_file = st.selectbox("Selecciona un archivo CSV", list(file_urls.keys()))

if selected_file:
    # Cargar el archivo seleccionado
    data = load_csv_from_drive(file_urls[selected_file])
    
    st.write(f"Archivo seleccionado: {selected_file}")
    st.write(data)
