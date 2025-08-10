
import streamlit as st
import pandas as pd

st.title("Cargar y visualizar un DataFrame")

# Botón para subir archivo
archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx"])

# Verificar si el usuario cargó un archivo
if archivo is not None:
    try:
        # Detectar si es CSV o Excel
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)

        st.success("Archivo cargado correctamente ✅")
        st.write("Vista previa del DataFrame:")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
