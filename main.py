import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ENASEM — Carga simple", layout="wide")
st.title("ENASEM — Carga de archivo")

# --- Barra lateral: cargar archivo ---
with st.sidebar:
    st.header("Cargar datos")
    archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx"])

if archivo is None:
    st.info("Sube un archivo en la barra lateral para comenzar.")
    st.stop()

# --- Leer archivo ---
try:
    if archivo.name.lower().endswith(".csv"):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

# --- Normalizar nombres de columnas: quitar _18 o _21 al final ---
df.columns = [re.sub(r'_(18|21)$', '', c) for c in df.columns]

# --- Mostrar DataFrame ---
st.subheader("Datos cargados")
st.dataframe(df, use_container_width=True)

# --- Expander con información básica ---
with st.expander("Información del conjunto de datos"):
    filas, columnas = df.shape
    st.write(f"**Filas:** {filas:,}")
    st.write(f"**Columnas:** {columnas:,}")
    st.write("**Primeras columnas detectadas:**")
    st.code(", ".join(map(str, df.columns[:20])) + ("..." if len(df.columns) > 20 else ""))
