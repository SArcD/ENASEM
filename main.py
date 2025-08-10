import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ENASEM — Carga y preparación", layout="wide")
st.title("ENASEM — Cargar y preparar columnas (2018/2021)")

# -----------------------------------------
# Barra lateral: subir archivo
# -----------------------------------------
with st.sidebar:
    st.header("Cargar datos")
    archivo = st.file_uploader("Sube un CSV o Excel (ENASEM 2018/2021)", type=["csv", "xlsx"])

if archivo is None:
    st.info("Sube un archivo en la barra lateral para comenzar.")
    st.stop()

# -----------------------------------------
# Leer archivo (CSV o Excel)
# -----------------------------------------
try:
    if archivo.name.lower().endswith(".csv"):
        df = pd.read_csv(archivo)
    else:
        df = pd.read_excel(archivo)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

# -----------------------------------------
# Normalizar nombres de columnas
# - quitar espacios repetidos
# - quitar sufijos _18 o _21 al final
# -----------------------------------------
cols = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
cols = [re.sub(r"_(18|21)$", "", c) for c in cols]
df.columns = cols

# Eliminar posibles columnas "Unnamed: x" (índices exportados)
df = df.loc[:, ~df.columns.str.match(r"^Unnamed:\s*\d+")]

# -----------------------------------------
# Definir columnas deseadas (base, sin sufijo)
# -----------------------------------------
columnas_deseadas_base = [
    "AGE","SEX","C4","C6","C12","C19","C22A","C26","C32","C37",
    "C49_1","C49_2","C49_8","C64","C66","C67_1","C67_2","C68E","C68G","C68H",
    "C69A","C69B","C71A","C76","H1","H4","H5","H6","H8","H9","H10","H11","H12",
    "H13","H15A","H15B","H15D","H16A","H16D","H17A","H17D","H18A","H18D","H19A","H19D"
]

# Seleccionar solo las columnas deseadas que existan
presentes = [c for c in columnas_deseadas_base if c in df.columns]
faltantes = sorted(set(columnas_deseadas_base) - set(presentes))
if faltantes:
    st.warning("Columnas no encontradas (se omiten): " + ", ".join(faltantes))

datos_seleccionados = df[presentes].copy()

# -----------------------------------------
# Combinar estatura: C67_1 (m) + C67_2 (cm) → C67 (m)
# (solo si existen ambas; si falta alguna, se omite sin error)
# -----------------------------------------
if {"C67_1", "C67_2"}.issubset(datos_seleccionados.columns):
    datos_seleccionados["C67_1"] = pd.to_numeric(datos_seleccionados["C67_1"], errors="coerce")
    datos_seleccionados["C67_2"] = pd.to_numeric(datos_seleccionados["C67_2"], errors="coerce")
    datos_seleccionados["C67"] = datos_seleccionados["C67_1"] + (datos_seleccionados["C67_2"] / 100.0)
    datos_seleccionados = datos_seleccionados.drop(columns=["C67_1","C67_2"])

# Agregar columna Indice (índice actual)
datos_seleccionados["Indice"] = datos_seleccionados.index

# Reordenar columnas (usando las que existan)
columnas_finales = (
    ["Indice","AGE","SEX","C4","C6","C12","C19","C22A","C26","C32","C37",
     "C49_1","C49_2","C49_8","C64","C66","C67","C68E","C68G","C68H","C69A","C69B",
     "C71A","C76","H1","H4","H5","H6","H8","H9","H10","H11","H12",
     "H13","H15A","H15B","H15D","H16A","H16D","H17A","H17D","H18A","H18D","H19A","H19D"]
)
columnas_finales_presentes = [c for c in columnas_finales if c in datos_seleccionados.columns]
datos_seleccionados = datos_seleccionados[columnas_finales_presentes]

# -----------------------------------------
# Mostrar resultados
# -----------------------------------------
st.subheader("Datos seleccionados y normalizados")
st.dataframe(datos_seleccionados, use_container_width=True)

with st.expander("Información del conjunto de datos"):
    filas, columnas = datos_seleccionados.shape
    st.write(f"**Filas:** {filas:,}")
    st.write(f"**Columnas:** {columnas:,}")
    st.write("**Primeras columnas detectadas:**")
    st.code(", ".join(map(str, datos_seleccionados.columns[:20])) + ("..." if columnas > 20 else ""))
