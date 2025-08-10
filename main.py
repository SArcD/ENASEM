import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ENASEM — Carga y filtros básicos", layout="wide")
st.title("ENASEM — Cargar y filtrar por sexo y edad")

# =========================
# 1) Cargar archivo y normalizar nombres de columnas
#    (quita solo el sufijo _18 o _21)
# =========================
archivo = st.file_uploader("Sube un CSV de ENASEM 2018 o 2021", type=["csv"])

if "df" not in st.session_state:
    st.session_state["df"] = None
if "df_sex" not in st.session_state:
    st.session_state["df_sex"] = None
if "df_filtered" not in st.session_state:
    st.session_state["df_filtered"] = None

if archivo is not None:
    try:
        df_raw = pd.read_csv(archivo)
        # Normaliza columnas quitando solo los sufijos _18 o _21
        df_raw.columns = [re.sub(r'_(18|21)$', '', c) for c in df_raw.columns]
        st.session_state["df"] = df_raw
        st.success("Archivo cargado y columnas normalizadas ✅")
        st.caption(f"Columnas detectadas: {', '.join(list(df_raw.columns)[:12])}" + ("..." if len(df_raw.columns) > 12 else ""))
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")

df = st.session_state["df"]

if df is None:
    st.info("Carga un archivo para continuar.")
    st.stop()

st.subheader("Vista previa (datos originales)")
st.dataframe(df.head(30), use_container_width=True)

# =========================
# 2) Selección de SEX (1=Hombre, 2=Mujer o Ambos)
# =========================
if "SEX" not in df.columns:
    st.error("No se encontró la columna 'SEX' (después de normalizar).")
    st.stop()

# Aseguramos que SEX sea numérico (1/2)
df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")

opcion_sex = st.radio(
    "Selecciona el sexo a analizar",
    options=["Ambos", "Hombre (1)", "Mujer (2)"],
    horizontal=True,
)

if opcion_sex == "Hombre (1)":
    df_sex = df[df["SEX"] == 1].copy()
elif opcion_sex == "Mujer (2)":
    df_sex = df[df["SEX"] == 2].copy()
else:
    df_sex = df.copy()

st.session_state["df_sex"] = df_sex

c1, c2 = st.columns(2)
c1.metric("Filas totales (archivo)", len(df))
c2.metric("Filas tras filtro de SEX", len(df_sex))

st.write("Vista previa tras filtrar por SEX:")
st.dataframe(df_sex.head(30), use_container_width=True)

# =========================
# 3) Selección de rango de edad (columna AGE)
# =========================
if "AGE" not in df_sex.columns:
    st.error("No se encontró la columna 'AGE' (después de normalizar).")
    st.stop()

# Aseguramos que AGE sea numérico
df_sex["AGE"] = pd.to_numeric(df_sex["AGE"], errors="coerce")

if df_sex["AGE"].notna().sum() == 0:
    st.warning("La columna AGE no tiene valores numéricos válidos.")
    st.stop()

edad_min = int(df_sex["AGE"].min(skipna=True))
edad_max = int(df_sex["AGE"].max(skipna=True))

st.subheader("Filtro por rango de edad")
rango = st.slider(
    "Selecciona el rango de edad (años)",
    min_value=edad_min,
    max_value=edad_max,
    value=(edad_min, edad_max),
    step=1,
)
edad_inf, edad_sup = rango

df_filtrado = df_sex[df_sex["AGE"].between(edad_inf, edad_sup, inclusive="both")].copy()
st.session_state["df_filtered"] = df_filtrado

c3, c4 = st.columns(2)
c3.metric("Edad mínima", edad_inf)
c4.metric("Edad máxima", edad_sup)

st.write("Vista previa tras filtro de SEX + EDAD:")
st.dataframe(df_filtrado.head(30), use_container_width=True)

st.success(f"Filtrado final: {len(df_filtrado):,} filas")
