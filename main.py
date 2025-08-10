# app.py
import re
import pandas as pd
import streamlit as st

# =========================
# Infraestructura modular
# =========================
FEATURES = []  # lista de (orden, nombre, función)

def feature(name: str, order: int = 100):
    """Decorator para registrar secciones de UI sin reescribir lo previo."""
    def deco(fn):
        FEATURES.append((order, name, fn))
        return fn
    return deco

# =========================
# Utilidades reutilizables
# =========================
def normalize_year_suffix(columns):
    """Quita _18/_21 al final."""
    return [re.sub(r'_(18|21)$', '', c) for c in columns]

def load_csv_streamlit(file, usecols=None):
    df = pd.read_csv(file)
    df.columns = normalize_year_suffix(df.columns)
    if usecols:
        missing = set(usecols) - set(df.columns)
        if missing:
            st.warning(f"Columnas faltantes ignoradas: {sorted(missing)}")
        df = df[[c for c in usecols if c in df.columns]]
    return df

def combine_height(df, col1="C67_1", col2="C67_2", out="C67"):
    if col1 in df.columns and col2 in df.columns:
        df[out] = pd.to_numeric(df[col1], errors="coerce") + pd.to_numeric(df[col2], errors="coerce")/100
        return df.drop(columns=[col1, col2])
    return df

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def indiscernibility(attr, table: pd.DataFrame):
    u_ind = {}
    for i in table.index:
        key = tuple(table.loc[i, a] for a in attr)
        u_ind.setdefault(key, set()).add(i)
    return sorted(u_ind.values(), key=len, reverse=True)

# =========================
# App: cabecera + carga
# =========================
st.set_page_config(page_title="ENASEM modular", layout="wide")
st.title("ENASEM (2018/2021) — App modular")

COLUMNAS_BASE = [
    "AGE","SEX","C4","C6","C12","C19","C22A","C26","C32","C37",
    "C49_1","C49_2","C49_8","C64","C66","C67_1","C67_2","C68E","C68G","C68H",
    "C69A","C69B","C71A","C76","H1","H4","H5","H6","H8","H9","H10","H11","H12",
    "H13","H15A","H15B","H15D","H16A","H16D","H17A","H17D","H18A","H18D","H19A","H19D"
]

with st.sidebar:
    st.header("Carga de datos")
    up = st.file_uploader("CSV ENASEM 2018/2021", type=["csv"])
    if up and st.button("Procesar archivo"):
        df = load_csv_streamlit(up, usecols=COLUMNAS_BASE)
        df = combine_height(df, "C67_1", "C67_2", "C67")
        df.insert(0, "Indice", df.index)  # opcional
        st.session_state["df"] = df
        st.success("Archivo cargado y normalizado.")

df = st.session_state.get("df")
if df is not None:
    st.caption(f"Filas: {len(df):,} | Columnas: {len(df.columns)}")
    st.dataframe(df.head(50), use_container_width=True)
else:
    st.info("Carga un archivo en la barra lateral para comenzar.")
