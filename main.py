import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ENASEM — Filtros en barra lateral", layout="wide")
st.title("ENASEM — Carga y filtros (sexo y edad)")

# =========================
# Inicializar session_state
# =========================
for key, default in [
    ("df", None),
    ("df_sex", None),
    ("df_filtered", None),
    ("sex_option", "Ambos"),
    ("age_min_input", None),
    ("age_max_input", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =========================
# Controles en barra lateral
# =========================
with st.sidebar:
    st.header("Controles")
    archivo = st.file_uploader("Sube CSV ENASEM 2018/2021", type=["csv"])

    if archivo is not None:
        try:
            df_raw = pd.read_csv(archivo)
            # Normaliza columnas quitando sufijos _18 o _21
            df_raw.columns = [re.sub(r'_(18|21)$', '', c) for c in df_raw.columns]
            st.session_state.df = df_raw
            st.success("Archivo cargado y columnas normalizadas ✅")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

    # ====== Selector de SEX (robusto) ======
    if st.session_state.df is not None:
        df = st.session_state.df.copy()

        if "SEX" not in df.columns:
            st.error("No se encontró la columna 'SEX' tras normalizar.")
            st.session_state.df_sex = None
            st.session_state.df_filtered = None
        else:
            # Convertir a numérico estricto y quedarnos solo con 1/2
            df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce").astype("Int64")
            df = df[df["SEX"].isin([1, 2])].copy()

            if df.empty:
                st.error("No hay filas con SEX igual a 1 (hombre) o 2 (mujer).")
                st.session_state.df_sex = None
                st.session_state.df_filtered = None
            else:
                # Conteos rápidos por sexo
                hombres = int((df["SEX"] == 1).sum())
                mujeres = int((df["SEX"] == 2).sum())
                st.caption(f"Distribución SEX — Hombres: {hombres:,} | Mujeres: {mujeres:,} | Total: {len(df):,}")

                sex_choice = st.selectbox(
                    "Sexo",
                    options=["Ambos", "Hombre (1)", "Mujer (2)"],
                    index=["Ambos", "Hombre (1)", "Mujer (2)"].index(
                        st.session_state.get("sex_option", "Ambos")
                    ),
                    key="sex_option",
                )

                if sex_choice == "Hombre (1)":
                    df_sex = df[df["SEX"] == 1].copy()
                elif sex_choice == "Mujer (2)":
                    df_sex = df[df["SEX"] == 2].copy()
                else:
                    df_sex = df.copy()

                st.session_state.df_sex = df_sex

                # ====== Rango de EDAD por inputs (solo si hay filas) ======
                if df_sex.empty:
                    st.warning("No hay filas tras filtrar por SEX.")
                    st.session_state.df_filtered = df_sex
                else:
                    if "AGE" not in df_sex.columns:
                        st.error("No se encontró la columna 'AGE' tras normalizar.")
                        st.session_state.df_filtered = df_sex.iloc[0:0].copy()
                    else:
                        df_sex = df_sex.copy()
                        df_sex["AGE"] = pd.to_numeric(df_sex["AGE"], errors="coerce")

                        # Tomamos solo edades válidas
                        edades_validas = df_sex["AGE"].dropna()
                        if edades_validas.empty:
                            st.warning("La columna AGE no tiene valores numéricos válidos tras el filtro de SEX.")
                            st.session_state.df_filtered = df_sex.iloc[0:0].copy()
                        else:
                            data_min = int(edades_validas.min())
                            data_max = int(edades_validas.max())

                            # Defaults seguros
                            if st.session_state.get("age_min_input") is None:
                                st.session_state["age_min_input"] = data_min
                            if st.session_state.get("age_max_input") is None:
                                st.session_state["age_max_input"] = data_max

                            st.markdown("**Rango de edad (años)**")
                            age_min = st.number_input(
                                "Edad mínima",
                                min_value=data_min,
                                max_value=data_max,
                                value=min(int(st.session_state["age_min_input"]), data_max),
                                step=1,
                                key="age_min_input",
                            )
                            age_max = st.number_input(
                                "Edad máxima",
                                min_value=data_min,
                                max_value=data_max,
                                value=max(int(st.session_state["age_max_input"]), data_min),
                                step=1,
                                key="age_max_input",
                            )

                            # Corregir si el usuario invierte los valores
                            if age_min > age_max:
                                st.warning("La edad mínima es mayor que la máxima. Se intercambian automáticamente.")
                                age_min, age_max = age_max, age_min
                                st.session_state["age_min_input"] = age_min
                                st.session_state["age_max_input"] = age_max

                            df_filtered = df_sex[
                                df_sex["AGE"].between(age_min, age_max, inclusive="both")
                            ].copy()
                            st.session_state.df_filtered = df_filtered

# =========================
# Contenido principal
# =========================
df = st.session_state.df
if df is None:
    st.info("Carga un archivo en la barra lateral para comenzar.")
    st.stop()

# Métricas y vistas
c1, c2, c3 = st.columns(3)
c1.metric("Filas (original)", len(df))
c2.metric("Tras SEX", len(st.session_state.df_sex) if st.session_state.df_sex is not None else 0)
c3.metric("Tras SEX + EDAD", len(st.session_state.df_filtered) if st.session_state.df_filtered is not None else 0)

st.subheader("Vista previa — Datos originales")
st.dataframe(df.head(30), use_container_width=True)

if st.session_state.df_sex is not None:
    st.subheader("Vista previa — Tras filtro de SEX")
    st.dataframe(st.session_state.df_sex.head(30), use_container_width=True)

if st.session_state.df_filtered is not None:
    st.subheader("Vista previa — Tras filtro de SEX + EDAD")
    st.dataframe(st.session_state.df_filtered.head(30), use_container_width=True)
    st.success(f"Filtrado final: {len(st.session_state.df_filtered):,} filas")
