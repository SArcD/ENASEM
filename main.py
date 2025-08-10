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

    # Mostrar filtros solo si hay DataFrame
    if st.session_state.df is not None:
        df = st.session_state.df.copy()

        # ====== Selector de SEX ======
        if "SEX" not in df.columns:
            st.error("No se encontró la columna 'SEX' tras normalizar.")
        else:
            df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")

            st.session_state.sex_option = st.selectbox(
                "Sexo",
                options=["Ambos", "Hombre (1)", "Mujer (2)"],
                index=["Ambos", "Hombre (1)", "Mujer (2)"].index(st.session_state.sex_option)
                if st.session_state.sex_option in ["Ambos", "Hombre (1)", "Mujer (2)"] else 0
            )

            if st.session_state.sex_option == "Hombre (1)":
                df_sex = df[df["SEX"] == 1].copy()
            elif st.session_state.sex_option == "Mujer (2)":
                df_sex = df[df["SEX"] == 2].copy()
            else:
                df_sex = df.copy()

            st.session_state.df_sex = df_sex

        # ====== Rango de EDAD por inputs ======
        if st.session_state.df_sex is not None:
            df_sex = st.session_state.df_sex

            if "AGE" not in df_sex.columns:
                st.error("No se encontró la columna 'AGE' tras normalizar.")
            else:
                df_sex = df_sex.copy()
                df_sex["AGE"] = pd.to_numeric(df_sex["AGE"], errors="coerce")
                if df_sex["AGE"].notna().sum() == 0:
                    st.error("La columna AGE no contiene valores numéricos válidos.")
                else:
                    data_min = int(df_sex["AGE"].min(skipna=True))
                    data_max = int(df_sex["AGE"].max(skipna=True))

                    # Establecer defaults si aún no existen
                    if st.session_state.age_min_input is None:
                        st.session_state.age_min_input = data_min
                    if st.session_state.age_max_input is None:
                        st.session_state.age_max_input = data_max

                    st.markdown("**Rango de edad (años)**")
                    age_min_input = st.number_input(
                        "Edad mínima",
                        min_value=data_min,
                        max_value=data_max,
                        value=int(st.session_state.age_min_input),
                        step=1,
                        key="age_min_input",
                    )
                    age_max_input = st.number_input(
                        "Edad máxima",
                        min_value=data_min,
                        max_value=data_max,
                        value=int(st.session_state.age_max_input),
                        step=1,
                        key="age_max_input",
                    )

                    # Validación simple: si min > max, intercambiamos
                    if st.session_state.age_min_input > st.session_state.age_max_input:
                        st.warning("La edad mínima es mayor que la máxima. Se intercambian automáticamente.")
                        min_tmp = st.session_state.age_max_input
                        max_tmp = st.session_state.age_min_input
                        st.session_state.age_min_input = min_tmp
                        st.session_state.age_max_input = max_tmp

                    # Aplicar filtro final
                    df_filtered = df_sex[
                        df_sex["AGE"].between(st.session_state.age_min_input,
                                              st.session_state.age_max_input,
                                              inclusive="both")
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
