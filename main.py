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


# =========================
# Filtro por SEX (en barra lateral)
# =========================
if "df_sexo" not in st.session_state:
    st.session_state["df_sexo"] = None

with st.sidebar:
    st.subheader("Seleccione el sexo")
    if "SEX" not in datos_seleccionados.columns:
        st.warning("No se encontró la columna 'SEX' en los datos seleccionados.")
        st.session_state["df_sexo"] = datos_seleccionados.copy()
    else:
        # Asegurar tipo numérico 1/2
        sex_series = pd.to_numeric(datos_seleccionados["SEX"], errors="coerce").astype("Int64")

        # Opciones visibles y mapeo a códigos
        opciones_visibles = ["Ambos", "Hombre", "Mujer"]
        seleccion = st.multiselect(
            "Seleccione el sexo",
            options=opciones_visibles,
            default=["Ambos"],
            help="‘Hombre’ = 1, ‘Mujer’ = 2. ‘Ambos’ selecciona 1 y 2."
        )

        # Traducir selección visible -> códigos 1/2
        if (not seleccion) or ("Ambos" in seleccion):
            codigos = [1, 2]
        else:
            codigos = []
            if "Hombre" in seleccion:
                codigos.append(1)
            if "Mujer" in seleccion:
                codigos.append(2)
            # Si por alguna razón quedó vacío, usar ambos
            if not codigos:
                codigos = [1, 2]

        # Filtrar
        df_sexo = datos_seleccionados[sex_series.isin(codigos)].copy()
        st.session_state["df_sexo"] = df_sexo

# =========================
# Vista previa del filtrado por SEX
# =========================
if st.session_state["df_sexo"] is not None:
    st.subheader("Vista previa — Filtrado por sexo")
    c1, c2 = st.columns(2)
    c1.metric("Filas totales", len(datos_seleccionados))
    c2.metric("Filas tras SEX", len(st.session_state["df_sexo"]))
    st.dataframe(st.session_state["df_sexo"].head(30), use_container_width=True)


# =========================
# Filtro por RANGO DE EDAD (en barra lateral)
# =========================
# session_state necesarios
for key, default in [("age_min", None), ("age_max", None), ("df_filtrado", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# Definir el DataFrame base para filtrar por edad:
# - si ya existe df_sexo (filtrado por SEX), úsalo
# - si no, usa datos_seleccionados
base_df = st.session_state.get("df_sexo", None)
if base_df is None:
    base_df = datos_seleccionados.copy()

with st.sidebar:
    st.subheader("Seleccione rango de edad")
    if "AGE" not in base_df.columns:
        st.warning("No se encontró la columna 'AGE' en los datos.")
    else:
        # Asegurar tipo numérico
        age_series = pd.to_numeric(base_df["AGE"], errors="coerce")
        edades_validas = age_series.dropna()

        if edades_validas.empty:
            st.warning("La columna AGE no tiene valores numéricos válidos.")
            st.session_state["df_filtrado"] = base_df.iloc[0:0].copy()
        else:
            data_min = int(edades_validas.min())
            data_max = int(edades_validas.max())

            # Defaults seguros
            if st.session_state["age_min"] is None:
                st.session_state["age_min"] = data_min
            if st.session_state["age_max"] is None:
                st.session_state["age_max"] = data_max

            age_min = st.number_input(
                "Edad mínima",
                min_value=data_min,
                max_value=data_max,
                value=int(max(min(st.session_state["age_min"], data_max), data_min)),
                step=1,
                key="age_min",
            )
            age_max = st.number_input(
                "Edad máxima",
                min_value=data_min,
                max_value=data_max,
                value=int(max(min(st.session_state["age_max"], data_max), data_min)),
                step=1,
                key="age_max",
            )

            # Corregir si el usuario invierte los valores
            if st.session_state["age_min"] > st.session_state["age_max"]:
                st.warning("La edad mínima es mayor que la máxima. Se intercambian automáticamente.")
                st.session_state["age_min"], st.session_state["age_max"] = (
                    st.session_state["age_max"],
                    st.session_state["age_min"],
                )

            # Aplicar filtro
            mask = age_series.between(st.session_state["age_min"], st.session_state["age_max"], inclusive="both")
            st.session_state["df_filtrado"] = base_df[mask].copy()

# =========================
# Vista previa del filtrado por SEX + EDAD
# =========================
if st.session_state["df_filtrado"] is not None:
    st.subheader("Vista previa — Filtrado por SEX + EDAD")
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas base", len(base_df))
    c2.metric("Edad mínima", st.session_state["age_min"] if st.session_state["age_min"] is not None else "-")
    c3.metric("Edad máxima", st.session_state["age_max"] if st.session_state["age_max"] is not None else "-")
    st.dataframe(st.session_state["df_filtrado"].head(30), use_container_width=True)
    st.success(f"Filtrado final: {len(st.session_state['df_filtrado']):,} filas")

# =========================
# Filtro por COMORBILIDADES (en barra lateral)
# =========================
# Inicializar session_state
for key, default in [("comorb_selection", []), ("df_comorb", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# Base para el filtro de comorbilidades:
# - si ya existe el filtrado por SEX+EDAD úsalo, si no el por SEX, y si no, los datos seleccionados
df_base_comorb = st.session_state.get("df_filtrado")
if not isinstance(df_base_comorb, pd.DataFrame) or df_base_comorb.empty:
    df_base_comorb = st.session_state.get("df_sexo")
if not isinstance(df_base_comorb, pd.DataFrame) or df_base_comorb.empty:
    df_base_comorb = datos_seleccionados.copy()

# Mapeo: etiqueta legible -> nombre de columna (ya sin _18/_21)
comorb_map = {
    "Diabetes (C4)": "C4",
    "Hipertensión (C6)": "C6",
    "Cáncer (C12)": "C12",
    "Asma/Efisema (C19)": "C19",
    "Infarto / Ataque al corazón (C22A)": "C22A",
    "Embolia/Derrame/ICT (C26)": "C26",
    "Artritis/Reumatismo (C32)": "C32",
}

with st.sidebar:
    st.subheader("Seleccione comorbilidades")
    # Opciones disponibles según las columnas que existan
    opciones_visibles = [lbl for lbl, col in comorb_map.items() if col in df_base_comorb.columns]

    if not opciones_visibles:
        st.warning("No se encontraron columnas de comorbilidades esperadas (C4, C6, C12, C19, C22A, C26, C32).")
        st.session_state["df_comorb"] = df_base_comorb.copy()
    else:
        # Agregamos la opción especial
        opciones_visibles_con_none = ["Sin comorbilidades"] + opciones_visibles

        seleccion = st.multiselect(
            "Comorbilidades (1 = Sí, 2/0 = No).",
            options=opciones_visibles_con_none,
            default=[],
            help=(
                "• ‘Sin comorbilidades’: conserva filas con TODAS las comorbilidades en 2/0.\n"
                "• Si seleccionas una o más comorbilidades: conserva filas con 1 en las seleccionadas y 2/0 en las demás."
            )
        )
        st.session_state["comorb_selection"] = seleccion

        # Preparar dataframe de trabajo y asegurar numérico 0/1/2
        df_work = df_base_comorb.copy()
        comorb_cols_presentes = [comorb_map[lbl] for lbl in opciones_visibles]  # columnas reales presentes
        for c in comorb_cols_presentes:
            df_work[c] = pd.to_numeric(df_work[c], errors="coerce").fillna(0)

        # Conjunto de valores que consideramos "No": soporta 0 o 2
        NO_SET = {0, 2}
        YES_VAL = 1

        if not seleccion:
            # Sin selección → no filtrar por comorbilidades
            df_out = df_work.copy()

        elif "Sin comorbilidades" in seleccion:
            # Si el usuario mezcla "Sin comorbilidades" con otras, damos prioridad a "Sin comorbilidades"
            if len(seleccion) > 1:
                st.info("Se seleccionó 'Sin comorbilidades'. Se ignorarán otras selecciones para este filtro.")
            # Todas las comorbilidades deben estar en 2/0
            mask_all_none = df_work[comorb_cols_presentes].isin(NO_SET).all(axis=1)
            df_out = df_work[mask_all_none].copy()

        else:
            # Selección específica: las seleccionadas en 1, las NO seleccionadas en 2/0
            cols_sel = [comorb_map[lbl] for lbl in seleccion if comorb_map[lbl] in df_work.columns]
            cols_rest = [c for c in comorb_cols_presentes if c not in cols_sel]

            if not cols_sel:
                # Nada mapeable → no filtrar
                df_out = df_work.copy()
            else:
                mask_selected_yes = (df_work[cols_sel] == YES_VAL).all(axis=1)
                mask_rest_no = True
                if cols_rest:
                    mask_rest_no = df_work[cols_rest].isin(NO_SET).all(axis=1)
                df_out = df_work[mask_selected_yes & mask_rest_no].copy()

        st.session_state["df_comorb"] = df_out

# =========================
# Vista previa — Filtrado por SEX + EDAD + COMORBILIDADES
# =========================
if st.session_state["df_comorb"] is not None:
    st.subheader("Vista previa — Tras filtros (SEX + EDAD + COMORB)")
    # Seleccionar base segura para longitud
    base_df_for_len = st.session_state.get("df_filtrado")
    if not isinstance(base_df_for_len, pd.DataFrame) or base_df_for_len.empty:
        base_df_for_len = st.session_state.get("df_sexo")
    if not isinstance(base_df_for_len, pd.DataFrame) or base_df_for_len.empty:
        base_df_for_len = datos_seleccionados

    base_len = len(base_df_for_len)

    c1, c2 = st.columns(2)
    c1.metric("Filas base para comorb.", base_len)
    c2.metric("Filas tras comorb.", len(st.session_state["df_comorb"]))
    st.dataframe(st.session_state["df_comorb"].head(30), use_container_width=True)

    # Resumen rápido (cuenta de 1 en cada comorbilidad seleccionada)
    if st.session_state["comorb_selection"] and "Sin comorbilidades" not in st.session_state["comorb_selection"]:
        with st.expander("Resumen de comorbilidades seleccionadas (conteos de 1)"):
            df_show = st.session_state["df_comorb"]
            for lbl in st.session_state["comorb_selection"]:
                col = comorb_map[lbl]
                if col in df_show.columns:
                    cnt = int((pd.to_numeric(df_show[col], errors="coerce") == 1).sum())
                    st.write(f"- **{lbl}**: {cnt:,} casos con valor 1")
