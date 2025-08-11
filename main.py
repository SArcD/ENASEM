import re
import pandas as pd
import streamlit as st
def determinar_color(valores):
    count_ones = sum(1 for v in valores if pd.to_numeric(v, errors="coerce") == 1)
    if count_ones == 0:   return 'blue'
    if 1 <= count_ones < 3: return 'green'
    if count_ones == 3:   return 'yellow'
    if 4 <= count_ones < 5: return 'orange'
    return 'red'

# --- Utilidades comunes para particiones/metricas (definir si no existen) ---
if "blocks_to_labels" not in globals():
    import numpy as np
    import pandas as pd

    def blocks_to_labels(blocks, universo):
        """Convierte una lista de sets (bloques) a un vector de etiquetas siguiendo el orden de 'universo'."""
        lbl = {}
        for k, S in enumerate(blocks):
            for idx in S:
                lbl[idx] = k
        return np.array([lbl[i] for i in universo])

    def contingency_from_labels(y_true, y_pred):
        """Matriz de contingencia n_ij entre dos particiones dadas por etiquetas."""
        s1 = pd.Series(y_true).astype("category")
        s2 = pd.Series(y_pred).astype("category")
        return pd.crosstab(s1, s2).values

    def pairs_same(counts):
        """Suma de C(n,2) por cada tamaño en 'counts'."""
        counts = np.asarray(counts, dtype=np.int64)
        return (counts * (counts - 1) // 2).sum()

    def ari_from_contingency(C):
        """ARI (Adjusted Rand Index) a partir de la matriz de contingencia."""
        n = C.sum()
        a = C.sum(axis=1)
        b = C.sum(axis=0)
        sum_comb = (C * (C - 1) // 2).sum()
        sum_a = (a * (a - 1) // 2).sum()
        sum_b = (b * (b - 1) // 2).sum()
        T = n * (n - 1) // 2
        expected = (sum_a * sum_b) / T if T else 0.0
        max_index = 0.5 * (sum_a + sum_b)
        denom = max_index - expected
        return float((sum_comb - expected) / denom) if denom != 0 else 1.0

    def nmi_from_contingency(C):
        """NMI (Normalized Mutual Information) a partir de la matriz de contingencia."""
        n = C.sum()
        if n == 0:
            return 1.0
        a = C.sum(axis=1)
        b = C.sum(axis=0)
        I = 0.0
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                nij = C[i, j]
                if nij > 0:
                    I += (nij / n) * np.log((nij * n) / (a[i] * b[j]))
        p = a / n
        q = b / n
        Hu = -np.sum([pi * np.log(pi) for pi in p if pi > 0])
        Hv = -np.sum([qj * np.log(qj) for qj in q if qj > 0])
        denom = np.sqrt(Hu * Hv)
        return float(I / denom) if denom > 0 else 1.0

    def preservation_metrics_from_contingency(C):
        """% preservación de 'iguales' y 'distintos' entre partición original y reducida."""
        n = C.sum()
        T = n * (n - 1) // 2 if n else 0
        a = C.sum(axis=1)
        b = C.sum(axis=0)
        same_orig = pairs_same(a)
        same_red  = pairs_same(b)
        same_both = (C * (C - 1) // 2).sum()
        pres_same = same_both / same_orig if same_orig > 0 else 1.0
        diff_orig = T - same_orig
        diff_to_same = same_red - same_both
        pres_diff = (diff_orig - diff_to_same) / diff_orig if diff_orig > 0 else 1.0
        return pres_same, pres_diff

import streamlit as st

LOGO_URL = "https://raw.githubusercontent.com/SArcD/ENASEM/main/logo_radar_pie_exact_v5.png"

col_logo, col_title = st.columns([1, 3], gap="small")

with col_logo:
    st.image(LOGO_URL, use_container_width=True)

with col_title:
    st.markdown(
        """
        <div style="display:inline-block; text-align:left;">
            <h1 style="margin-bottom:6px; display:inline-block;">
                RS²: Rough Sets para Riesgo de Sarcopenia
            </h1>
            <div style="font-size:1.05rem; color:#444; max-width: 40ch;">
                Análisis y visualización con conjuntos rugosos para perfilar el riesgo de sarcopenia.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )




st.set_page_config(page_title="ENASEM — Carga y preparación", layout="wide")
st.title("ENASEM — Predictor de riesgo de sarcopenia (Encuestas 2018/2021)")
st.markdown("""
<style>
/* Justificar todo el texto de párrafos, listas y tablas */
p, li, td { text-align: justify; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
En esta sección se usan datos de la **Encuesta Nacional sobre Envejecimiento en México**.

1. **Cargue el archivo** del año que desee analizar desde el botón en la barra lateral  
   (ejemplo: `conjunto_de_datos_sect_a_c_d_f_e_pc_h_i_enasem_20XX.csv`).  
2. Puede **seleccionar el sexo** de los participantes o incluir a ambos.  
3. Use las **casillas de la barra lateral** para definir rangos de edad específicos.  
4. En comorbilidades:  
   - **Sin comorbilidades**: ignora cualquier otra seleccionada.  
   - **AND**: incluye solo a quienes tienen todas las comorbilidades seleccionadas.  
   - **OR**: incluye a quienes tienen al menos una de las seleccionadas.  
5. Para iniciar el estudio, indique:  
   - Número de conjuntos que desea crear.  
   - Número mínimo de participantes que debe tener un conjunto para que se considere en el estudio (esto evita estudiar casos poco representativos).  
   Luego presione el botón **Calcular indiscernibilidad**.
""")


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
st.subheader("Datos cargados para el análisis")
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
    st.subheader("Seleccione el rango de edad (puede teclear los valores dentro de los recuadros")
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

#with st.sidebar:
#    st.subheader("Seleccione comorbilidades del grupo de pacientes a estudiar")

#    # Opciones visibles (labels) cuya columna real existe en el DF
#    opciones_visibles = [lbl for lbl, col in comorb_map.items() if col in df_base_comorb.columns]

#    if not opciones_visibles:
#        st.warning("No se encontraron columnas de comorbilidades esperadas (C4, C6, C12, C19, C22A, C26, C32).")
#        st.session_state["df_comorb"] = df_base_comorb.copy()
#    else:
#        # Lógica AND/OR y restricción de no seleccionadas
#        modo = st.radio("Lógica entre las seleccionadas", ["Todas (AND)", "Cualquiera (OR)"],
#                        index=0, horizontal=True)
#        exigir_no = st.checkbox("Exigir que las NO seleccionadas estén en 0/2", value=True,
#                                help="Si está activado, las comorbilidades no seleccionadas deben ser 0 o 2.")

#        opciones_visibles_con_none = ["Sin comorbilidades"] + opciones_visibles
#        seleccion = st.multiselect(
#            "Comorbilidades (1 = Sí, 2/0 = No).",
#            options=opciones_visibles_con_none,
#            default=[],
#            help=("• ‘Sin comorbilidades’: conserva filas con TODAS las comorbilidades en 2/0.\n"
#                  "• Si seleccionas comorbilidades: puedes combinar con lógica AND/OR, "
#                  "y decidir si las NO seleccionadas deben estar en 0/2.")
#        )
#        st.session_state["comorb_selection"] = seleccion

#        # Preparar DF y asegurar numérico 0/1/2
#        df_work = df_base_comorb.copy()
#        comorb_cols_presentes = [comorb_map[lbl] for lbl in opciones_visibles]  # columnas reales presentes
#        for c in comorb_cols_presentes:
#            df_work[c] = pd.to_numeric(df_work[c], errors="coerce").fillna(0)

#        NO_SET = {0, 2}
#        YES_VAL = 1

#        if not seleccion:
#            df_out = df_work.copy()

#        elif "Sin comorbilidades" in seleccion:
#            if len(seleccion) > 1:
#                st.info("Se seleccionó 'Sin comorbilidades'. Se ignorarán otras selecciones para este filtro.")
#            mask_all_none = df_work[comorb_cols_presentes].isin(NO_SET).all(axis=1)
#            df_out = df_work[mask_all_none].copy()

#        else:
#            cols_sel = [comorb_map[lbl] for lbl in seleccion if comorb_map[lbl] in df_work.columns]
#            cols_rest = [c for c in comorb_cols_presentes if c not in cols_sel]

#            if not cols_sel:
#                df_out = df_work.copy()
#            else:
#                # AND vs OR
#                if modo.startswith("Todas"):
#                    mask_sel = (df_work[cols_sel] == YES_VAL).all(axis=1)
#                else:
#                    mask_sel = (df_work[cols_sel] == YES_VAL).any(axis=1)

#                if exigir_no and cols_rest:
#                    mask_rest = df_work[cols_rest].isin(NO_SET).all(axis=1)
#                    mask = mask_sel & mask_rest
#                else:
#                    mask = mask_sel

#                df_out = df_work[mask].copy()

#        st.session_state["df_comorb"] = df_out
#        st.caption(f"Filas tras filtro de comorbilidades: {len(df_out):,}")


with st.sidebar:
    st.subheader("Seleccione comorbilidades del grupo de pacientes a estudiar")

    opciones_visibles = [lbl for lbl, col in comorb_map.items() if col in df_base_comorb.columns]

    if not opciones_visibles:
        st.warning("No se encontraron columnas de comorbilidades esperadas (C4, C6, C12, C19, C22A, C26, C32).")
        df_out = df_base_comorb.copy()
        # Agregamos columna comorbilidad (todo 'Desconocido' si no hay columnas)
        df_out["comorbilidad"] = "Desconocido"
        st.session_state["df_comorb"] = df_out
        st.caption(f"Filas tras filtro de comorbilidades: {len(df_out):,}")
    else:
        # NUEVO: Modo de estudio
        modo_estudio = st.radio(
            "Modo de estudio",
            ["Filtrar (según selección)", "Comparar (Sin vs Con)", "Todos (un solo grupo)"],
            index=0, horizontal=False
        )

        # Configuración del filtro (solo relevante para 'Filtrar' y 'Comparar')
        modo = st.radio("Lógica entre las seleccionadas", ["Todas (AND)", "Cualquiera (OR)"],
                        index=0, horizontal=True)
        exigir_no = st.checkbox(
            "Exigir que las NO seleccionadas estén en 0/2",
            value=True,
            help="Si está activado, las comorbilidades no seleccionadas deben ser 0 o 2."
        )

        opciones_visibles_con_none = ["Sin comorbilidades"] + opciones_visibles
        seleccion = st.multiselect(
            "Comorbilidades (1 = Sí, 2/0 = No).",
            options=opciones_visibles_con_none,
            default=[],
            help=("• ‘Sin comorbilidades’: conserva filas con TODAS las comorbilidades en 2/0.\n"
                  "• Si seleccionas comorbilidades: combina con lógica AND/OR y decide si las NO seleccionadas deben estar en 0/2.")
        )
        st.session_state["comorb_selection"] = seleccion

        # Preparar DF y asegurar numérico 0/1/2
        df_work = df_base_comorb.copy()
        comorb_cols_presentes = [comorb_map[lbl] for lbl in opciones_visibles]
        for c in comorb_cols_presentes:
            df_work[c] = pd.to_numeric(df_work[c], errors="coerce").fillna(0)

        NO_SET = {0, 2}
        YES_VAL = 1

        # Máscaras base
        mask_none = df_work[comorb_cols_presentes].isin(NO_SET).all(axis=1)   # Sin comorbilidades
        mask_any  = (df_work[comorb_cols_presentes] == YES_VAL).any(axis=1)   # Al menos una

        # Auxiliar para "con comorbilidades" según selección
        def build_with_group():
            if not seleccion or (len(seleccion) == 1 and "Sin comorbilidades" in seleccion):
                return df_work[mask_any].copy()

            cols_sel = [comorb_map[lbl] for lbl in seleccion if lbl in comorb_map and comorb_map[lbl] in df_work.columns]
            cols_rest = [c for c in comorb_cols_presentes if c not in cols_sel]

            if not cols_sel:
                return df_work[mask_any].copy()

            if modo.startswith("Todas"):
                mask_sel = (df_work[cols_sel] == YES_VAL).all(axis=1)  # AND
            else:
                mask_sel = (df_work[cols_sel] == YES_VAL).any(axis=1)  # OR

            if exigir_no and cols_rest:
                mask_rest = df_work[cols_rest].isin(NO_SET).all(axis=1)
                mask = mask_sel & mask_rest
            else:
                mask = mask_sel

            return df_work[mask].copy()

        # ----- LÓGICA POR MODO -----
        if modo_estudio == "Todos (un solo grupo)":
            # No filtramos filas: estudiamos todo el universo como un solo grupo
            df_out = df_work.copy()
            # Etiquetamos por conveniencia (no obliga a agrupar)
            df_out["comorbilidad"] = np.where(mask_none, "Sin comorbilidades", "Con comorbilidades")

            st.session_state["df_comorb"] = df_out
            st.caption(f"Filas (todos los pacientes): {len(df_out):,}")

        elif modo_estudio == "Comparar (Sin vs Con)":
            df_none = df_work[mask_none].copy()
            df_with = build_with_group()

            df_none["comorbilidad"] = "Sin comorbilidades"
            df_with["comorbilidad"] = "Con comorbilidades"

            df_both = pd.concat([df_none, df_with], axis=0, ignore_index=True)

            st.session_state["df_comorb"] = df_both
            st.caption(
                f"Sin comorbilidades: {len(df_none):,} | Con comorbilidades: {len(df_with):,} | Total: {len(df_both):,}"
            )

        else:  # "Filtrar (según selección)"
            if not seleccion:
                df_out = df_work.copy()
                # Etiqueta basada en presencia/ausencia
                df_out["comorbilidad"] = np.where(
                    mask_none, "Sin comorbilidades", "Con comorbilidades"
                )

            elif "Sin comorbilidades" in seleccion:
                if len(seleccion) > 1:
                    st.info("Se seleccionó 'Sin comorbilidades'. Se ignorarán otras selecciones para este filtro.")
                df_out = df_work[mask_none].copy()
                df_out["comorbilidad"] = "Sin comorbilidades"

            else:
                cols_sel = [comorb_map[lbl] for lbl in seleccion if comorb_map[lbl] in df_work.columns]
                cols_rest = [c for c in comorb_cols_presentes if c not in cols_sel]

                if not cols_sel:
                    df_out = df_work.copy()
                    df_out["comorbilidad"] = np.where(mask_none, "Sin comorbilidades", "Con comorbilidades")
                else:
                    if modo.startswith("Todas"):
                        mask_sel = (df_work[cols_sel] == YES_VAL).all(axis=1)
                    else:
                        mask_sel = (df_work[cols_sel] == YES_VAL).any(axis=1)

                    if exigir_no and cols_rest:
                        mask_rest = df_work[cols_rest].isin(NO_SET).all(axis=1)
                        mask = mask_sel & mask_rest
                    else:
                        mask = mask_sel

                    df_out = df_work[mask].copy()
                    df_out["comorbilidad"] = "Con comorbilidades"

            st.session_state["df_comorb"] = df_out
            st.caption(f"Filas tras filtro de comorbilidades: {len(df_out):,}")







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



# HAsta aqui el filtrado

# =========================
# Indiscernibilidad + resumen + pastel + radar
# =========================
# Indiscernibilidad + resumen + pastel + radar (con exclusión de NaN)
# =========================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from math import log1p
import re

# --- Funciones ---
def indiscernibility(attr, table: pd.DataFrame):
    """
    Forma clases de indiscernibilidad usando tuplas (sin colisiones).
    (Aquí ya NO habrá NaN porque filtramos antes con dropna).
    """
    u_ind = {}
    for i in table.index:
        key = tuple(table.loc[i, a] for a in attr)
        u_ind.setdefault(key, set()).add(i)
    return sorted(u_ind.values(), key=len, reverse=True)

def lower_approximation(R, X):
    l_approx = set()
    for x in X:
        for r in R:
            if r.issubset(x):
                l_approx.update(r)
    return l_approx

def upper_approximation(R, X):
    u_approx = set()
    for x in X:
        for r in R:
            if r.intersection(x):
                u_approx.update(r)
    return u_approx

# --- DataFrame base: usa el más filtrado disponible ---
df_base_ind = st.session_state.get("df_comorb")
if not isinstance(df_base_ind, pd.DataFrame) or df_base_ind.empty:
    df_base_ind = st.session_state.get("df_filtrado")
if not isinstance(df_base_ind, pd.DataFrame) or df_base_ind.empty:
    df_base_ind = st.session_state.get("df_sexo")
if not isinstance(df_base_ind, pd.DataFrame) or df_base_ind.empty:
    df_base_ind = datos_seleccionados.copy()

# --- Asegurar columna de índice visible ---
if isinstance(df_base_ind, pd.DataFrame):
    if "Indice" not in df_base_ind.columns:
        df_base_ind = df_base_ind.copy()
        df_base_ind["Indice"] = df_base_ind.index

# --- Resolver columnas ADL sin depender de _18/_21 (incluye C37) ---
ADL_BASE = [
    "H1","H4","H5","H6","H8","H9","H10","H11","H12",
    "H13","H15A","H15B","H15D","H16A","H16D",
    "H17A","H17D","H18A","H18D","H19A","H19D","C37"
]

def match_col(base_name: str, cols) -> str | None:
    candidates = [base_name, f"{base_name}_18", f"{base_name}_21"]
    for cand in candidates:
        if cand in cols:
            return cand
    return None

cols_real, cols_norm = [], []
for base in ADL_BASE:
    c = match_col(base, df_base_ind.columns)
    if c is not None:
        cols_real.append(c)
        cols_norm.append(base)

if not cols_real:
    st.warning("No se encontraron columnas de ADL esperadas (H*).")
    st.stop()

# --- DF reducido: solo Indice + ADL (mantener NaN) ---
df_ind_min = df_base_ind[["Indice"] + cols_real].copy()
df_ind_min.rename(columns={r: n for r, n in zip(cols_real, cols_norm)}, inplace=True)
for c in cols_norm:
    df_ind_min[c] = pd.to_numeric(df_ind_min[c], errors="coerce").astype("float32")

# --- Referencias en sesión ---
st.session_state["ind_df_full_ref"] = df_base_ind          # DF completo (con Indice)
st.session_state["ind_df_reducido"] = df_ind_min           # Solo Indice + ADL
st.session_state["ind_adl_cols"]   = cols_norm             # Nombres normalizados ADL

# --- Controles en barra lateral ---
with st.sidebar:
    st.subheader("Indiscernibilidad")
    adl_opts = st.session_state.get("ind_adl_cols", [])
    sugeridas = [c for c in ["C37","H11","H15A","H5","H6"] if c in adl_opts]
    cols_attrs = st.multiselect(
        "Atributos (ADL) para agrupar",
        options=adl_opts,
        default=sugeridas or adl_opts[:5],
        help="Se forman clases con la combinación exacta de estas ADL."
    )
    min_size_for_pie = st.number_input(
        "Tamaño mínimo de clase para incluir en el pastel",
        min_value=2, max_value=100000, value=30, step=1
    )
    top_n_radar = st.number_input(
        "N conjuntos más numerosos para radar",
        min_value=1, max_value=100, value=15, step=1
    )
    # ✅ guarda el valor para re-render fuera del botón
    st.session_state["top_n_radar_value"] = int(top_n_radar)
    generar = st.button("Calcular indiscernibilidad")

# --- Cálculo ---
if generar:
    if not cols_attrs:
        st.warning("Selecciona al menos una ADL para indiscernibilidad.")
    else:
        src = st.session_state.get("ind_df_reducido")
        if not isinstance(src, pd.DataFrame) or src.empty:
            st.error("No hay DF reducido en sesión. Revisa la sección de 'Indice + ADL'.")
            st.stop()

        # Índice por 'Indice'
        df_ind = src.copy()
        if "Indice" in df_ind.columns:
            df_ind.set_index("Indice", inplace=True)
        df_ind.index.name = "Indice"

        # 0) EXCLUIR filas con NaN en las columnas seleccionadas
        df_eval = df_ind.dropna(subset=cols_attrs).copy()
        quitadas = len(df_ind) - len(df_eval)
        if quitadas > 0:
            st.caption(f"Se excluyeron {quitadas:,} filas por faltantes en {cols_attrs}")

        # 1) Clases sobre df_eval (sin NaN)
        clases = indiscernibility(cols_attrs, df_eval)

        # 2) Resumen
        longitudes = [(i, len(s)) for i, s in enumerate(clases)]
        longitudes_orden = sorted(longitudes, key=lambda x: x[1], reverse=True)
        nombres = {idx: f"Conjunto {k+1}" for k, (idx, _) in enumerate(longitudes_orden)}

        if not clases:
            st.warning("No se formaron clases (verifica ADL seleccionadas).")
        else:
            st.success(f"Se formaron {len(clases)} clases de indiscernibilidad.")

            resumen_df = pd.DataFrame({
                "Conjunto": [nombres[i] for i, _ in longitudes_orden],
                "Tamaño":   [tam for _, tam in longitudes_orden]
            })
            st.subheader("Resumen de clases (ordenadas por tamaño)")
            st.dataframe(resumen_df, use_container_width=True)

            # Persistir artefactos para pasos siguientes
            st.session_state["ind_cols"] = cols_attrs
            st.session_state["ind_df"] = df_ind.copy()      # completo (con NaN)
            st.session_state["ind_df_eval"] = df_eval.copy()  # SIN NaN (usado para clases)
            st.session_state["ind_classes"] = clases
            st.session_state["ind_lengths"] = longitudes_orden
            st.session_state["ind_min_size"] = int(min_size_for_pie)

# ==== RENDER FUERA DEL BOTÓN: usa lo que quedó en session_state ====

def _render_ind_outputs_from_state():
    ss = st.session_state
    need = ("ind_cols", "ind_df", "ind_df_eval", "ind_classes", "ind_lengths", "ind_min_size")
    if not all(k in ss for k in need) or not ss["ind_classes"]:
        return  # aún no hay datos para render

    cols_attrs       = ss["ind_cols"]
    df_ind           = ss["ind_df"]        # con NaN
    df_eval          = ss["ind_df_eval"]   # SIN NaN en cols_attrs
    clases           = ss["ind_classes"]
    longitudes_orden = ss["ind_lengths"]
    min_size_for_pie = int(ss["ind_min_size"])
    top_n_radar      = ss.get("top_n_radar_value", 15)

    nombres = {idx: f"Conjunto {k+1}" for k, (idx, _) in enumerate(longitudes_orden)}

    # --- Pastel ---
    candidatas = [(nombres[i], tam) for i, tam in longitudes_orden if tam >= min_size_for_pie]
    if candidatas:
        labels  = [n for n, _ in candidatas]
        valores = [v for _, v in candidatas]
        total   = sum(valores)
        fig_pie, ax_pie = plt.subplots(figsize=(7, 7))
        ax_pie.pie(valores, labels=labels,
                   autopct=lambda p: f"{p:.1f}%\n({int(p*total/100):,})",
                   startangle=140)
        ax_pie.axis('equal')
        ax_pie.set_title(f"Participación de clases (≥ {min_size_for_pie} filas)")
        st.pyplot(fig_pie)
    else:
        st.info(f"No hay clases con tamaño ≥ {min_size_for_pie} para el pastel.")
        return  # sin pastel no tiene sentido seguir

    # --- DataFrame debajo del pastel + nivel_riesgo (solo filas de df_eval) ---
    if not df_eval.empty:
        vals = df_eval[cols_attrs].apply(pd.to_numeric, errors="coerce")
        count_ones = (vals == 1).sum(axis=1)
        all_twos   = (vals == 2).all(axis=1)
        nivel = np.where(
            all_twos, "Riesgo nulo",
            np.where(
                count_ones <= 2,
                np.where(count_ones >= 1, "Riesgo leve", "Riesgo leve"),
                np.where(count_ones == 3, "Riesgo moderado", "Riesgo severo")
            )
        )
        df_eval_riesgo = df_eval.copy()
        df_eval_riesgo["nivel_riesgo"] = nivel
        st.subheader("Filas usadas en el pastel (sin NaN) + nivel_riesgo")
        st.dataframe(df_eval_riesgo.reset_index(), use_container_width=True)
        st.download_button(
            "Descargar filas del pastel con nivel_riesgo (CSV)",
            data=df_eval_riesgo.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="filas_pastel_con_nivel_riesgo.csv",
            mime="text/csv",
            key="dl_df_eval_riesgo"
        )
        ss["df_eval_riesgo"] = df_eval_riesgo.copy()

    # --- Radar de los N conjuntos más grandes (sobre df_eval) ---
    def determinar_color(valores):
        cnt = sum(1 for v in valores if pd.to_numeric(v, errors="coerce") == 1)
        if cnt == 0: return 'blue'
        if 1 <= cnt < 3: return 'green'
        if cnt == 3: return 'yellow'
        if 4 <= cnt < 5: return 'orange'
        return 'red'

    st.subheader("Radar de los conjuntos más numerosos")
    top_idxs = [i for i, _ in longitudes_orden[:int(top_n_radar)]]
    top_sets = [(nombres[i], clases[i]) for i in top_idxs]

    total_pacientes = len(df_eval)
    n = int(top_n_radar)
    cols_grid = 5
    rows_grid = int(np.ceil(n / cols_grid))
    fig, axs = plt.subplots(rows_grid, cols_grid, figsize=(cols_grid*6, rows_grid*5), subplot_kw=dict(polar=True))
    axs = np.atleast_2d(axs); fig.subplots_adjust(hspace=0.8, wspace=0.6)

    k = len(cols_attrs)
    angulos = np.linspace(0, 2 * np.pi, k, endpoint=False).tolist()
    angulos_cerrado = angulos + angulos[:1]

    for idx_plot in range(rows_grid * cols_grid):
        r = idx_plot // cols_grid; c = idx_plot % cols_grid
        ax = axs[r, c]
        if idx_plot >= n:
            ax.axis('off'); continue
        nombre, conjunto_idx = top_sets[idx_plot]
        indices = sorted(list(conjunto_idx))
        df_conj = df_eval.loc[indices, cols_attrs]
        if df_conj.empty:
            valores = [0]*k; num_filas_df = 0
        else:
            valores = df_conj.iloc[0].tolist(); num_filas_df = len(df_conj)
        valores_cerrados = list(valores) + [valores[0]]
        color = determinar_color(valores)
        ax.plot(angulos_cerrado, valores_cerrados, color=color)
        ax.fill(angulos_cerrado, valores_cerrados, color=color, alpha=0.25)
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_xticks(angulos); ax.set_xticklabels(cols_attrs, fontsize=10)
        ax.yaxis.grid(True); ax.set_ylim(0, 2)
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels([0, 1, 2], fontsize=9)
        pct = (num_filas_df / total_pacientes * 100) if total_pacientes else 0.0
        ax.set_title(nombre, fontsize=12)
        ax.text(0.5, -0.2, f"Filas: {num_filas_df} ({pct:.2f}%)",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
    st.pyplot(fig)

    # --- Gráfico compuesto (pastel + radares incrustados) ---
    candidatas_idx_nom_tam = [(i, nombres[i], tam) for i, tam in longitudes_orden if tam >= min_size_for_pie]
    if candidatas_idx_nom_tam:
        nombres_dataframes = [nom for _, nom, _ in candidatas_idx_nom_tam]
        tamanios = [tam for _, _, tam in candidatas_idx_nom_tam]
        total_incluido = sum(tamanios)
        porcentajes = [(nom, (tam/total_incluido*100.0) if total_incluido else 0.0)
                       for _, nom, tam in candidatas_idx_nom_tam]

        valores_dataframes, colores_dataframes = [], []
        for idx, _, _ in candidatas_idx_nom_tam:
            indices = sorted(list(clases[idx]))
            sub = df_eval.loc[indices, cols_attrs]
            vals = sub.iloc[0].tolist() if not sub.empty else [0]*len(cols_attrs)
            valores_dataframes.append(vals)
            colores_dataframes.append(determinar_color(vals))

        min_radio = 1.0; max_radio = 2.40
        radar_size_min = 0.10; radar_size_max = 0.19
        etiquetas_radar = [et.replace('_21','').replace('_18','') for et in cols_attrs]

        fig_comp = plt.figure(figsize=(16, 16))
        main_ax = plt.subplot(111); main_ax.set_position([0.1, 0.1, 0.8, 0.8])

        if porcentajes:
            _, valores_porcentajes = zip(*porcentajes)
            valores_porcentajes = [float(p) for p in valores_porcentajes]
        else:
            valores_porcentajes = []

        colores_ajustados = colores_dataframes[:len(valores_porcentajes)]
        wedges, texts, autotexts = main_ax.pie(
            valores_porcentajes, colors=colores_ajustados,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 17}, labeldistance=1.1
        )

        if wedges:
            angulos_pastel = [(w.theta1 + w.theta2)/2 for w in wedges]
            anchos = [abs(w.theta2 - w.theta1) for w in wedges]
            max_ancho = max(anchos) if anchos else 1
            angulos_rad = [np.deg2rad(a) for a in angulos_pastel]

            radios_personalizados = [
                min_radio + (1 - (log1p(a)/log1p(max_ancho))) * (max_radio - min_radio)
                for a in anchos
            ]
            tamaños_radar = [
                radar_size_min + (a/max_ancho) * (radar_size_max - radar_size_min)
                for a in anchos
            ]

            angulos_rad_separados = angulos_rad.copy()
            min_sep = np.deg2rad(7)
            for i in range(1, len(angulos_rad_separados)):
                while abs(angulos_rad_separados[i] - angulos_rad_separados[i-1]) < min_sep:
                    angulos_rad_separados[i] += min_sep/2

            for i, (nombre, vals, color, ang_rad, r_inset, tam_radar) in enumerate(
                zip(nombres_dataframes, valores_dataframes, colores_dataframes,
                    angulos_rad_separados, radios_personalizados, tamaños_radar)
            ):
                factor_alejamiento = 2.3
                x = 0.5 + r_inset*np.cos(ang_rad)/factor_alejamiento
                y = 0.5 + r_inset*np.sin(ang_rad)/factor_alejamiento
                radar_ax = fig_comp.add_axes([x - tam_radar/2, y - tam_radar/2, tam_radar, tam_radar], polar=True)

                vals = list(vals)[:len(cols_attrs)] or [0]*len(cols_attrs)
                vals_c = vals + [vals[0]]
                angs = np.linspace(0, 2*np.pi, len(cols_attrs), endpoint=False).tolist()
                angs_c = angs + [angs[0]]

                radar_ax.set_theta_offset(np.pi/2); radar_ax.set_theta_direction(-1)
                radar_ax.plot(angs_c, vals_c, color=color)
                radar_ax.fill(angs_c, vals_c, color=color, alpha=0.3)
                radar_ax.set_xticks(angs); radar_ax.set_xticklabels(etiquetas_radar, fontsize=13)
                radar_ax.set_yticks([0,1,2]); radar_ax.set_yticklabels(['0','1','2'], fontsize=11)
                radar_ax.set_ylim(0,2); radar_ax.yaxis.grid(True, linestyle='dotted', linewidth=0.5)

                x0 = 0.5 + 0.3*np.cos(ang_rad); y0 = 0.5 + 0.3*np.sin(ang_rad)
                con = ConnectionPatch(
                    xyA=(x0, y0), coordsA=fig_comp.transFigure,
                    xyB=(x, y), coordsB=fig_comp.transFigure,
                    color='gray', lw=0.8, linestyle='--'
                )
                fig_comp.add_artist(con)

        st.pyplot(fig_comp)

# 👉 Llamada al renderer SIEMPRE, con o sin botón
_render_ind_outputs_from_state()





# ==================================================================== hasta aqui todo bien

# ====== Inspección de un subconjunto (del pastel) + correlaciones ======
ss = st.session_state
need = ("ind_classes", "ind_lengths", "ind_min_size", "ind_df_reducido", "ind_adl_cols", "ind_cols")
if not all(k in ss for k in need) or not ss["ind_classes"]:
    st.info("Calcula indiscernibilidad para habilitar la inspección por subconjunto.")
else:
    # Candidatos: solo clases que entraron al pastel (≥ umbral)
    umbral = int(ss["ind_min_size"])
    candidatos = [(i, tam) for i, tam in ss["ind_lengths"] if tam >= umbral]

    if not candidatos:
        st.info("No hay subconjuntos en el pastel para inspeccionar (ajusta el umbral).")
    else:
        # Nombres legibles coherentes con el resumen
        nombres = {idx: f"Conjunto {k+1}" for k, (idx, _) in enumerate(ss["ind_lengths"])}
        labels_map = {f"{nombres[i]} — {tam} filas": i for i, tam in candidatos}

        sel_label = st.selectbox(
            "Elige un subconjunto del pastel para visualizar y correlacionar",
            options=list(labels_map.keys()),
            index=0,
            key="sel_subconjunto_pastel"
        )
        sel_i = labels_map[sel_label]

        # Índices de filas del subconjunto (en df_eval/df_ind)
        idxs = sorted(list(ss["ind_classes"][sel_i]))

        # DF con TODAS las ADL normalizadas (no solo las usadas en ind)
        dfr = ss["ind_df_reducido"]
        dfr2 = dfr.set_index("Indice") if "Indice" in dfr.columns else dfr
        adl_cols_all = ss["ind_adl_cols"]
        df_sub = dfr2.loc[idxs, adl_cols_all].copy()

        # ---- nivel_riesgo (según columnas usadas en indiscernibilidad) ----
        cols_attrs = ss["ind_cols"]
        cols_usables = [c for c in cols_attrs if c in df_sub.columns]
        if cols_usables:
            vals = df_sub[cols_usables].apply(pd.to_numeric, errors="coerce")
            count_ones = (vals == 1).sum(axis=1)
            all_twos   = (vals == 2).all(axis=1)
            nivel = np.where(
                all_twos, "Riesgo nulo",
                np.where(
                    count_ones <= 2,
                    np.where(count_ones >= 1, "Riesgo leve", "Riesgo leve"),
                    np.where(count_ones == 3, "Riesgo moderado", "Riesgo severo")
                )
            )
        else:
            nivel = np.array(["Riesgo nulo"] * len(df_sub))  # fallback

        df_sub_disp = df_sub.copy()
        df_sub_disp.insert(0, "nivel_riesgo", nivel)

        st.subheader(f"Vista del {nombres[sel_i]} — {len(df_sub_disp):,} filas")
        st.dataframe(df_sub_disp.reset_index(), use_container_width=True)
        st.download_button(
            "Descargar subconjunto ADL (CSV)",
            data=df_sub_disp.reset_index().to_csv(index=False).encode("utf-8"),
            file_name=f"{nombres[sel_i]}_ADL.csv",
            mime="text/csv",
            key=f"dl_{sel_i}_adl"
        )


        # ---- Matriz de correlación (todas las ADL del subconjunto), sin NaN en el gráfico ----
        st.subheader("Matriz de correlación (todas las ADL del subconjunto)")

        # 1) Limpieza: quedarnos solo con columnas con suficientes datos y variación
        num_all = df_sub.apply(pd.to_numeric, errors="coerce")

        min_valid = max(2, int(0.5 * len(num_all)))   # al menos 50% de filas no nulas
        keep_cols = [
            c for c in num_all.columns
            if num_all[c].notna().sum() >= min_valid and num_all[c].nunique(dropna=True) > 1
        ]

        dropped = [c for c in num_all.columns if c not in keep_cols]
        if dropped:
            st.caption(f"Columnas excluidas por falta de datos/variación: {', '.join(dropped)}")

        num = num_all[keep_cols]
        corr = num.corr()  # Pearson por pares válidos

        # 2) Plot: enmascarar NaN para que no aparezcan bloques 'nan'
        cmap = plt.cm.coolwarm
        cmap.set_bad(color='lightgray')  # celdas sin valor quedarán gris claro
        mat = np.ma.masked_invalid(corr.values)

        fig_w = max(8, 0.45 * len(corr.columns))
        fig_h = max(6, 0.45 * len(corr.columns))
        figc, axc = plt.subplots(figsize=(fig_w, fig_h))
        im = axc.imshow(mat, cmap=cmap, vmin=-1, vmax=1)
        figc.colorbar(im, ax=axc, fraction=0.046, pad=0.04)

        axc.set_xticks(range(len(corr.columns))); axc.set_xticklabels(corr.columns, rotation=90)
        axc.set_yticks(range(len(corr.index)));  axc.set_yticklabels(corr.index)
        axc.set_title(f"Correlaciones — {nombres[sel_i]}")

        # 3) Anotar coeficientes solo donde hay valor
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                val = corr.values[i, j]
                if not np.isnan(val):
                    axc.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        figc.tight_layout()
        st.pyplot(figc)


# =========================
# Reductos de 4 y 3 variables (evaluación vs. partición original en el subconjunto del pastel)
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

ss = st.session_state
need = ("ind_df_eval", "ind_cols", "ind_classes", "ind_lengths", "ind_min_size")
if not all(k in ss for k in need) or not isinstance(ss["ind_df_eval"], pd.DataFrame):
    st.info("👉 Calcula indiscernibilidad primero (se necesitan ind_df_eval, ind_cols, ind_classes...).")
else:
    # ---------- utilidades (locales) ----------
    def blocks_to_labels(blocks, universo):
        lbl = {}
        for k, S in enumerate(blocks):
            for idx in S:
                lbl[idx] = k
        return np.array([lbl.get(i, -1) for i in universo])

    def contingency_from_labels(y_true, y_pred):
        s1 = pd.Series(y_true).astype("category")
        s2 = pd.Series(y_pred).astype("category")
        return pd.crosstab(s1, s2).values

    def pairs_same(counts):
        counts = np.asarray(counts, dtype=np.int64)
        return (counts * (counts - 1) // 2).sum()

    def ari_from_contingency(C):
        n = C.sum()
        if n == 0:
            return 1.0
        a = C.sum(axis=1)
        b = C.sum(axis=0)
        sum_comb = (C * (C - 1) // 2).sum()
        sum_a = (a * (a - 1) // 2).sum()
        sum_b = (b * (b - 1) // 2).sum()
        T = n * (n - 1) // 2
        expected = (sum_a * sum_b) / T if T else 0.0
        max_index = 0.5 * (sum_a + sum_b)
        denom = max_index - expected
        return float((sum_comb - expected) / denom) if denom != 0 else 1.0

    def nmi_from_contingency(C):
        n = C.sum()
        if n == 0:
            return 1.0
        a = C.sum(axis=1)
        b = C.sum(axis=0)
        # Mutual Information
        I = 0.0
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                nij = C[i, j]
                if nij > 0:
                    I += (nij / n) * np.log((nij * n) / (a[i] * b[j]))
        p = a / n
        q = b / n
        Hu = -np.sum([pi * np.log(pi) for pi in p if pi > 0])
        Hv = -np.sum([qj * np.log(qj) for qj in q if qj > 0])
        denom = np.sqrt(Hu * Hv)
        return float(I / denom) if denom > 0 else 1.0

    def preservation_metrics_from_contingency(C):
        n = C.sum()
        if n == 0:
            return 1.0, 1.0
        T = n * (n - 1) // 2
        a = C.sum(axis=1)
        b = C.sum(axis=0)
        same_orig = pairs_same(a)
        same_red  = pairs_same(b)
        same_both = (C * (C - 1) // 2).sum()
        pres_same = same_both / same_orig if same_orig > 0 else 1.0
        diff_orig = T - same_orig
        diff_to_same = same_red - same_both
        pres_diff = (diff_orig - diff_to_same) / diff_orig if diff_orig > 0 else 1.0
        return pres_same, pres_diff

    # ---------- subconjunto del pastel ----------
    umbral = int(ss["ind_min_size"])
    ids_pastel = [i for i, tam in ss["ind_lengths"] if tam >= umbral]
    if not ids_pastel:
        st.info(f"No hay clases con tamaño ≥ {umbral} para evaluar reductos.")
    else:
        universo_sel = sorted(set().union(*[ss["ind_classes"][i] for i in ids_pastel]))
        if len(universo_sel) == 0:
            st.info("No hay filas en el subconjunto del pastel.")
        else:
            df_eval_sub = ss["ind_df_eval"].loc[universo_sel].copy()  # SIN NaN en columnas usadas originalmente
            cols_all = list(ss["ind_cols"])
            m = len(cols_all)
            if m < 3:
                st.info("Se requieren al menos 3 variables en la partición original para evaluar reductos de 3/4.")
            else:
                # ---------- partición original en el subconjunto ----------
                bloques_orig = indiscernibility(cols_all, df_eval_sub)
                y_orig = blocks_to_labels(bloques_orig, universo_sel)

                # ---------- generar reductos de tamaño 4 y 3 ----------
                reductos = {}
                if m >= 4:
                    for comb in combinations(cols_all, 4):
                        reductos[f"De 4: {', '.join(comb)}"] = list(comb)
                if m >= 3:
                    for comb in combinations(cols_all, 3):
                        reductos[f"De 3: {', '.join(comb)}"] = list(comb)

                # (opcional) limitar por seguridad si hay demasiadas combinaciones
                MAX_MODELOS = 500
                if len(reductos) > MAX_MODELOS:
                    st.warning(f"Hay {len(reductos)} combinaciones. Se evaluarán solo las primeras {MAX_MODELOS}.")
                    reductos = dict(list(reductos.items())[:MAX_MODELOS])

                # ---------- evaluar reductos ----------
                resultados = []
                block_sizes = {"Original": [len(S) for S in bloques_orig]}

                for nombre, cols in reductos.items():
                    bloques_red = indiscernibility(cols, df_eval_sub)
                    y_red = blocks_to_labels(bloques_red, universo_sel)
                    C = contingency_from_labels(y_orig, y_red)

                    ari = ari_from_contingency(C)
                    nmi = nmi_from_contingency(C)
                    pres_same, pres_diff = preservation_metrics_from_contingency(C)

                    resultados.append({
                        "Reducto": nombre,
                        "#vars": len(cols),
                        "#bloques(orig)": len(bloques_orig),
                        "#bloques(red)": len(bloques_red),
                        "ARI": round(ari, 3),
                        "NMI": round(nmi, 3),
                        "Preservación iguales (%)": round(pres_same * 100, 1),
                        "Preservación distintos (%)": round(pres_diff * 100, 1),
                    })
                    block_sizes[nombre] = [len(S) for S in bloques_red]

                if not resultados:
                    st.info("No se pudieron evaluar reductos en el subconjunto.")
                else:
                    df_closeness = pd.DataFrame(resultados).sort_values(
                        by=["ARI", "Preservación iguales (%)", "Preservación distintos (%)"],
                        ascending=False
                    ).reset_index(drop=True)

                    st.subheader("Reductos de 4 y 3 variables — Métricas (subconjunto del pastel)")
                    st.caption(f"Filas en evaluación: {len(universo_sel):,} | Variables originales: {m}")
                    st.dataframe(df_closeness, use_container_width=True)

                    st.download_button(
                        "Descargar métricas de reductos (CSV)",
                        data=df_closeness.to_csv(index=False).encode("utf-8"),
                        file_name="reductos_4y3_metricas.csv",
                        mime="text/csv",
                        key="dl_reductos_4y3"
                    )

                    # ---------- mejores reductos de 4 y 3 ----------
                    best4 = df_closeness[df_closeness["#vars"] == 4].head(1)
                    best3 = df_closeness[df_closeness["#vars"] == 3].head(1)

                    if not best4.empty:
                        r = best4.iloc[0]
                        st.success(
                            f"🟩 Mejor reducto de 4 variables: **{r['Reducto']}** — "
                            f"ARI={r['ARI']}, NMI={r['NMI']}, "
                            f"Pres. iguales={r['Preservación iguales (%)']}%, "
                            f"Pres. distintos={r['Preservación distintos (%)']}%"
                        )
                    if not best3.empty:
                        r = best3.iloc[0]
                        st.success(
                            f"🟨 Mejor reducto de 3 variables: **{r['Reducto']}** — "
                            f"ARI={r['ARI']}, NMI={r['NMI']}, "
                            f"Pres. iguales={r['Preservación iguales (%)']}%, "
                            f"Pres. distintos={r['Preservación distintos (%)']}%"
                        )

                    # ---------- Expander con gráficos opcionales ----------
                    with st.expander("Gráficos: Boxplot de tamaños y Heatmap del mejor reducto", expanded=False):
                        # Boxplot: Original vs top-K (mezcla de 4 y 3)
                        K = min(10, len(df_closeness))
                        top_names = ["Original"] + df_closeness.loc[:K-1, "Reducto"].tolist()
                        max_len = max(len(block_sizes[n]) for n in top_names)
                        data_box = np.full((max_len, len(top_names)), np.nan)
                        for j, nm in enumerate(top_names):
                            arr = np.array(block_sizes[nm], dtype=float)
                            data_box[:len(arr), j] = arr

                        fig_box, ax_box = plt.subplots(figsize=(max(10, 1.2*len(top_names)), 6))
                        ax_box.boxplot([data_box[:, j][~np.isnan(data_box[:, j])] for j in range(len(top_names))],
                                       notch=True)
                        ax_box.set_xticks(range(1, len(top_names)+1))
                        ax_box.set_xticklabels(top_names, rotation=45, ha="right")
                        ax_box.set_ylabel("Tamaño de bloque")
                        ax_box.set_title("Distribución de tamaños de bloques — Original vs. mejores reductos")
                        ax_box.grid(axis='y', linestyle='--', alpha=0.4)
                        st.pyplot(fig_box)

                        # Heatmap del mejor global (máximo ARI)
                        best_name = df_closeness.iloc[0]["Reducto"]
                        # Recuperar columnas desde el nombre:
                        # El nombre tiene formato "De k: A, B, C" -> parsear
                        cols_best = [c.strip() for c in best_name.split(":", 1)[1].split(",")]
                        bloques_best = indiscernibility(cols_best, df_eval_sub)

                        M = np.zeros((len(bloques_orig), len(bloques_best)), dtype=int)
                        for i, Bo in enumerate(bloques_orig):
                            for j, Br in enumerate(bloques_best):
                                M[i, j] = len(Bo.intersection(Br))

                        fig_hm, ax_hm = plt.subplots(figsize=(10, 6))
                        im = ax_hm.imshow(M, cmap="Blues")
                        fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
                        ax_hm.set_xlabel(f"Partición reducida ({best_name})")
                        ax_hm.set_ylabel("Partición original (subconjunto)")
                        ax_hm.set_title("Correspondencia entre bloques (conteos)")
#                        ax_hm.set_xticks(range(M.shape[1])); ax_hm.set_xticklabels([f"Red_{j+1}" for j in range(M.shape[1])])
#                        ax_hm.set_yticks(range(M.shape[0])); ax_hm.set_yticklabels([f"Orig_{i+1}" for i in range(M.shape[0])])

                        # Ticks y etiquetas (evitar traslapes)
                        ax_hm.set_xticks(np.arange(M.shape[1]))
                        ax_hm.set_yticks(np.arange(M.shape[0]))
                        ax_hm.set_xticklabels([f"Red_{j+1}" for j in range(M.shape[1])], rotation=45, ha="right", fontsize=9)
                        ax_hm.set_yticklabels([f"Orig_{i+1}" for i in range(M.shape[0])], fontsize=9)

                        # separa un poco las etiquetas del eje
                        ax_hm.tick_params(axis="x", which="major", pad=8)
                        ax_hm.tick_params(axis="y", which="major", pad=8)

                        
                        # deja más espacio para las etiquetas
                        fig_hm.tight_layout()
                        fig_hm.subplots_adjust(bottom=0.24, left=0.24)  # ajusta si aún ves traslape
                        
                        if M.shape[0] * M.shape[1] <= 900:
                            for i in range(M.shape[0]):
                                for j in range(M.shape[1]):
                                    ax_hm.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8)
                        st.pyplot(fig_hm)






# =========================
# Reductos + RF (rápido) + Predicción en todo el pastel + barras comparativas
# =========================
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

ss = st.session_state
needed = ("ind_cols","ind_df","ind_classes","ind_lengths","ind_min_size")
if not all(k in ss for k in needed):
    st.info("⚠️ Calcula indiscernibilidad primero.")
else:
    # ---------- utilidades ligeras (sin sklearn para métricas de partición) ----------
    def blocks_to_labels(blocks, universo):
        lab = {}
        for k, S in enumerate(blocks):
            for i in S: lab[i] = k
        return np.array([lab[i] for i in universo])

    def contingency_from_labels(y_true, y_pred):
        s1 = pd.Series(y_true).astype("category")
        s2 = pd.Series(y_pred).astype("category")
        return pd.crosstab(s1, s2).values

    def pairs_same(counts):
        counts = np.asarray(counts, dtype=np.int64)
        return (counts*(counts-1)//2).sum()

    def ari_from_contingency(C):
        n = C.sum()
        if n == 0: return 1.0
        a = C.sum(axis=1); b = C.sum(axis=0)
        sum_comb = (C*(C-1)//2).sum()
        sum_a = (a*(a-1)//2).sum()
        sum_b = (b*(b-1)//2).sum()
        T = n*(n-1)//2
        expected = (sum_a*sum_b)/T if T else 0.0
        max_index = 0.5*(sum_a+sum_b)
        denom = max_index - expected
        return float((sum_comb - expected)/denom) if denom != 0 else 1.0

    # ---------- universo del pastel ----------
    umbral = int(ss["ind_min_size"])
    ids_pastel = [i for i, tam in ss["ind_lengths"] if tam >= umbral]
    idxs_pastel = sorted(set().union(*[ss["ind_classes"][i] for i in ids_pastel])) if ids_pastel else []

    if not idxs_pastel:
        st.info("No hay filas en el pastel con el umbral actual.")
    else:
        df_full = ss["ind_df"]                       # ADL indexado por 'Indice' (puede tener NaN)
        ind_cols = list(ss["ind_cols"])              # típicamente 5 columnas elegidas
        df_pastel_full = df_full.loc[idxs_pastel].copy()

        # ---------- entrenamiento SOLO con filas sin NaN en TODAS las ind_cols ----------
        df_pastel_eval = df_pastel_full.dropna(subset=ind_cols).copy()
        if df_pastel_eval.empty:
            st.warning("No hay filas sin NaN en TODAS las columnas para entrenar.")
            st.stop()

        # Etiqueta por REGLA (nulo, leve, moderado, severo)
        vals = df_pastel_eval[ind_cols].apply(pd.to_numeric, errors="coerce")
        count_ones = (vals == 1).sum(axis=1)
        all_twos   = (vals == 2).all(axis=1)
        y_regla = np.where(
            all_twos, "Riesgo nulo",
            np.where(
                count_ones <= 2,
                np.where(count_ones >= 1, "Riesgo leve", "Riesgo leve"),
                np.where(count_ones == 3, "Riesgo moderado", "Riesgo severo")
            )
        )
        df_pastel_eval = df_pastel_eval.copy()
        df_pastel_eval["nivel_riesgo"] = y_regla

        # ---------- mejor reducto de 4 y de 3 (rápido, con ARI en df_pastel_eval) ----------
        # partición original (todas las ind_cols) y universo ordenado
        universe = list(df_pastel_eval.index)
        bloques_orig = indiscernibility(ind_cols, df_pastel_eval)
        y_orig = blocks_to_labels(bloques_orig, universe)

        def score_cols(cols):
            bloques = indiscernibility(cols, df_pastel_eval)
            y = blocks_to_labels(bloques, universe)
            C = contingency_from_labels(y_orig, y)
            return ari_from_contingency(C)

        best4 = None; best4_score = -1
        for comb in combinations(ind_cols, max(len(ind_cols)-1, 4)):  # típicamente 4 vars
            ari = score_cols(list(comb))
            if ari > best4_score:
                best4_score, best4 = ari, list(comb)

        best3 = None; best3_score = -1
        if len(ind_cols) >= 5:  # si hay 5 originales
            for comb in combinations(ind_cols, 3):
                ari = score_cols(list(comb))
                if ari > best3_score:
                    best3_score, best3 = ari, list(comb)
        else:
            # si no hay 5 originales, intenta len(ind_cols)-2
            k3 = max(3, len(ind_cols)-2)
            for comb in combinations(ind_cols, k3):
                ari = score_cols(list(comb))
                if ari > best3_score:
                    best3_score, best3 = ari, list(comb)

        st.markdown(f"**Mejor reducto (4 vars)**: {best4} — ARI={best4_score:.3f}")
        if best3 is not None:
            st.markdown(f"**Mejor reducto (3 vars)**: {best3} — ARI={best3_score:.3f}")

        # ---------- Entrenar RF(s) (rápido) ----------
        # Usamos class_weight en lugar de SMOTE para acelerar. n_estimators ajustado.
        def entrenar_rf(df_train, feat_cols, target_col="nivel_riesgo"):
            X = df_train[feat_cols].apply(pd.to_numeric, errors="coerce")
            y_raw = df_train[target_col].astype(str).values
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            imp = SimpleImputer(strategy="median")
            X_imp = imp.fit_transform(X)

            rf = RandomForestClassifier(
                n_estimators=200,  # rápido y decente
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
                oob_score=False
            )
            rf.fit(X_imp, y)
            return rf, le, imp

        rf4, le4, imp4 = entrenar_rf(df_pastel_eval, best4)
        ss["rf_best4"] = rf4
        ss["rf_best4_cols"] = best4
        ss["rf_best4_le"] = le4
        ss["rf_best4_imp"] = imp4

        if best3 is not None:
            rf3, le3, imp3 = entrenar_rf(df_pastel_eval, best3)
            ss["rf_best3"] = rf3
            ss["rf_best3_cols"] = best3
            ss["rf_best3_le"] = le3
            ss["rf_best3_imp"] = imp3

        st.success("Modelos RF entrenados (best4 y, si procede, best3).")


####FALLBACK
# === Fallback que acepta NaN en predicción: HistGradientBoosting ===
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# Usa el mismo dataset del pastel con la etiqueta 'nivel_riesgo'
# (si en tu código se llama distinto, cambia df_pastel_eval por el correcto,
# por ejemplo: st.session_state["df_eval_riesgo"])
df_fb = df_pastel_eval.copy()

Xfb = df_fb[st.session_state["ind_cols"]].apply(pd.to_numeric, errors="coerce")
yfb_raw = df_fb["nivel_riesgo"].astype(str)

le_fb = LabelEncoder()
yfb = le_fb.fit_transform(yfb_raw.values)

hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=42)
hgb.fit(Xfb, yfb)

st.session_state["fb_hgb_model"] = hgb
st.session_state["fb_hgb_cols"]  = st.session_state["ind_cols"]
st.session_state["fb_hgb_le"]    = le_fb

st.info("Modelo fallback (HGB) entrenado.")



# =========================
# Predicción en TODO el DF indiscernible (no solo el pastel)
# =========================
ss = st.session_state

have4 = all(k in ss for k in ("rf_best4","rf_best4_cols","rf_best4_imp","rf_best4_le"))
have3 = all(k in ss for k in ("rf_best3","rf_best3_cols","rf_best3_imp","rf_best3_le"))

if "ind_df" not in ss:
    st.info("Calcula indiscernibilidad primero (para disponer de ind_df).")
else:
    st.subheader("Predicción en todo el DataFrame (indiscernible)")

    df_all = ss["ind_df"].copy()  # ADL con posibles NaN (index='Indice')

    if not (have4 or have3):
        st.warning("No encuentro modelos entrenados (4/3 vars). Entrena los RF para generar predicciones.")
        df_pred_all = df_all.copy()
        df_pred_all["nivel_riesgo_pred"] = "Sin datos"
        ss["df_pred_all_rf"] = df_pred_all

    else:
        # usa el LE del modelo disponible (prefiere 4 vars)
        le = ss["rf_best4_le"] if have4 else ss["rf_best3_le"]

        
        # Serie donde iremos llenando las predicciones finales
        pred_all = pd.Series(index=df_all.index, dtype="object")

        # --- helper: predice imputando faltantes y permitiendo umbral de observados ---
        def predict_with_impute(cols, model, imputer, rows_mask=None, min_obs=0):
            # candidatos (todas las filas o solo las indicadas por rows_mask)
            if rows_mask is None:
                idx_cand = df_all.index
            else:
                idx_cand = df_all.index[rows_mask]

            if len(idx_cand) == 0:
                return pd.Index([]), None

            Xraw = df_all.loc[idx_cand, cols].apply(pd.to_numeric, errors="coerce")

            # exigir un mínimo de features observadas (antes de imputar)
            if min_obs and min_obs > 0:
                obs = Xraw.notna().sum(axis=1)
                Xraw = Xraw.loc[obs >= min_obs]
                idx_cand = Xraw.index
                if len(idx_cand) == 0:
                    return pd.Index([]), None

            # imputar como en el entrenamiento
            Ximp = imputer.transform(Xraw)
            yhat = model.predict(Ximp)
            return idx_cand, le.inverse_transform(yhat)

        # 1) Modelo de 4 variables (prioridad). Requiere ≥3 observadas para usarlo.
        if have4:
            idx4, lab4 = predict_with_impute(
                ss["rf_best4_cols"], ss["rf_best4"], ss["rf_best4_imp"],
                rows_mask=None, min_obs=3
            )
            if lab4 is not None and len(idx4) > 0:
                pred_all.loc[idx4] = lab4

        # 2) Modelo de 3 variables (respaldo) SOLO donde aún no hay predicción. Requiere ≥2 observadas.
        if have3:
            restante_mask = pred_all.isna()
            idx3, lab3 = predict_with_impute(
                ss["rf_best3_cols"], ss["rf_best3"], ss["rf_best3_imp"],
                rows_mask=restante_mask, min_obs=2
            )
            if lab3 is not None and len(idx3) > 0:
                pred_all.loc[idx3] = lab3

        # 3) Fallback HGB para las que aún queden sin predicción
        have_fb = all(k in st.session_state for k in ("fb_hgb_model","fb_hgb_cols","fb_hgb_le"))
        if have_fb:
            faltantes = pred_all.isna()
            if faltantes.any():
                Xfb_all = df_all.loc[faltantes, st.session_state["fb_hgb_cols"]].apply(pd.to_numeric, errors="coerce")
                yfb_hat = st.session_state["fb_hgb_model"].predict(Xfb_all)
                lab_fb  = st.session_state["fb_hgb_le"].inverse_transform(yfb_hat)
                pred_all.loc[faltantes] = lab_fb

        pred_all.fillna("Sin datos", inplace=True)

        df_pred_all = df_all.copy()
        df_pred_all["nivel_riesgo_pred"] = pred_all
        ss["df_pred_all_rf"] = df_pred_all
   
    
    # Muestra y descarga
    st.dataframe(df_pred_all.reset_index().head(50), use_container_width=True)
    st.download_button(
        "Descargar predicciones RF (todo ind_df) CSV",
        data=df_pred_all.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="predicciones_rf_todo_ind_df.csv",
        mime="text/csv",
        key="dl_pred_all_rf"
    )

    # =========================
    # Barras: Regla (sin NaN) vs RF (todo ind_df)
    # =========================
    orden = ["Riesgo nulo","Riesgo leve","Riesgo moderado","Riesgo severo"]

    cols_sel = ss.get("ind_cols", [])
    if cols_sel:
        df_regla = df_all.dropna(subset=cols_sel).copy()
        vals = df_regla[cols_sel].apply(pd.to_numeric, errors="coerce")
        ones = (vals == 1).sum(axis=1)
        twos = (vals == 2).all(axis=1)
        regla = np.where(
            twos, "Riesgo nulo",
            np.where(
                ones <= 2,
                np.where(ones >= 1, "Riesgo leve", "Riesgo leve"),
                np.where(ones == 3, "Riesgo moderado", "Riesgo severo")
            )
        )
        dist_regla = pd.Series(regla).value_counts().reindex(orden, fill_value=0)
    else:
        dist_regla = pd.Series([0]*len(orden), index=orden)

    dist_rf_all = (
        df_pred_all["nivel_riesgo_pred"]
        .value_counts()
        .reindex(orden + (["Sin datos"] if "Sin datos" in df_pred_all["nivel_riesgo_pred"].values else []), fill_value=0)
    )
    dist_rf_bar = dist_rf_all.reindex(orden, fill_value=0)

    x = np.arange(len(orden)); width = 0.38
    ymax = max(dist_regla.max(), dist_rf_bar.max(), 1)

    fig_b, ax_b = plt.subplots(figsize=(9, 4.8))
    b1 = ax_b.bar(x - width/2, dist_regla.values, width, label="Regla (sin NaN)")
    b2 = ax_b.bar(x + width/2, dist_rf_bar.values,  width, label="RF (todo ind_df)")
    ax_b.set_xticks(x); ax_b.set_xticklabels(orden)
    ax_b.set_ylabel("Participantes")
    ax_b.set_title("Niveles de riesgo: Regla vs RF")
    ax_b.legend(); ax_b.grid(axis='y', linestyle='--', alpha=0.3)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax_b.text(r.get_x()+r.get_width()/2, h + 0.01*ymax, f"{int(h)}",
                      ha="center", va="bottom", fontsize=9)
    st.pyplot(fig_b)

    # =========================
    # Pasteles: (1) Regla sin NaN vs (2) RF todo ind_df
    # =========================
    fig1, ax1 = plt.subplots(figsize=(6.5, 6.5))
    v1 = dist_regla.values
    if v1.sum() == 0:
        ax1.text(0.5, 0.5, "Sin filas válidas (sin NaN) para la regla",
                 ha="center", va="center", fontsize=12)
        ax1.axis("off")
    else:
        ax1.pie(v1, labels=dist_regla.index, autopct=lambda p: f"{p:.1f}%", startangle=120)
        ax1.axis('equal'); ax1.set_title("Riesgo — solo filas sin NaN (regla)")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6.5, 6.5))
    v2 = dist_rf_all.values
    if v2.sum() == 0:
        ax2.text(0.5, 0.5, "No hay predicciones RF disponibles",
                 ha="center", va="center", fontsize=12)
        ax2.axis("off")
    else:
        ax2.pie(v2, labels=dist_rf_all.index, autopct=lambda p: f"{p:.1f}%", startangle=120)
        ax2.axis('equal'); ax2.set_title("Riesgo — todo ind_df (RF 4→3 vars)")
    st.pyplot(fig2)



################################

# ==========================================================
# Formularios: ✍️ Captura manual y 📄 Subir Excel
# Requiere: pandas as pd, numpy as np, streamlit as st
# Usa modelos si existen en st.session_state; si no, aplica la "regla".
# ==========================================================

# --- Helper: normalizar nombres ADL externos (H11_18 -> H11, etc.) ---
def _normalize_adl_columns(df: pd.DataFrame, adl_base: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    rename_map = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for base in adl_base:
        # busca exacto, _18, _21 (case-insensitive)
        for cand in (base, f"{base}_18", f"{base}_21"):
            lc = cand.lower()
            if lc in lower_cols:
                rename_map[lower_cols[lc]] = base  # renombra a base
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

# --- Helper: riesgo por "regla" con las columnas elegidas en indiscernibilidad ---
def _riesgo_regla(df_in: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not isinstance(df_in, pd.DataFrame) or not cols:
        return pd.Series(index=df_in.index if isinstance(df_in, pd.DataFrame) else [], dtype="object")
    sub = df_in[cols].apply(pd.to_numeric, errors="coerce")
    mask_ok = sub.notna().all(axis=1)
    if not mask_ok.any():
        return pd.Series(index=df_in.index, dtype="object")

    vals = sub.loc[mask_ok]
    count_ones = (vals == 1).sum(axis=1)
    all_twos   = (vals == 2).all(axis=1)
    nivel = np.where(
        all_twos, "Riesgo nulo",
        np.where(
            count_ones <= 2,
            np.where(count_ones >= 1, "Riesgo leve", "Riesgo leve"),
            np.where(count_ones == 3, "Riesgo moderado", "Riesgo considerable")
        )
    )
    out = pd.Series(index=df_in.index, dtype="object")
    out.loc[mask_ok] = nivel
    return out

# --- Helper: predecir usando modelos (best4 -> best3) y si falta, "regla" ---
def predecir_con_modelos(df_in: pd.DataFrame) -> pd.DataFrame:
    ss = st.session_state
    if df_in is None or df_in.empty:
        return pd.DataFrame(index=[], columns=["nivel_riesgo_pred"])

    # Recuperar modelos entrenados (acepta variantes de nombre)
    have4 = (("rf_best4" in ss and "rf_best4_cols" in ss and ("rf_best4_le" in ss or "rf_label_encoder" in ss))
             or ("rf_best4_model" in ss and "rf_best4_cols" in ss and ("rf_best4_le" in ss or "rf_label_encoder" in ss)))
    have3 = (("rf_best3" in ss and "rf_best3_cols" in ss and ("rf_best3_le" in ss or "rf_label_encoder" in ss))
             or ("rf_best3_model" in ss and "rf_best3_cols" in ss and ("rf_best3_le" in ss or "rf_label_encoder" in ss)))

    model4 = ss.get("rf_best4") or ss.get("rf_best4_model")
    cols4  = ss.get("rf_best4_cols", [])
    le4    = ss.get("rf_best4_le") or ss.get("rf_label_encoder")
    imp4   = ss.get("rf_best4_imp") or ss.get("rf_best4_imputer")

    model3 = ss.get("rf_best3") or ss.get("rf_best3_model")
    cols3  = ss.get("rf_best3_cols", [])
    le3    = ss.get("rf_best3_le") or ss.get("rf_label_encoder")
    imp3   = ss.get("rf_best3_imp") or ss.get("rf_best3_imputer")

    # Asegurar índice 'Indice'
    df = df_in.copy()
    if "Indice" in df.columns and df.index.name != "Indice":
        df = df.set_index("Indice", drop=False)

    # Convertir ADL a numérico ("" -> NaN)
    adl_all = st.session_state.get("ind_adl_cols", st.session_state.get("ind_cols", []))
    for c in adl_all:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    pred = pd.Series(index=df.index, dtype="object")

    # Helper para predecir con un modelo y sus columnas
    def _predict_block(cols, model, le, imputer, mask_limit=None):
        if not cols or model is None or le is None:
            return pd.Index([]), None
        if not all(c in df.columns for c in cols):
            return pd.Index([]), None
        mask = df[cols].notna().all(axis=1)
        if mask_limit is not None:
            mask = mask & mask_limit
        if not mask.any():
            return pd.Index([]), None
        X = df.loc[mask, cols].apply(pd.to_numeric, errors="coerce")
        if imputer is not None:
            X = imputer.transform(X)
        yhat = model.predict(X)
        labels = le.inverse_transform(yhat)
        return df.index[mask], labels

    # 1) Modelo de 4 variables
    if have4:
        idx4, lab4 = _predict_block(cols4, model4, le4, imp4)
        if len(idx4) > 0:
            pred.loc[idx4] = lab4

    # 2) Modelo de 3 variables (solo donde falte)
    if have3:
        restante = pred.isna()
        idx3, lab3 = _predict_block(cols3, model3, le3, imp3, mask_limit=restante)
        if idx3 is not None and len(idx3) > 0:
            pred.loc[idx3] = lab3

    # 3) Fallback de "regla" (donde siga faltando)
    if pred.isna().any():
        ind_cols = st.session_state.get("ind_cols", [])
        if ind_cols:
            regla = _riesgo_regla(df, ind_cols)
            pred.loc[pred.isna()] = regla.loc[pred.isna()]

    pred.fillna("Sin datos", inplace=True)
    return pd.DataFrame({"nivel_riesgo_pred": pred})

# ==========================================================
# UI: Tabs de captura manual y subir Excel
# ==========================================================
st.markdown("## 📋 Diagnóstico por captura o carga de archivo")

manual_adl_cols = st.session_state.get("ind_adl_cols", st.session_state.get("ind_cols", []))
if not manual_adl_cols:
    # Fallback de ejemplo si aún no se calcularon ADL/ind_cols
    manual_adl_cols = ["H11", "H15A", "H5", "H6", "C37"]

tabs = st.tabs(["✍️ Captura manual", "📄 Subir Excel"])

# ==========================================================
# TAB 1: Captura manual (edición dinámica, recálculo in-place)
# ==========================================================
with tabs[0]:
    st.markdown("Ingresa pacientes manualmente (puedes agregar/eliminar filas). Luego presiona **Recalcular diagnósticos**.")

    # Inicializa tabla en sesión si no existe
    if "manual_df" not in st.session_state:
        base_cols = ["Indice", "Sexo", "Edad"] + manual_adl_cols
        st.session_state["manual_df"] = pd.DataFrame([{
            "Indice": "",
            "Sexo": "",
            "Edad": ""
        } | {c: "" for c in manual_adl_cols}], columns=base_cols)

    # Editor dinámico (NO predice aún)
    edited = st.data_editor(
        st.session_state["manual_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Sexo": st.column_config.SelectboxColumn(options=["", "M", "F"], help="Opcional"),
            "Edad": st.column_config.NumberColumn(min_value=0, max_value=120, step=1, help="Opcional"),
            **{c: st.column_config.NumberColumn(min_value=0, max_value=2, step=1, help="Valores esperados: 0/1/2") for c in manual_adl_cols}
        },
        key="manual_editor"
    )
    # 🔄 Sincroniza SIEMPRE lo editado
    st.session_state["manual_df"] = edited.copy()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Añadir fila vacía", use_container_width=True):
            nueva = {k: "" for k in st.session_state["manual_df"].columns}
            st.session_state["manual_df"] = pd.concat(
                [st.session_state["manual_df"], pd.DataFrame([nueva])],
                ignore_index=True
            )
            st.rerun()
    with c2:
        recalcular = st.button("Recalcular diagnósticos", use_container_width=True, type="primary")

    if recalcular:
        df_man = st.session_state["manual_df"].copy()

        # Asegurar índice 'Indice'
        if "Indice" not in df_man.columns or df_man["Indice"].isna().all() or (df_man["Indice"] == "").all():
            df_man["Indice"] = [f"row_{i+1}" for i in range(len(df_man))]
        df_man.set_index("Indice", inplace=True, drop=False)

        # Asegurar columnas que usan los modelos
        need4 = st.session_state.get("rf_best4_cols", [])
        need3 = st.session_state.get("rf_best3_cols", [])
        needed = set(need4) | set(need3)
        for c in needed:
            if c not in df_man.columns:
                df_man[c] = np.nan

        # Convertir ADL a numérico ("" -> NaN)
        for c in manual_adl_cols:
            if c in df_man.columns:
                df_man[c] = pd.to_numeric(df_man[c], errors="coerce")

        # Diagnóstico (modelos + regla)
        df_pred = predecir_con_modelos(df_man)
        out = df_man.join(df_pred, how="left")

        total = len(out)
        sin_datos = (out["nivel_riesgo_pred"] == "Sin datos").sum()
        if sin_datos > 0:
            st.info(
                f"{sin_datos} de {total} fila(s) quedaron como **'Sin datos'**. "
                "Completa **al menos** las 4 variables del mejor modelo (o 3 del secundario) "
                "o todas las usadas en indiscernibilidad para que la **regla** aplique."
            )

        st.subheader("Resultados (captura manual)")
        st.dataframe(out.reset_index(drop=True), use_container_width=True)

        st.download_button(
            "Descargar diagnósticos (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="diagnosticos_manual.csv",
            mime="text/csv",
            key="dl_diag_manual"
        )

# ==========================================================
# TAB 2: Subir Excel (normaliza columnas y permite edición)
# ==========================================================
with tabs[1]:
    st.markdown("Sube un **Excel (.xlsx)** o **CSV** con columnas de ADL. Se usarán los modelos entrenados y, si falta info, la regla.")

    up = st.file_uploader("Archivo Excel/CSV", type=["xlsx", "xls", "csv"], accept_multiple_files=False, key="up_excel_rf")
    if up is not None:
        # Cargar archivo
        try:
            if up.name.lower().endswith(".csv"):
                df_up = pd.read_csv(up)
            else:
                df_up = pd.read_excel(up)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            df_up = None

        if df_up is not None and not df_up.empty:
            # Normalizar nombres ADL (H11_18 -> H11, etc.)
            df_up = _normalize_adl_columns(df_up, manual_adl_cols)

            # Asegurar columnas mínimas
            base_cols = ["Indice", "Sexo", "Edad"]
            for c in base_cols:
                if c not in df_up.columns:
                    df_up[c] = "" if c != "Edad" else np.nan

            # Editor (permite corregir antes de calcular)
            edited_up = st.data_editor(
                df_up,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Sexo": st.column_config.SelectboxColumn(options=["", "M", "F"], help="Opcional"),
                    "Edad": st.column_config.NumberColumn(min_value=0, max_value=120, step=1, help="Opcional"),
                    **{c: st.column_config.NumberColumn(min_value=0, max_value=2, step=1, help="Valores esperados: 0/1/2") for c in manual_adl_cols if c in df_up.columns}
                },
                key="excel_editor"
            )
            st.session_state["excel_df"] = edited_up.copy()

            if st.button("Calcular diagnósticos (archivo)", type="primary", use_container_width=True, key="btn_calc_excel"):
                df_file = st.session_state["excel_df"].copy()

                # Asegurar índice
                if "Indice" not in df_file.columns or df_file["Indice"].isna().all() or (df_file["Indice"] == "").all():
                    df_file["Indice"] = [f"row_{i+1}" for i in range(len(df_file))]
                df_file.set_index("Indice", inplace=True, drop=False)

                # Convertir ADL a numérico
                for c in manual_adl_cols:
                    if c in df_file.columns:
                        df_file[c] = pd.to_numeric(df_file[c], errors="coerce")

                # Asegurar columnas de modelos
                need4 = st.session_state.get("rf_best4_cols", [])
                need3 = st.session_state.get("rf_best3_cols", [])
                needed = set(need4) | set(need3)
                for c in needed:
                    if c not in df_file.columns:
                        df_file[c] = np.nan

                # Diagnóstico
                df_pred = predecir_con_modelos(df_file)
                outf = df_file.join(df_pred, how="left")

                total = len(outf)
                sin_datos = (outf["nivel_riesgo_pred"] == "Sin datos").sum()
                if sin_datos > 0:
                    st.info(
                        f"{sin_datos} de {total} fila(s) quedaron como **'Sin datos'**. "
                        "Completa 4 (o 3) ADL según el modelo, o todas las de indiscernibilidad para que aplique la regla."
                    )

                st.subheader("Resultados (archivo cargado)")
                st.dataframe(outf.reset_index(drop=True), use_container_width=True)

                st.download_button(
                    "Descargar diagnósticos (CSV)",
                    data=outf.to_csv(index=False).encode("utf-8"),
                    file_name="diagnosticos_archivo.csv",
                    mime="text/csv",
                    key="dl_diag_excel"
                )
    else:
        st.caption("Formato esperado: columnas **Indice, Sexo, Edad** (opcionales) + columnas ADL (H11, H15A, H5, H6, C37, etc.).")

