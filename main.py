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

#            # 3) Pastel (usando tamaños de df_eval)
#            candidatas = [(nombres[i], tam) for i, tam in longitudes_orden if tam >= int(min_size_for_pie)]
#            if candidatas:
#                labels = [n for n, _ in candidatas]
#                valores = [v for _, v in candidatas]
#                total = sum(valores)
#                fig_pie, ax_pie = plt.subplots(figsize=(7, 7))
#                ax_pie.pie(valores, labels=labels,
#                           autopct=lambda p: f"{p:.1f}%\n({int(p*total/100):,})",
#                           startangle=140)
#                ax_pie.axis('equal')
#                ax_pie.set_title(f"Participación de clases (≥ {min_size_for_pie} filas)")
#                st.pyplot(fig_pie)
#            else:
#                st.info(f"No hay clases con tamaño ≥ {min_size_for_pie} para el pastel.")
#
#            # ====== NUEVO (pegar aquí, después del pastel; dentro de if generar:) ======
#            # df_ind existe en este bloque; usamos solo filas SIN NaN en las columnas seleccionadas
#            #df_eval = df_ind.loc[df_ind[cols_attrs].dropna().index].copy()
#
#            # Guardarlo en sesión por si lo necesitas en otras secciones
#            st.session_state["df_eval"] = df_eval.copy()

#            # Calcular nivel de riesgo según la regla:
#            # - Riesgo nulo: TODAS las columnas seleccionadas valen 2
#            # - Riesgo leve: 1 o 2 columnas valen 1  (o 0 si no son todas 2)
#            # - Riesgo moderado: exactamente 3 columnas valen 1
#            # - Riesgo severo: 4 o 5 columnas valen 1
#            vals = df_eval[cols_attrs].apply(pd.to_numeric, errors="coerce")
#            count_ones = (vals == 1).sum(axis=1)
#            all_twos   = (vals == 2).all(axis=1)

#            nivel = np.where(
#                all_twos, "Riesgo nulo",
#                np.where(
#                    count_ones <= 2,
#                    np.where(count_ones >= 1, "Riesgo leve", "Riesgo leve"),
#                    np.where(count_ones == 3, "Riesgo moderado", "Riesgo severo")
#                )
#            )
#
#            df_eval_riesgo = df_eval.copy()
#            df_eval_riesgo["nivel_riesgo"] = nivel
#
#            st.subheader("Filas usadas en el pastel (sin NaN) + nivel_riesgo")
#            st.dataframe(df_eval_riesgo.reset_index(), use_container_width=True)
#
#            st.download_button(
#                "Descargar filas del pastel con nivel_riesgo (CSV)",
#                data=df_eval_riesgo.reset_index().to_csv(index=False).encode("utf-8"),
#                file_name="filas_pastel_con_nivel_riesgo.csv",
#                mime="text/csv",
#                key="dl_df_eval_riesgo"
#            )
#
#            # Por si más adelante quieres reutilizarlo
#            st.session_state["df_eval_riesgo"] = df_eval_riesgo.copy()
#            # ====== FIN NUEVO ======



            
#            # 4) Radar de los N conjuntos más grandes (sobre df_eval)
#            st.subheader("Radar de los conjuntos más numerosos")
#            top_idxs = [i for i, _ in longitudes_orden[:int(top_n_radar)]]
#            top_sets = [(nombres[i], clases[i]) for i in top_idxs]

#            def determinar_color(valores):
#                count_ones = sum(1 for v in valores if pd.to_numeric(v, errors="coerce") == 1)
#                if count_ones == 0:
#                    return 'blue'
#                elif 1 <= count_ones < 3:
#                    return 'green'
#                elif count_ones == 3:
#                    return 'yellow'
#                elif 4 <= count_ones < 5:
#                    return 'orange'
#                else:
#                    return 'red'

#            total_pacientes = len(df_eval)
#            n = int(top_n_radar)
#            cols_grid = 5
#            rows_grid = int(np.ceil(n / cols_grid))
#            fig, axs = plt.subplots(rows_grid, cols_grid, figsize=(cols_grid*6, rows_grid*5), subplot_kw=dict(polar=True))
#            axs = np.atleast_2d(axs)
#            fig.subplots_adjust(hspace=0.8, wspace=0.6)
#
#            k = len(cols_attrs)
#            angulos = np.linspace(0, 2 * np.pi, k, endpoint=False).tolist()
#            angulos_cerrado = angulos + angulos[:1]

#            for idx_plot in range(rows_grid * cols_grid):
#                r = idx_plot // cols_grid
#                c = idx_plot % cols_grid
#                ax = axs[r, c]
#                if idx_plot >= n:
#                    ax.axis('off')
#                    continue

#                nombre, conjunto_idx = top_sets[idx_plot]
#                indices = sorted(list(conjunto_idx))
#                df_conj = df_eval.loc[indices, cols_attrs]

#                if df_conj.empty:
#                    valores = [0]*k
#                    num_filas_df = 0
#                else:
#                    valores = df_conj.iloc[0].tolist()
#                    num_filas_df = len(df_conj)

#                valores_cerrados = list(valores) + [valores[0]]
#                color = determinar_color(valores)

#                ax.plot(angulos_cerrado, valores_cerrados, color=color)
#                ax.fill(angulos_cerrado, valores_cerrados, color=color, alpha=0.25)
#                ax.set_theta_offset(np.pi / 2)
#                ax.set_theta_direction(-1)
#                ax.set_xticks(angulos)
#                ax.set_xticklabels(cols_attrs, fontsize=10)
#                ax.yaxis.grid(True)
#                ax.set_ylim(0, 2)
#                ax.set_yticks([0, 1, 2])
#                ax.set_yticklabels([0, 1, 2], fontsize=9)

#                porcentaje = (num_filas_df / total_pacientes * 100) if total_pacientes else 0.0
#                ax.set_title(nombre, fontsize=12)
#                ax.text(0.5, -0.2, f"Filas: {num_filas_df} ({porcentaje:.2f}%)",
#                        transform=ax.transAxes, ha="center", va="center", fontsize=10)
#
#            st.pyplot(fig)

#            # ============ Gráfico compuesto (pastel + radares incrustados) ============
#            candidatas_idx_nom_tam = [(i, nombres[i], tam) for i, tam in longitudes_orden if tam >= int(min_size_for_pie)]
#            if candidatas_idx_nom_tam:
#                nombres_dataframes = [nom for _, nom, _ in candidatas_idx_nom_tam]
#                tamanios = [tam for _, _, tam in candidatas_idx_nom_tam]
#                total_incluido = sum(tamanios)
#                porcentajes = [(nom, (tam/total_incluido*100.0) if total_incluido else 0.0)
#                               for _, nom, tam in candidatas_idx_nom_tam]

#                valores_dataframes, colores_dataframes = [], []
#                for idx, _, _ in candidatas_idx_nom_tam:
 #                   indices = sorted(list(clases[idx]))
#                    sub = df_eval.loc[indices, cols_attrs]
#                    vals = sub.iloc[0].tolist() if not sub.empty else [0]*len(cols_attrs)
#                    valores_dataframes.append(vals)
#                    colores_dataframes.append(determinar_color(vals))

#                min_radio = 1.0
#                max_radio = 2.40
#                radar_size_min = 0.10
#                radar_size_max = 0.19
#                etiquetas_radar = [et.replace('_21','').replace('_18','') for et in cols_attrs]

#                fig_comp = plt.figure(figsize=(16, 16))
#                main_ax = plt.subplot(111)
#                main_ax.set_position([0.1, 0.1, 0.8, 0.8])

#                if porcentajes:
#                    _, valores_porcentajes = zip(*porcentajes)
#                    valores_porcentajes = [float(p) for p in valores_porcentajes]
#                else:
#                    valores_porcentajes = []

#                colores_ajustados = colores_dataframes[:len(valores_porcentajes)]
#                wedges, texts, autotexts = main_ax.pie(
#                    valores_porcentajes,
#                    colors=colores_ajustados,
#                    autopct='%1.1f%%',
#                    startangle=90,
#                    textprops={'fontsize': 17},
#                    labeldistance=1.1
#                )
#
#                if wedges:
#                    angulos_pastel = [(w.theta1 + w.theta2)/2 for w in wedges]
#                    anchos = [abs(w.theta2 - w.theta1) for w in wedges]
#                    max_ancho = max(anchos) if anchos else 1
#                    angulos_rad = [np.deg2rad(a) for a in angulos_pastel]
#
#                    radios_personalizados = [
#                        min_radio + (1 - (log1p(a)/log1p(max_ancho))) * (max_radio - min_radio)
#                        for a in anchos
#                    ]
#                    tamaños_radar = [
#                        radar_size_min + (a/max_ancho) * (radar_size_max - radar_size_min)
#                        for a in anchos
#                    ]

#                    angulos_rad_separados = angulos_rad.copy()
#                    min_sep = np.deg2rad(7)
#                    for i in range(1, len(angulos_rad_separados)):
#                        while abs(angulos_rad_separados[i] - angulos_rad_separados[i-1]) < min_sep:
#                            angulos_rad_separados[i] += min_sep/2
#
#                    for i, (nombre, vals, color, ang_rad, r_inset, tam_radar) in enumerate(
#                        zip(nombres_dataframes, valores_dataframes, colores_dataframes,
#                            angulos_rad_separados, radios_personalizados, tamaños_radar)
#                    ):
#                        factor_alejamiento = 2.3
#                        x = 0.5 + r_inset*np.cos(ang_rad)/factor_alejamiento
#                        y = 0.5 + r_inset*np.sin(ang_rad)/factor_alejamiento
#                        radar_ax = fig_comp.add_axes([x - tam_radar/2, y - tam_radar/2, tam_radar, tam_radar], polar=True)
#
#                        vals = list(vals)[:len(cols_attrs)] or [0]*len(cols_attrs)
#                        vals_c = vals + [vals[0]]
#                        angs = np.linspace(0, 2*np.pi, len(cols_attrs), endpoint=False).tolist()
#                        angs_c = angs + [angs[0]]
#
#                        radar_ax.set_theta_offset(np.pi/2)
#                        radar_ax.set_theta_direction(-1)
#                        radar_ax.plot(angs_c, vals_c, color=color)
#                        radar_ax.fill(angs_c, vals_c, color=color, alpha=0.3)
#                        radar_ax.set_xticks(angs)
#                        radar_ax.set_xticklabels(etiquetas_radar, fontsize=13)
#                        radar_ax.set_yticks([0,1,2])
#                        radar_ax.set_yticklabels(['0','1','2'], fontsize=11)
#                        radar_ax.set_ylim(0,2)
#                        radar_ax.yaxis.grid(True, linestyle='dotted', linewidth=0.5)

#                        x0 = 0.5 + 0.3*np.cos(ang_rad)
#                        y0 = 0.5 + 0.3*np.sin(ang_rad)
#                        con = ConnectionPatch(
#                            xyA=(x0, y0), coordsA=fig_comp.transFigure,
#                            xyB=(x, y), coordsB=fig_comp.transFigure,
#                            color='gray', lw=0.8, linestyle='--'
#                        )
#                        fig_comp.add_artist(con)

#                st.pyplot(fig_comp)
#                try:
#                    plt.savefig("radar_pastel_final.png", dpi=300, bbox_inches='tight', facecolor='white')
#                    st.download_button(
#                        "Descargar imagen (PNG)",
#                        data=open("radar_pastel_final.png", "rb").read(),
#                        file_name="radar_pastel_final.png",
#                        mime="image/png"
#                    )
#                except Exception:
#                    pass

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

        # =========================
# Predicción en TODO el DF indiscernible (no solo el pastel)
# =========================
ss = st.session_state
req = ("rf_best4", "rf_best4_cols", "rf_best4_le", "rf_best4_imp", "ind_df")
if not all(k in ss for k in req):
    st.info("Entrena primero el modelo (best4) para habilitar la predicción completa.")
else:
    st.subheader("Predicción en todo el DataFrame (indiscernible)")

    # 1) DF completo (ADL, index='Indice'; puede tener NaN)
    df_all = ss["ind_df"].copy()

    # 2) Preparar X con las columnas del modelo; crear faltantes y convertir a numérico
    feat = ss["rf_best4_cols"]
    for c in feat:
        if c not in df_all.columns:
            df_all[c] = np.nan
    X_all = df_all[feat].apply(pd.to_numeric, errors="coerce")

    # Imputar con la MISMA mediana del train
    X_all_imp = ss["rf_best4_imp"].transform(X_all)

    # 3) Predecir
    ypred_all = ss["rf_best4"].predict(X_all_imp)
    labels_all = ss["rf_best4_le"].inverse_transform(ypred_all)

    df_pred_all = df_all.copy()
    df_pred_all["nivel_riesgo_pred"] = labels_all
    ss["df_pred_all_rf"] = df_pred_all

    st.dataframe(df_pred_all.reset_index().head(50), use_container_width=True)
    st.download_button(
        "Descargar predicciones RF (todo ind_df) CSV",
        data=df_pred_all.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="predicciones_rf_todo_ind_df.csv",
        mime="text/csv",
        key="dl_pred_all_rf"
    )

    # =========================
    # Comparativa: RF pastel vs RF todo ind_df
    # =========================
    orden = ["Riesgo nulo", "Riesgo leve", "Riesgo moderado", "Riesgo severo"]

    # A) Distribución RF en pastel (si existe)
    df_pastel_pred = ss.get("df_pred_pastel_rf")
    if isinstance(df_pastel_pred, pd.DataFrame) and "nivel_riesgo_pred" in df_pastel_pred.columns:
        dist_rf_pastel = (
            df_pastel_pred["nivel_riesgo_pred"]
            .value_counts()
            .reindex(orden, fill_value=0)
        )
    else:
        dist_rf_pastel = pd.Series([0]*len(orden), index=orden)

    # B) Distribución RF en todo ind_df
    dist_rf_all = (
        df_pred_all["nivel_riesgo_pred"]
        .value_counts()
        .reindex(orden, fill_value=0)
    )

    x = np.arange(len(orden)); width = 0.38
    ymax = max(dist_rf_pastel.max(), dist_rf_all.max()) if len(orden) else 1

    fig_cmp, ax_cmp = plt.subplots(figsize=(8, 4.5))
    b1 = ax_cmp.bar(x - width/2, dist_rf_pastel.values, width, label="RF (solo pastel)")
    b2 = ax_cmp.bar(x + width/2, dist_rf_all.values,  width, label="RF (todo ind_df)")
    ax_cmp.set_xticks(x); ax_cmp.set_xticklabels(orden, rotation=0)
    ax_cmp.set_ylabel("Conteos"); ax_cmp.set_title("RF: pastel vs. todo ind_df")
    ax_cmp.legend(); ax_cmp.grid(axis='y', linestyle='--', alpha=0.3)

    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax_cmp.text(r.get_x()+r.get_width()/2, h + 0.01*ymax, f"{int(h)}",
                        ha="center", va="bottom", fontsize=9)
    st.pyplot(fig_cmp)

    # =========================
    # (Opcional) Comparar contra la "regla" en TODO el DF (solo filas sin NaN en ind_cols)
    # =========================
    ind_cols = ss.get("ind_cols", [])
    if ind_cols:
        df_all_regla = df_all.dropna(subset=ind_cols).copy()
        if not df_all_regla.empty:
            valr = df_all_regla[ind_cols].apply(pd.to_numeric, errors="coerce")
            ones = (valr == 1).sum(axis=1)
            twos = (valr == 2).all(axis=1)
            regla_all = np.where(
                twos, "Riesgo nulo",
                np.where(
                    ones <= 2,
                    np.where(ones >= 1, "Riesgo leve", "Riesgo leve"),
                    np.where(ones == 3, "Riesgo moderado", "Riesgo severo")
                )
            )
            dist_regla_all = pd.Series(regla_all).value_counts().reindex(orden, fill_value=0)

            ymax2 = max(dist_regla_all.max(), dist_rf_all.max()) if len(orden) else 1
            fig_cmp2, ax2 = plt.subplots(figsize=(8, 4.5))
            b1 = ax2.bar(x - width/2, dist_regla_all.values, width, label="Regla (todo ind_df sin NaN)")
            b2 = ax2.bar(x + width/2, dist_rf_all.values,    width, label="RF (todo ind_df)")
            ax2.set_xticks(x); ax2.set_xticklabels(orden)
            ax2.set_ylabel("Conteos"); ax2.set_title("Regla vs RF en todo ind_df")
            ax2.legend(); ax2.grid(axis='y', linestyle='--', alpha=0.3)
            for bars in (b1, b2):
                for r in bars:
                    h = r.get_height()
                    ax2.text(r.get_x()+r.get_width()/2, h + 0.01*ymax2, f"{int(h)}",
                             ha="center", va="bottom", fontsize=9)
            st.pyplot(fig_cmp2)


# === FINAL: Pasteles de distribución de riesgo ===
ss = st.session_state
if all(k in ss for k in ("rf_best3_model","rf_best3_cols","rf_label_encoder","ind_df")) and "df_eval_riesgo" in ss:
    # (aquí va el bloque de los dos gráficos de pastel)
    # =========================
    # Pasteles de riesgo: (1) sin NaN (regla) vs (2) todo ind_df (RF)
    # =========================
    ss = st.session_state
    need = ("ind_df", "ind_cols")
    if not all(k in ss for k in need):
        st.info("Primero calcula indiscernibilidad (para disponer de ind_df e ind_cols).")
    else:
        # ---------- 1) Filas SIN NaN en las ADL seleccionadas (regla) ----------
        df_all: pd.DataFrame = ss["ind_df"].copy()        # ADL con posibles NaN (index=Indice)
        cols_sel = ss["ind_cols"]

        # Subconjunto sin NaN SOLO en las columnas elegidas para ind
        df_no_nan = df_all.dropna(subset=cols_sel).copy()

        # Regla de riesgo sobre df_no_nan
        vals = df_no_nan[cols_sel].apply(pd.to_numeric, errors="coerce")
        count_ones = (vals == 1).sum(axis=1)
        all_twos   = (vals == 2).all(axis=1)

        riesgo_regla = np.where(
            all_twos, "Riesgo nulo",
            np.where(
                count_ones <= 2,
                np.where(count_ones >= 1, "Riesgo leve", "Riesgo leve"),
                np.where(count_ones == 3, "Riesgo moderado", "Riesgo considerable")
            )
        )
        s_regla = pd.Series(riesgo_regla, index=df_no_nan.index, name="nivel_riesgo_regla")

        # ---------- 2) TODO ind_df con modelos (RF 4-vars y 3-vars) ----------
        # Se usan solo si existen. Si no, avisamos.
        have4 = all(k in ss for k in ("rf_best4_model", "rf_best4_cols", "rf_label_encoder"))
        have3 = all(k in ss for k in ("rf_best3_model", "rf_best3_cols", "rf_label_encoder"))

        if not (have4 or have3):
            st.warning("No encuentro modelos entrenados (4/3 vars). Entrena los RF para generar el pastel global.")
            s_pred = pd.Series([], dtype="object")
        else:
            le = ss["rf_label_encoder"]
            pred_all = pd.Series(index=df_all.index, dtype="object")  # aquí caerán las etiquetas finales

            # — 4 variables (prioridad)
            if have4:
                cols4 = ss["rf_best4_cols"]
                # solo filas con todas las 4 presentes
                mask4 = df_all[cols4].notna().all(axis=1)
                if mask4.any():
                    X4 = df_all.loc[mask4, cols4].apply(pd.to_numeric, errors="coerce")
                    y4 = ss["rf_best4_model"].predict(X4)
                    pred_all.loc[mask4] = le.inverse_transform(y4)

            # — 3 variables (para las que no entraron arriba)
            if have3:
                cols3 = ss["rf_best3_cols"]
                mask_restante = pred_all.isna()
                if mask_restante.any():
                    m3 = df_all.loc[mask_restante, cols3].notna().all(axis=1)
                    idx3 = df_all.loc[mask_restante].index[m3]
                    if len(idx3) > 0:
                        X3 = df_all.loc[idx3, cols3].apply(pd.to_numeric, errors="coerce")
                        y3 = ss["rf_best3_model"].predict(X3)
                        pred_all.loc[idx3] = le.inverse_transform(y3)

            # Etiquetar las que quedaron sin poder predecir
            pred_all.fillna("Sin datos", inplace=True)
            s_pred = pred_all.rename("nivel_riesgo_pred")

        # ---------- Pasteles ----------
        orden = ["Riesgo nulo", "Riesgo leve", "Riesgo moderado", "Riesgo considerable"]
        # (para el segundo pastel añadimos 'Sin datos' si aparece)
        orden_full = orden + (["Sin datos"] if "Sin datos" in s_pred.values else [])

        # Conteos
        c1 = s_regla.value_counts().reindex(orden, fill_value=0)
        c2 = s_pred.value_counts().reindex(orden_full, fill_value=0)

        # 1) Pastel sin NaN (regla)
        fig1, ax1 = plt.subplots(figsize=(6.5, 6.5))
        vals1 = c1.values
        if vals1.sum() == 0:
            ax1.text(0.5, 0.5, "Sin filas válidas\n(sin NaN) para la regla",
                     ha="center", va="center", fontsize=12)
            ax1.axis("off")
        else:
            ax1.pie(vals1, labels=c1.index, autopct=lambda p: f"{p:.1f}%", startangle=120)
            ax1.axis('equal')
            ax1.set_title("Riesgo — solo filas sin NaN (regla)")

        st.pyplot(fig1)

        # 2) Pastel sobre todo ind_df (RF)
        fig2, ax2 = plt.subplots(figsize=(6.5, 6.5))
        vals2 = c2.values
        if vals2.sum() == 0:
            ax2.text(0.5, 0.5, "No hay predicciones RF disponibles",
                     ha="center", va="center", fontsize=12)
            ax2.axis("off")
        else:
            ax2.pie(vals2, labels=c2.index, autopct=lambda p: f"{p:.1f}%", startangle=120)
            ax2.axis('equal')
            ax2.set_title("Riesgo — todo ind_df (RF 4→3 vars)")

        st.pyplot(fig2)

        # Descargas opcionales
        out1 = pd.DataFrame({"Indice": s_regla.index, "nivel_riesgo_regla": s_regla.values})
        st.download_button(
            "Descargar niveles (regla, sin NaN)",
            data=out1.to_csv(index=False).encode("utf-8"),
            file_name="riesgo_regla_sin_nan.csv",
            mime="text/csv",
            key="dl_regla_sin_nan"
        )
        if len(s_pred) > 0:
            out2 = pd.DataFrame({"Indice": s_pred.index, "nivel_riesgo_pred": s_pred.values})
            st.download_button(
                "Descargar niveles (RF, todo ind_df)",
                data=out2.to_csv(index=False).encode("utf-8"),
                file_name="riesgo_rf_todo_ind_df.csv",
                mime="text/csv",
                key="dl_rf_todo"
            )

else:
    st.info("👉 Entrena el RF y calcula indiscernibilidad antes de mostrar los pasteles.")


