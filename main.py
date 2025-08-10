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



# HAsta aqui el filtrado

# =========================
# Indiscernibilidad + resumen + pastel + radar
# =========================
import numpy as np
import matplotlib.pyplot as plt

# --- Funciones (sin sufijos, seguras) ---
def indiscernibility(attr, table: pd.DataFrame):
    """Clases de indiscernibilidad como lista de sets de índices, ordenadas por tamaño desc."""
    u_ind = {}
    for i in table.index:
        # clave como tupla para evitar colisiones ("1","23") vs ("12","3")
        key = tuple(table.loc[i, a] for a in attr)
        u_ind.setdefault(key, set()).add(i)
    return sorted(u_ind.values(), key=len, reverse=True)

def lower_approximation(R, X):
    """Aproximación inferior: une los bloques de R contenidos en algún conjunto de X."""
    l_approx = set()
    for x in X:
        for r in R:
            if r.issubset(x):
                l_approx.update(r)
    return l_approx

def upper_approximation(R, X):
    """Aproximación superior: une los bloques de R que intersectan algún conjunto de X."""
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

# --- Controles en barra lateral ---
with st.sidebar:
    st.subheader("Indiscernibilidad")
    # Sugerencia por defecto (solo las que existan)
    sugeridas = [c for c in ["C37", "H11", "H15A", "H5", "H6"] if c in df_base_ind.columns]
    cols_attrs = st.multiselect(
        "Atributos (columnas) para agrupar",
        options=list(df_base_ind.columns),
        default=sugeridas,
        help="Se generarán clases de indiscernibilidad con la combinación exacta de estos atributos."
    )
    min_size_for_pie = st.number_input(
        "Tamaño mínimo de clase para incluir en el pastel",
        min_value=2, max_value=100000, value=30, step=1
    )
    top_n_radar = st.number_input(
        "N conjuntos más numerosos para radar",
        min_value=1, max_value=100, value=15, step=1
    )
    generar = st.button("Calcular indiscernibilidad")

if generar:
    if not cols_attrs:
        st.warning("Selecciona al menos una columna para indiscernibilidad.")
    else:
        # Asegurar que las columnas sean numéricas si procede (no obligatorio)
        df_ind = df_base_ind.copy()
        for c in cols_attrs:
            df_ind[c] = pd.to_numeric(df_ind[c], errors="ignore")

        # 1) Calcular clases
        clases = indiscernibility(cols_attrs, df_ind)
        if not clases:
            st.warning("No se formaron clases (verifica columnas seleccionadas).")
        else:
            st.success(f"Se formaron {len(clases)} clases de indiscernibilidad.")

            # 2) Resumen de tamaños
            longitudes = [(i, len(s)) for i, s in enumerate(clases) if len(s) >= 1]
            longitudes_orden = sorted(longitudes, key=lambda x: x[1], reverse=True)
            nombres = {idx: f"Conjunto {k+1}" for k, (idx, _) in enumerate(longitudes_orden)}

            resumen_df = pd.DataFrame({
                "Conjunto": [nombres[i] for i, _ in longitudes_orden],
                "Tamaño":   [tam for _, tam in longitudes_orden]
            })
            st.subheader("Resumen de clases (ordenadas por tamaño)")
            st.dataframe(resumen_df, use_container_width=True)

            # 3) Pastel de clases con tamaño >= umbral
            candidatas = [(nombres[i], tam) for i, tam in longitudes_orden if tam >= min_size_for_pie]
            if candidatas:
                labels = [n for n, _ in candidatas]
                valores = [v for _, v in candidatas]
                total = sum(valores)
                fig_pie, ax_pie = plt.subplots(figsize=(7, 7))
                ax_pie.pie(valores, labels=labels, autopct=lambda p: f"{p:.1f}%\n({int(p*total/100):,})", startangle=140)
                ax_pie.axis('equal')
                ax_pie.set_title(f"Participación de clases (≥ {min_size_for_pie} filas)")
                st.pyplot(fig_pie)
            else:
                st.info(f"No hay clases con tamaño ≥ {min_size_for_pie} para el pastel.")

            # 4) Radar plots para los N conjuntos más numerosos
            #    - usaremos las columnas seleccionadas como ejes del radar
            st.subheader("Radar de los conjuntos más numerosos")
            # Preparar top-N
            top_idxs = [i for i, _ in longitudes_orden[:int(top_n_radar)]]
            top_sets = [(nombres[i], clases[i]) for i in top_idxs]

            # Utilidad para color según cantidad de "1" en la primera fila del conjunto
            def determinar_color(valores):
                count_ones = sum(1 for v in valores if pd.to_numeric(v, errors="coerce") == 1)
                if count_ones == 0:
                    return 'blue'
                elif 1 <= count_ones < 3:
                    return 'green'
                elif count_ones == 3:
                    return 'yellow'
                elif 4 <= count_ones < 5:
                    return 'orange'
                else:
                    return 'red'

            total_pacientes = len(df_ind)

            # Calcular rejilla para mostrar top_n_radar
            n = int(top_n_radar)
            cols_grid = 5
            rows_grid = int(np.ceil(n / cols_grid))
            fig, axs = plt.subplots(rows_grid, cols_grid, figsize=(cols_grid*6, rows_grid*5), subplot_kw=dict(polar=True))
            axs = np.atleast_2d(axs)
            fig.subplots_adjust(hspace=0.8, wspace=0.6)

            # Ángulos del radar
            k = len(cols_attrs)
            angulos = np.linspace(0, 2 * np.pi, k, endpoint=False).tolist()
            angulos_cerrado = angulos + angulos[:1]

            # Dibujar cada conjunto
            for idx_plot in range(rows_grid * cols_grid):
                r = idx_plot // cols_grid
                c = idx_plot % cols_grid
                ax = axs[r, c]

                if idx_plot >= n:
                    ax.axis('off')
                    continue

                nombre, conjunto_idx = top_sets[idx_plot]
                indices = sorted(list(conjunto_idx))
                df_conj = df_ind.loc[indices, cols_attrs]

                # Tomar solo la primera fila como representación (como en tu código original)
                if df_conj.empty:
                    valores = [0]*k
                    num_filas_df = 0
                else:
                    valores = df_conj.iloc[0].tolist()
                    num_filas_df = len(df_conj)

                # Cierre del polígono
                valores_cerrados = list(valores) + [valores[0]]

                # Color
                color = determinar_color(valores)

                ax.plot(angulos_cerrado, valores_cerrados, color=color)
                ax.fill(angulos_cerrado, valores_cerrados, color=color, alpha=0.25)
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_xticks(angulos)
                ax.set_xticklabels(cols_attrs, fontsize=10)
                ax.yaxis.grid(True)
                ax.set_ylim(0, 2)           # escala 0..2 como en tu versión
                ax.set_yticks([0, 1, 2])
                ax.set_yticklabels([0, 1, 2], fontsize=9)

                porcentaje = (num_filas_df / total_pacientes * 100) if total_pacientes else 0.0
                ax.set_title(nombre, fontsize=12)
                ax.text(0.5, -0.2, f"Filas: {num_filas_df} ({porcentaje:.2f}%)",
                        transform=ax.transAxes, ha="center", va="center", fontsize=10)

            st.pyplot(fig)


# ============ NUEVO: Gráfico compuesto (Pastel + radares incrustados) ============
from matplotlib.patches import ConnectionPatch
from math import log1p

# Conjuntos para el pastel compuesto (mismo umbral que arriba)
candidatas_idx_nom_tam = [(i, nombres[i], tam) for i, tam in longitudes_orden if tam >= int(min_size_for_pie)]
if not candidatas_idx_nom_tam:
    st.info(f"No hay clases con tamaño ≥ {min_size_for_pie} para el gráfico compuesto.")
else:
    # Datos base para pastel y radares
    nombres_dataframes = [nom for _, nom, _ in candidatas_idx_nom_tam]
    tamanios = [tam for _, _, tam in candidatas_idx_nom_tam]
    total_incluido = sum(tamanios)
    porcentajes = [(nom, (tam/total_incluido*100.0) if total_incluido else 0.0)
                   for _, nom, tam in candidatas_idx_nom_tam]

    valores_dataframes, colores_dataframes = [], []
    for idx, _, _ in candidatas_idx_nom_tam:
        indices = sorted(list(clases[idx]))
        sub = df_ind.loc[indices, cols_attrs]
        vals = sub.iloc[0].tolist() if not sub.empty else [0]*len(cols_attrs)
        valores_dataframes.append(vals)
        colores_dataframes.append(determinar_color(vals))

    # Parámetros del compuesto
    min_radio = 1.0
    max_radio = 2.40
    radar_size_min = 0.10
    radar_size_max = 0.19
    etiquetas_radar = [et.replace('_21','').replace('_18','') for et in cols_attrs]

    # Figura y pastel
    fig_comp = plt.figure(figsize=(16, 16))
    main_ax = plt.subplot(111)
    main_ax.set_position([0.1, 0.1, 0.8, 0.8])

    if porcentajes:
        nombres_legibles, valores_porcentajes = zip(*porcentajes)
        valores_porcentajes = [float(p) for p in valores_porcentajes]
    else:
        nombres_legibles, valores_porcentajes = [], []

    colores_ajustados = colores_dataframes[:len(valores_porcentajes)]
    wedges, texts, autotexts = main_ax.pie(
        valores_porcentajes,
        colors=colores_ajustados,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 17},
        labeldistance=1.1
    )

    # Posiciones para radares
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

        # Separación angular suave
        angulos_rad_separados = angulos_rad.copy()
        min_sep = np.deg2rad(7)
        for i in range(1, len(angulos_rad_separados)):
            while abs(angulos_rad_separados[i] - angulos_rad_separados[i-1]) < min_sep:
                angulos_rad_separados[i] += min_sep/2

        # Radares incrustados + conexiones
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

            radar_ax.set_theta_offset(np.pi/2)
            radar_ax.set_theta_direction(-1)
            radar_ax.plot(angs_c, vals_c, color=color)
            radar_ax.fill(angs_c, vals_c, color=color, alpha=0.3)
            radar_ax.set_xticks(angs)
            radar_ax.set_xticklabels(etiquetas_radar, fontsize=13)
            radar_ax.set_yticks([0,1,2])
            radar_ax.set_yticklabels(['0','1','2'], fontsize=11)
            radar_ax.set_ylim(0,2)
            radar_ax.yaxis.grid(True, linestyle='dotted', linewidth=0.5)

            x0 = 0.5 + 0.3*np.cos(ang_rad)
            y0 = 0.5 + 0.3*np.sin(ang_rad)
            con = ConnectionPatch(
                xyA=(x0, y0), coordsA=fig_comp.transFigure,
                xyB=(x, y), coordsB=fig_comp.transFigure,
                color='gray', lw=0.8, linestyle='--'
            )
            fig_comp.add_artist(con)

    st.pyplot(fig_comp)
    try:
        plt.savefig("radar_pastel_final.png", dpi=300, bbox_inches='tight', facecolor='white')
        st.download_button(
            "Descargar imagen (PNG)",
            data=open("radar_pastel_final.png", "rb").read(),
            file_name="radar_pastel_final.png",
            mime="image/png"
        )
    except Exception:
        pass
# ====================================================================

