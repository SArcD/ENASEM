import streamlit as st
import pandas as pd
import re

# --- Función indiscernibilidad ---
def indiscernibility(attr, table: pd.DataFrame):
    u_ind = {}
    for i in table.index:
        key = tuple(table.loc[i, a] for a in attr)
        if key in u_ind:
            u_ind[key].add(i)
        else:
            u_ind[key] = {i}
    return sorted(u_ind.values(), key=len, reverse=True)

st.title("Cargar y procesar ENASEM (2018 o 2021)")

# Columnas base (sin sufijo de año)
columnas_deseadas_base = [
    "AGE",'SEX','C4','C6', 'C12', 'C19', 'C22A', 'C26', "C32", 'C37',
    "C49_1",'C49_2','C49_8','C64','C66', 'C67_1','C67_2','C68E', 'C68G', 'C68H', 'C69A', 'C69B',
    'C71A', 'C76', 'H1','H4', 'H5',
    'H6', 'H8', 'H9', 'H10', 'H11', 'H12',
    'H13', 'H15A', 'H15B', 'H15D', 'H16A',
    'H16D', 'H17A', 'H17D', 'H18A', 'H18D', 'H19A',
    'H19D'
]

# Subir archivo
archivo = st.file_uploader("Sube el archivo CSV de ENASEM 2018 o 2021", type=["csv"])

df = None
if archivo is not None:
    try:
        # Leer todo el archivo primero
        df_raw = pd.read_csv(archivo)

        # Quitar sufijo _18 o _21 de las columnas
        df_raw.columns = [re.sub(r'_(18|21)$', '', col) for col in df_raw.columns]

        # Filtrar columnas deseadas
        datos_seleccionados = df_raw[columnas_deseadas_base]

        # Combinar columnas de estatura
        datos_seleccionados['C67'] = datos_seleccionados['C67_1'] + datos_seleccionados['C67_2'] / 100
        datos_seleccionados = datos_seleccionados.drop(columns=['C67_1', 'C67_2'])

        # Agregar índice
        datos_seleccionados['Indice'] = datos_seleccionados.index

        # Reordenar columnas (ya sin sufijo de año)
        columnas_finales = ['Indice',"AGE",'SEX','C4','C6', 'C12', 'C19', 'C22A', 'C26', "C32", 'C37',
            "C49_1",'C49_2','C49_8','C64','C66', 'C67','C68E', 'C68G', 'C68H', 'C69A', 'C69B',
            'C71A', 'C76', 'H1', 'H4', 'H5',
            'H6', 'H8', 'H9', 'H10', 'H11', 'H12',
            'H13', 'H15A', 'H15B', 'H15D', 'H16A',
            'H16D', 'H17A', 'H17D', 'H18A', 'H18D', 'H19A',
            'H19D'
        ]
        datos_seleccionados = datos_seleccionados[columnas_finales]

        df = datos_seleccionados
        st.success("Archivo procesado correctamente ✅")
        st.dataframe(df.head(50))

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")

# --- Calcular indiscernibilidad ---
if df is not None:
    columnas = st.multiselect(
        "Selecciona columnas para indiscernibilidad:",
        options=list(df.columns)
    )
    if st.button("Calcular"):
        if columnas:
            clases = indiscernibility(columnas, df)
            st.write(f"Se encontraron {len(clases)} clases.")
            st.write([list(c) for c in clases[:5]])  # Muestra las 5 primeras clases
        else:
            st.warning("Selecciona al menos una columna.")
