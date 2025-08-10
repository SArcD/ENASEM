import streamlit as st
import pandas as pd

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

# --- Interfaz Streamlit ---
st.title("Cargar y procesar ENASEM 2018")

# Columnas específicas
columnas_deseadas = [
    "AGE_18",'SEX_18','C4_18','C6_18', 'C12_18', 'C19_18', 'C22A_18', 'C26_18', "C32_18", 'C37_18',
    "C49_1_18",'C49_2_18','C49_8_18','C64_18','C66_18', 'C67_1_18','C67_2_18','C68E_18', 'C68G_18', 'C68H_18', 'C69A_18', 'C69B_18',
    'C71A_18', 'C76_18', 'H1_18','H4_18', 'H5_18',
    'H6_18', 'H8_18', 'H9_18', 'H10_18', 'H11_18', 'H12_18',
    'H13_18', 'H15A_18', 'H15B_18', 'H15D_18', 'H16A_18',
    'H16D_18', 'H17A_18', 'H17D_18', 'H18A_18', 'H18D_18', 'H19A_18',
    'H19D_18'
]

# Subir archivo
archivo = st.file_uploader("Sube el archivo CSV de ENASEM 2018", type=["csv"])

df = None
if archivo is not None:
    try:
        # Leer solo columnas deseadas
        datos_seleccionados = pd.read_csv(archivo, usecols=columnas_deseadas)

        # Combinar columnas de estatura
        datos_seleccionados['C67_18'] = datos_seleccionados['C67_1_18'] + datos_seleccionados['C67_2_18'] / 100
        datos_seleccionados = datos_seleccionados.drop(columns=['C67_1_18', 'C67_2_18'])

        # Agregar índice
        datos_seleccionados['Indice'] = datos_seleccionados.index

        # Reordenar columnas
        columnas_finales = ['Indice',"AGE_18",'SEX_18','C4_18','C6_18', 'C12_18', 'C19_18', 'C22A_18', 'C26_18', "C32_18", 'C37_18',
            "C49_1_18",'C49_2_18','C49_8_18','C64_18','C66_18', 'C67_18','C68E_18', 'C68G_18', 'C68H_18', 'C69A_18', 'C69B_18',
            'C71A_18', 'C76_18', 'H1_18', 'H4_18', 'H5_18',
            'H6_18', 'H8_18', 'H9_18', 'H10_18', 'H11_18', 'H12_18',
            'H13_18', 'H15A_18', 'H15B_18', 'H15D_18', 'H16A_18',
            'H16D_18', 'H17A_18', 'H17D_18', 'H18A_18', 'H18D_18', 'H19A_18',
            'H19D_18'
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
