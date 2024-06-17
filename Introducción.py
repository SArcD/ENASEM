import streamlit as st
import pandas as pd
import gdown
import io  # Asegúrate de importar el módulo io


# Enlaces a los archivos CSV en Google Drive
file_urls = {
    "ENASEM_2021_sec_a": "https://drive.google.com/uc?id=1OXrglgbqvwA1Oa2aMB5iLh9bMLJNo-uu",
    "ENASEM_2018_sec_a": "https://drive.google.com/uc?id=1pn8-1nCeVb8piMgad-7foAI9z1nmfqsO",
    "ENASEM_2021_sec_g": "https://drive.google.com/uc?id=1-u7LB4soK-g3w7Ll2qQ6cEqIwvjgGyxY",
    "ENASEM_2018_sec_g": "https://drive.google.com/uc?id=1t3jf686XhTDQmL1Tmhz5nSzQ9v7hRmZD"
}


# Función para descargar y cargar un archivo CSV desde Google Drive
def load_csv_from_drive(url):
    output = "temp.csv"
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# Función para convertir el dataframe a csv
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


# Función para convertir el dataframe a xlsx
def convert_df_to_xlsx(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data


# Crear una barra lateral para la selección de pestañas
st.sidebar.title("Navegación")
option = st.sidebar.selectbox("Seleccione una pestaña", ["Introducción", "Filtrar datos", "Buscador de variables", "Buscador de datos", "Relaciones de Indiscernibilidad 2018", "Relaciones de Indiscernibilidad 2021", "Equipo de trabajo"])
#option = st.sidebar.selectbox("Seleccione una pestaña", ["Introducción", "Filtrar datos", "Equipo de trabajo"])


if option == "Introducción":
    #
    st.subheader("Sobre el envejecimiento en México")
    st.write("Al igual que para otros países, la tendencia actual para la distribución por grupos de edad en México indica que en el futuro cercano la población de personas adultas mayores será considerablemente superior que la de personas jóvenes. De acuerdo con un estudio publicado por el Centro Latinoamericano y Caribeño de Demografía, esto podría ocurrir en el año 2027 y, si la tendencia continúa, para el año 2085 la población de personas adultas mayores podría llegar a 48 millones (CEPAL, 2013). Debido a lo anterior, las estrategías de prevención de enfermedades de alta incidencia en personas adultas mayores se volverán cada vez más relevantes.")
    
    st.markdown("**La Encuesta Nacional Sobre Envejecimiento en México**") 
    st.write ("Es uno de los estudios de mayor escala en la recolección de información sobre el estado de salud de las personas adultas mayores. Este estudio longitudinal, desarrollado por el Instituto Nacional de Estadística y Geografía (INEGI), en colaboración con el Centro Médico de la Universidad de Texas (UTMB), el Instituto Nacional de Geriatría (INGER) y el Instituto Nacional de Salud Pública (INSP), tiene como objetivo actualizar y dar seguimiento a la información estadística recabada en los levantamientos sobre la población de 50 años y más en México, con representación urbana y rural.")
    st.write("La ENASEM forma parte de una familia global de estudios longitudinales que tratan de entender el proceso de envejecimiento humano bajo distintas condiciones de vida. En Estados Unidos se lleva a cabo el “Health and Retirement Study” (18), en Brasil el “Estudo Longitudinal da Saúde dos Idosos Brasileiros” (3) y en la Unión Europea, “The Survey of Health, Ageing and Retirement in Europe” (17). La información recabada es fundamental para la creación de estrategias que permitan la mitigación de condiciones debilitantes para las personas adultas mayores, tales como los síndromes geriátricos.")
   
    st.markdown("**Los síndromes geriátricos**") 
    st.write("Se definen como aquellas condiciones clínicas en personas adultas mayores que no logran encajar en las categorías de otras enfermedades, sin embargo, son altamente prevalentes e implican un empobrecimiento de su calidad de vida (Inouye et al., 2007). La fragilidad, la malnutrición, la inflamación crónica y el deterioro cognitivo son ejemplos de síndromes geriátricos y se estima que las personas adultas mayores de 65 años tiene un riesgo de muerte mucho más alto que el resto de los individuos en este rango de edad (Kane, Shamliyan y Pascala, 2012).")

    st.subheader("Sarcopenia")
    st.write("La sarcopenia es uno de los síndromes geriátricos más comunes. Su definición tradicional implica una alteración progresiva y generalizada del músculo esquelético, con pérdida acelerada de masa y función muscular. La incidencia prolongada de sarcopenia en personas adultas mayores puede correlacionarse con la aparición de deterioro funcional, caídas, fragilidad y un aumento en la mortalidad (Montero-Errasquín & Cruz-Jentoft, 2022). Además, la sarcopenia incrementa la predisposición a comorbilidades, añadiendo una capa adicional de complejidad a la gestión de la salud en el contexto geriátrico (Cruz-Jentoft, 2019). ")

    st.subheader("Comorbilidades asociadas a la sarcopenia")

import tkinter as tk
from tkinter import ttk

ventana = tk.Tk()
    ventana.title("Ventana con pestañas")
    
    pestañas = ttk.Notebook(ventana)
    pestañas.pack()
    
    pestaña1 = ttk.Frame(pestañas)
    pestaña2 = ttk.Frame(pestañas)
    
    pestañas.add(pestaña1, text="Pestaña 1")
    pestañas.add(pestaña2, text="Pestaña 2")
    
    etiqueta1 = tk.Label(pestaña1, text="Etiqueta en la pestaña 1")
    etiqueta1.pack()
    
    boton2 = tk.Button(pestaña2, text="Botón en la pestaña 2")
    boton2.pack()



elif option == "Filtrar datos":
    # Menú desplegable para elegir el archivo
    selected_file = st.selectbox("Selecciona un archivo CSV", list(file_urls.keys()))

    if selected_file:
        # Cargar el archivo seleccionado
        data = load_csv_from_drive(file_urls[selected_file])
        
        st.write(f"Archivo seleccionado: {selected_file}")
        st.write(data)
        
        # Lista de verificación para seleccionar columnas
        selected_columns = st.multiselect("Selecciona las columnas para mostrar", data.columns.tolist())
        
        if selected_columns:
            # Crear dataframe reducido
            reduced_data = data[selected_columns]
            
            st.write("Dataframe reducido:")
            st.write(reduced_data)
            
            # Mostrar información del dataframe reducido
            num_rows, num_cols = reduced_data.shape
            st.write(f"Número de filas: {num_rows}")
            st.write(f"Número de columnas: {num_cols}")
            
            # Contar valores NaN por columna
            nan_counts = reduced_data.isna().sum().reset_index()
            nan_counts.columns = ["Clave", "Conteo"]
            
            st.write("Conteo de valores NaN por columna:")
            st.write(nan_counts)

            # Botón para descargar el dataframe reducido en formato csv
            csv_data = convert_df_to_csv(reduced_data)
            st.download_button(
                label="Descargar Dataframe en formato CSV",
                data=csv_data,
                file_name="dataframe_reducido.csv",
                mime="text/csv"
            )

            xlsx_data = convert_df_to_xlsx(reduced_data)
            st.download_button(
                label="Descargar Dataframe en formato XLSX",
                data=xlsx_data,
                file_name="dataframe_reducido.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


    st.subheader("Unir dataframes")

    
    # Seleccionar dos archivos CSV para unir
    selected_files = st.multiselect("Selecciona dos archivos CSV para unir", list(file_urls.keys()), default=None, max_selections=2)

    if len(selected_files) == 2:
        # Cargar los dos archivos seleccionados
        df1 = load_csv_from_drive(file_urls[selected_files[0]])
        df2 = load_csv_from_drive(file_urls[selected_files[1]])
        
        if df1 is not None and df2 is not None:
            # Unir los dataframes usando la columna 'CUNICAH'
            merged_data = pd.merge(df1, df2, on='CUNICAH', how='inner')
            
            st.write("Dataframe unido:")
            st.write(merged_data)
            
            # Mostrar información del dataframe reducido
            num_rows, num_cols = merged_data.shape
            st.write(f"Número de filas: {num_rows}")
            st.write(f"Número de columnas: {num_cols}")
            
            # Contar valores NaN por columna
            nan_counts = merged_data.isna().sum().reset_index()
            nan_counts.columns = ["Clave", "Conteo"]
            
            st.write("Conteo de valores NaN por columna:")
            st.write(nan_counts)

            # Botón para descargar el dataframe reducido en formato csv
            csv_data = convert_df_to_csv(merged_data)
            st.download_button(
                label="Descargar Dataframe en formato CSV",
                data=csv_data,
                file_name="dataframe_unificado.csv",
                mime="text/csv"
            )
            
            # Botón para descargar el dataframe unido en formato CSV
            csv_data = convert_df_to_csv(merged_data)
            st.download_button(
                label="Descargar Dataframe unido en formato CSV",
                data=csv_data,
                file_name="dataframe_unido.csv",
                mime="text/csv"
            )

        st.subheader("Selección de columnas")

        # Lista de verificación para seleccionar columnas
        selected_columns = st.multiselect("Selecciona las columnas para mostrar", merged_data.columns.tolist())
        
        if selected_columns:
            # Crear dataframe reducido
            reduced_merged_data = merged_data[selected_columns]
            
            st.write("Dataframe reducido:")
            st.write(reduced_merged_data)

            # Mostrar información del dataframe reducido
            num_rows, num_cols = reduced_merged_data.shape
            st.write(f"Número de filas: {num_rows}")
            st.write(f"Número de columnas: {num_cols}")
            
            # Contar valores NaN por columna
            nan_counts = reduced_merged_data.isna().sum().reset_index()
            nan_counts.columns = ["Clave", "Conteo"]
            
            st.write("Conteo de valores NaN por columna:")
            st.write(nan_counts)

            # Botón para descargar el dataframe reducido en formato csv
            csv_data = convert_df_to_csv(reduced_merged_data)
            st.download_button(
                label="Descargar Dataframe en formato CSV",
                data=csv_data,
                file_name="dataframe_unificado_reducido.csv",
                mime="text/csv"
            )

            xlsx_data = convert_df_to_xlsx(reduced_merged_data)
            st.download_button(
                label="Descargar Dataframe en formato XLSX",
                data=xlsx_data,
                file_name="dataframe_unificado_reducido.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
elif option == "Buscador de variables":

#    import pandas as pd
#    import gdown
#    import streamlit as st

#    def cargar_diccionario(url, nombre):
#        output = f'{nombre}.csv'
#        gdown.download(url, output, quiet=False)

#        try:
#            # Intentar leer el archivo CSV
#            df = pd.read_csv(output, header=None, encoding='utf-8', on_bad_lines='skip')
#            st.write("Archivo CSV cargado correctamente.")
#        except pd.errors.ParserError as e:
#            st.error(f"Error al leer el archivo CSV: {e}")
#            return {}

#        diccionario = {}
#        for index, row in df.iterrows():
#            variable = row[0]
#            if variable.startswith("Pregunta"):
#                partes = variable.split(" ", 2)
#                if len(partes) < 3:
#                    continue
#                codigo = partes[1].replace('.', '_')
#                explicacion = partes[2]
#                diccionario[codigo] = explicacion
#        return diccionario

#    # URLs de los archivos en Google Drive
#    urls = {
#        '2018': 'https://drive.google.com/uc?id=1ChWgiZ7JY0H-pAOqrUhsgoUyXfACz-qR',
#        '2021': 'https://drive.google.com/uc?id=1DTEFIkQVc2D-KwBBHlK4ed2qEn2EiZUg'
#    }

#    # Nombres de los diccionarios
#    nombres_diccionarios = list(urls.keys())

#    # Interfaz de selección múltiple en Streamlit
#    st.title("Buscador de Variables por Año")

#    # Barra de selección múltiple para elegir el año
#    años_seleccionados = st.multiselect('Selecciona el año del diccionario', nombres_diccionarios)

#    # Si se seleccionan años, cargar los diccionarios correspondientes
#    diccionarios = {}
#    for año in años_seleccionados:
#        url = urls[año]
#        diccionarios[año] = cargar_diccionario(url, f'diccionario_{año}')

#    # Interfaz de búsqueda por código en los diccionarios seleccionados
#    if años_seleccionados:
#        codigo_busqueda = st.text_input("Ingrese el código de la variable (por ejemplo, AA21_21):")
#        if codigo_busqueda:
#            for año, diccionario in diccionarios.items():
#                explicacion = diccionario.get(codigo_busqueda, None)
#                if explicacion:
#                    st.write(f"Explicación para el código {codigo_busqueda} en {año}: {explicacion}")
#                else:
#                    st.write(f"No se encontró explicación para el código {codigo_busqueda} en {año}.")
#        else:
#            st.write("Por favor, ingrese un código de variable.")
#    else:
#        st.write("Por favor, selecciona al menos un año.")

    import pandas as pd
    import gdown
    import streamlit as st

    def cargar_diccionario(url, nombre):
        output = f'{nombre}.xlsx'
        gdown.download(url, output, quiet=False)

        try:
            # Intentar leer el archivo Excel
            xls = pd.ExcelFile(output)
            sheet_name = xls.sheet_names[0]  # Obtener el nombre de la primera hoja
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None, engine='openpyxl')
            st.write("Archivo xlsx cargado correctamente.")
        except Exception as e:
            st.error(f"Error al leer el archivo xlsx: {e}")
            return {}

        diccionario = {}
        for index, row in df.iterrows():
            variable = row[0]
            if variable.startswith("Pregunta"):
                partes = variable.split(" ", 2)
                if len(partes) < 3:
                    continue
                codigo = partes[1].replace('.', '_')
                explicacion = partes[2]
                diccionario[codigo] = explicacion
        return diccionario

    # URLs de los archivos en Google Drive (usando enlaces directos de descarga)
    urls = {
        '2018': 'https://drive.google.com/uc?id=17o5hDLk_RHU6kGKinJtmRGtApv927Abu',
        '2021': 'https://drive.google.com/uc?id=1K0wPIeN5gE5NizkmzBdvCw0WNGpKZFbs'
    }

    # Nombres de los diccionarios
    nombres_diccionarios = list(urls.keys())

    # Interfaz de selección múltiple en Streamlit
    st.title("Buscador de Variables por Año")

    # Inicializar el estado de la sesión para el historial de búsquedas
    if 'historico_busquedas' not in st.session_state:
        st.session_state.historico_busquedas = pd.DataFrame(columns=['Año', 'Código', 'Explicación'])

    # Barra de selección múltiple para elegir el año
    años_seleccionados = st.multiselect('Selecciona el año del diccionario', nombres_diccionarios)

    # Si se seleccionan años, cargar los diccionarios correspondientes
    diccionarios = {}
    for año in años_seleccionados:
        url = urls[año]
        diccionarios[año] = cargar_diccionario(url, f'diccionario_{año}')

    # Interfaz de búsqueda por código en los diccionarios seleccionados
    if años_seleccionados:
        codigo_busqueda = st.text_input("Ingrese el código de la variable (por ejemplo, AA21_21):")
        if codigo_busqueda:
            for año, diccionario in diccionarios.items():
                explicacion = diccionario.get(codigo_busqueda, None)
                if explicacion:
                    st.write(f"Explicación para el código {codigo_busqueda} en {año}: {explicacion}")
                    # Agregar la búsqueda al histórico
                    nueva_fila = pd.DataFrame([[año, codigo_busqueda, explicacion]], columns=['Año', 'Código', 'Explicación'])
                    st.session_state.historico_busquedas = pd.concat([st.session_state.historico_busquedas, nueva_fila], ignore_index=True)
                else:
                    st.write(f"No se encontró explicación para el código {codigo_busqueda} en {año}.")
            # Mostrar el histórico de búsquedas
            st.dataframe(st.session_state.historico_busquedas)
        else:
            st.write("Por favor, ingrese un código de variable.")
    else:
        st.write("Por favor, selecciona al menos un año.")


elif option == "Buscador de datos":

    st.title('Filtrar DataFrame por Columnas')

    # Crear una caja de carga de archivos
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

    if uploaded_file is not None:
        # Cargar el archivo como un DataFrame de pandas
        df = pd.read_csv(uploaded_file)

    # Mostrar el DataFrame cargado
    st.write('DataFrame cargado:')
    st.dataframe(df)

    # Opcional: mostrar estadísticas básicas del DataFrame
    st.write('Descripción del DataFrame:')
    st.write(df.describe())

    # Lista de verificación para seleccionar columnas
    selected_columns = st.multiselect("Selecciona las columnas para mostrar", df.columns.tolist())
        
    if selected_columns:
        # Crear dataframe reducido
        df = df[selected_columns]
            
        st.write("Dataframe reducido:")
        st.write(df)
            
        # Mostrar información del dataframe reducido
        num_rows, num_cols = df.shape
        st.write(f"Número de filas: {num_rows}")
        st.write(f"Número de columnas: {num_cols}")
            
        # Contar valores NaN por columna
        nan_counts = df.isna().sum().reset_index()
        nan_counts.columns = ["Clave", "Conteo"]
            
        st.write("Conteo de valores NaN por columna:")
        st.write(nan_counts)

    
        columnas_seleccionadas = list(df.columns)
        
        
        # Crear widgets de selección para cada columna seleccionada
        filtros = {}
        for col in columnas_seleccionadas:
            if df[col].dtype == 'object':
                valores_unicos = df[col].unique().tolist()
                seleccion = st.multiselect(f'Seleccionar valores para {col}', valores_unicos)
                if seleccion:
                    filtros[col] = seleccion
            else:
                rango = st.slider(f'Seleccionar rango para {col}', min_value=float(df[col].min()), max_value=float(df[col].max()), value=(float(df[col].min()), float(df[col].max())), step=1.0)
                if rango:
                    filtros[col] = rango

        # Filtrar el DataFrame basado en los valores seleccionados
        df_filtrado = df.copy()
        for col, condicion in filtros.items():
            if isinstance(condicion, list):
                df_filtrado = df_filtrado[df_filtrado[col].isin(condicion)]
            else:
                df_filtrado = df_filtrado[(df_filtrado[col] >= condicion[0]) & (df_filtrado[col] <= condicion[1])]

        st.write('DataFrame Filtrado')
        st.dataframe(df_filtrado)

elif option == "Relaciones de Indiscernibilidad 2018":
    
    def indiscernibility(attr, table):
        u_ind = {}  # un diccionario vacío para almacenar los elementos de la relación de indiscernibilidad (U/IND({conjunto de atributos}))
        attr_values = []  # una lista vacía para almacenar los valores de los atributos
        for i in table.index:
            attr_values = []
            for j in attr:
                attr_values.append(table.loc[i, j])  # encontrar el valor de la tabla en la fila correspondiente y el atributo deseado y agregarlo a la lista attr_values
            # convertir la lista en una cadena y verificar si ya es una clave en el diccionario
            key = ''.join(str(k) for k in attr_values)
            if key in u_ind:  # si la clave ya existe en el diccionario
                u_ind[key].add(i)
            else:  # si la clave aún no existe en el diccionario
                u_ind[key] = set()
                u_ind[key].add(i)
        # Ordenar la relación de indiscernibilidad por la longitud de cada conjunto
        u_ind_sorted = sorted(u_ind.values(), key=len, reverse=True)
        return u_ind_sorted
    
    def lower_approximation(R, X):  # Describir el conocimiento en X respecto al conocimiento en R; ambos son LISTAS DE CONJUNTOS [{},{}]
        l_approx = set()  # cambiar a [] si quieres que el resultado sea una lista de conjuntos
        for i in range(len(X)):
            for j in range(len(R)):
                if R[j].issubset(X[i]):
                    l_approx.update(R[j])  # cambiar a .append() si quieres que el resultado sea una lista de conjuntos
        return l_approx

    def upper_approximation(R, X):  # Describir el conocimiento en X respecto al conocimiento en R; ambos son LISTAS DE CONJUNTOS [{},{}]
        u_approx = set()  # cambiar a [] si quieres que el resultado sea una lista de conjuntos
        for i in range(len(X)):
            for j in range(len(R)):
                if R[j].intersection(X[i]):
                    u_approx.update(R[j])  # cambiar a .append() si quieres que el resultado sea una lista de conjuntos
        return u_approx



    st.title('Filtrar DataFrame por Columnas')

    # Crear una caja de carga de archivos
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

    if uploaded_file is not None:
        # Cargar el archivo como un DataFrame de pandas
        df = pd.read_csv(uploaded_file)

    #df.columns = df.columns.str.replace(r'(_18|_19)$', '', regex=True)

    # Mostrar el DataFrame cargado
    #df.columns = df.columns.str.replace('_18', '', regex=False)
    #df.columns = df.columns.str.replace('_18', '', regex=False).str.replace('_21', '', regex=False)
    st.write('DataFrame cargado:')
    st.dataframe(df)

    # Opcional: mostrar estadísticas básicas del DataFrame
    st.write('Descripción del DataFrame:')
    st.write(df.describe())
    st.write(df.shape)


    # Lista de verificación para seleccionar columnas
    selected_columns = st.multiselect("Selecciona las columnas para mostrar", df.columns.tolist())
        
    if selected_columns:
        # Crear dataframe reducido
        df = df[selected_columns]
            
        st.write("Dataframe reducido:")
        st.write(df)
            
        # Mostrar información del dataframe reducido
        num_rows, num_cols = df.shape
        st.write(f"Número de filas: {num_rows}")
        st.write(f"Número de columnas: {num_cols}")
            
        # Contar valores NaN por columna
        nan_counts = df.isna().sum().reset_index()
        nan_counts.columns = ["Clave", "Conteo"]
            
        st.write("Conteo de valores NaN por columna:")
        st.write(nan_counts)

######################################
        import pandas as pd
        import gdown
        import streamlit as st

        def cargar_diccionario(url, nombre):
            output = f'{nombre}.xlsx'
            gdown.download(url, output, quiet=False)

            try:
                # Intentar leer el archivo Excel
                xls = pd.ExcelFile(output)
                sheet_name = xls.sheet_names[0]  # Obtener el nombre de la primera hoja
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None, engine='openpyxl')
                st.write("Archivo xlsx cargado correctamente.")
            except Exception as e:
                st.error(f"Error al leer el archivo xlsx: {e}")
                return {}

            diccionario = {}
            for index, row in df.iterrows():
                variable = row[0]
                if variable.startswith("Pregunta"):
                    partes = variable.split(" ", 2)
                    if len(partes) < 3:
                        continue
                    codigo = partes[1].replace('.', '_')
                    explicacion = partes[2]
                    diccionario[codigo] = explicacion
            return diccionario

# URLs de los archivos en Google Drive (usando enlaces directos de descarga)
        urls = {
            '2018': 'https://drive.google.com/uc?id=17o5hDLk_RHU6kGKinJtmRGtApv927Abu',
            '2021': 'https://drive.google.com/uc?id=1K0wPIeN5gE5NizkmzBdvCw0WNGpKZFbs'
        }

# Nombres de los diccionarios
        nombres_diccionarios = list(urls.keys())

# Interfaz de selección múltiple en Streamlit
        st.title("Buscador de Variables por Año")

# Inicializar el estado de la sesión para el historial de búsquedas
        if 'historico_busquedas' not in st.session_state:
            st.session_state.historico_busquedas = pd.DataFrame(columns=['Año', 'Código', 'Explicación'])

        # Cargar el dataframe df_c (esto puede ser reemplazado con la carga real de df_c)
        df_c = df.copy()
        # Obtener la lista de nombres de columnas
        columnas = df_c.columns.tolist()

        # Barra de selección múltiple para elegir el año
        años_seleccionados = st.multiselect('Selecciona el año del diccionario', nombres_diccionarios)

        # Si se seleccionan años, cargar los diccionarios correspondientes
        diccionarios = {}
        for año in años_seleccionados:
            url = urls[año]
            diccionarios[año] = cargar_diccionario(url, f'diccionario_{año}')

        # Interfaz de búsqueda por código en los diccionarios seleccionados usando una barra desplegable
        if años_seleccionados:
            codigos_busqueda = st.multiselect("Seleccione el código de la variable:", columnas)
            if codigos_busqueda:
                for codigo_busqueda in codigos_busqueda:
                    for año, diccionario in diccionarios.items():
                        explicacion = diccionario.get(codigo_busqueda, None)
                        if explicacion:
                            st.write(f"Explicación para el código {codigo_busqueda} en {año}: {explicacion}")
                            # Agregar la búsqueda al histórico
                            nueva_fila = pd.DataFrame([[año, codigo_busqueda, explicacion]], columns=['Año', 'Código', 'Explicación'])
                            st.session_state.historico_busquedas = pd.concat([st.session_state.historico_busquedas, nueva_fila], ignore_index=True)
                        else:
                            st.write(f"No se encontró explicación para el código {codigo_busqueda} en {año}.")
                # Mostrar el histórico de búsquedas
                st.dataframe(st.session_state.historico_busquedas)
            else:
                st.write("Por favor, seleccione al menos un código de variable.")
        else:
            st.write("Por favor, selecciona al menos un año.")



######################################
        df = df[df['AGE_18'] < 100]
        # Lista de columnas a modificar
        columnas_modificar = ['H5_18', 'H6_18', 'H11_18', 'H15A_18', 'C37_18']

        # Convertir valores 6.0 o 7.0 en 1.0 en las columnas especificadas
        df[columnas_modificar] = df[columnas_modificar].replace({6.0: 1.0, 7.0: 1.0})

        # Combinar los campos de las columnas de estatura en una sola columna de estatura en metros
        df['C67_18'] = df['C67_1_18'] + df['C67_2_18'] / 100
        df = df.drop(columns=['C67_1_18', 'C67_2_18'])
        df['Indice'] = df.index



        # Eliminar filas que contengan valores 8.0 o 9.0 en cualquiera de las columnas especificadas
        df = df[~df[columnas_modificar].isin([8.0, 9.0]).any(axis=1)]

        columnas_seleccionadas = list(df.columns)
        
        # Crear widgets de selección para cada columna seleccionada
        filtros = {}
        for col in columnas_seleccionadas:
            if df[col].dtype == 'object':
                valores_unicos = df[col].unique().tolist()
                seleccion = st.multiselect(f'Seleccionar valores para {col}', valores_unicos)
                if seleccion:
                    filtros[col] = seleccion
            else:
                rango = st.slider(f'Seleccionar rango para {col}', min_value=float(df[col].min()), max_value=float(df[col].max()), value=(float(df[col].min()), float(df[col].max())), step=1.0)
                if rango:
                    filtros[col] = rango

        # Filtrar el DataFrame basado en los valores seleccionados
        df_filtrado = df.copy()
        for col, condicion in filtros.items():
            if isinstance(condicion, list):
                df_filtrado = df_filtrado[df_filtrado[col].isin(condicion)]
            else:
                df_filtrado = df_filtrado[(df_filtrado[col] >= condicion[0]) & (df_filtrado[col] <= condicion[1])]

        st.write('DataFrame Filtrado')
        st.dataframe(df_filtrado)
        st.write("Las dimensiones de la base de datos son:")
        df_filtrado.shape
        datos_filtrados=df_filtrado.copy()

     # Definir condiciones para cada grupo
        conditions = {
            "sanos": {
                'C4_18': 2.0,
                'C6_18': 2.0,
                'C12_18': 2.0,
                'C19_18': 2.0,
                'C22A_18': 2.0,
                'C26_18': 2.0,
                'C32_18': 2.0
            },
            "diabetes": {
                'C4_18': 1.0,
                'C6_18': 2.0,
                'C12_18': 2.0,
                'C19_18': 2.0,
                'C22A_18': 2.0,
                'C26_18': 2.0,
                'C32_18': 2.0
            },
            "Hipertension": {
                'C4_18': 2.0,
                'C6_18': 1.0,
                'C12_18': 2.0,
                'C19_18': 2.0,
                'C22A_18': 2.0,
                'C26_18': 2.0,
                'C32_18': 2.0
            },
            "diabetes e hipertension": {
                'C4_18': 1.0,
                'C6_18': 1.0,
                'C12_18': 2.0,
                'C19_18': 2.0,
                'C22A_18': 2.0,
                'C26_18': 2.0,
                'C32_18': 2.0
            }
        }



        # Crear una selección en Streamlit para elegir entre los conjuntos
        seleccion = st.selectbox("Selecciona un grupo", list(conditions.keys()))

        # Crear una selección múltiple en Streamlit para el valor de SEX_18
        sex_values = df_filtrado['SEX_18'].unique()
        sex_selection = st.multiselect("Selecciona el valor de SEX_18", sex_values, default=sex_values)

        # Filtrar el DataFrame en función de las condiciones seleccionadas y el valor de SEX_18
        condiciones_seleccionadas = conditions[seleccion]
        nuevo_dataframe_filtrado = df_filtrado.copy()

        # Aplicar las condiciones seleccionadas
        for columna, valor in condiciones_seleccionadas.items():
            nuevo_dataframe_filtrado = nuevo_dataframe_filtrado[nuevo_dataframe_filtrado[columna] == valor]

        # Aplicar el filtro del valor de SEX_18
        #nuevo_dataframe_filtrado = nuevo_dataframe_filtrado[nuevo_dataframe_filtrado['SEX_18'] == sex_selection]
        # Aplicar el filtro del valor de SEX_18
        nuevo_dataframe_filtrado = nuevo_dataframe_filtrado[nuevo_dataframe_filtrado['SEX_18'].isin(sex_selection)]



        # Mostrar el DataFrame filtrado
        st.dataframe(nuevo_dataframe_filtrado)
        datos_limpios = nuevo_dataframe_filtrado.dropna()
        st.write("Las dimensiones de la base de datos son:")
        st.write(datos_limpios.shape)


##########################################

    ind=indiscernibility(['C37_18', 'H11_18', 'H15A_18', 'H5_18', 'H6_18'], datos_limpios)
    

    import matplotlib.pyplot as plt


    # Calcular las longitudes de los conjuntos con longitud >= 2
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(ind) if len(conjunto) >= 2]

    # Ordenar la lista de tuplas por la longitud de los conjuntos en orden descendente
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Crear un diccionario para mapear el índice del conjunto al nuevo nombre
    nombres_conjuntos_nuevos = {num_conjunto: f"Conjunto {i}" for i, (num_conjunto, _) in enumerate(longitudes_conjuntos_ordenadas)}

    st.write("Longitudes de los conjuntos con su respectivo nuevo nombre (en orden descendente):")
    for num_conjunto, longitud in longitudes_conjuntos_ordenadas:
        nombre_conjunto_nuevo = nombres_conjuntos_nuevos[num_conjunto]
        st.write(f"{nombre_conjunto_nuevo}: {longitud}")

    # Calcular el total de elementos en todos los conjuntos
    total_elementos = sum(longitud for _, longitud in longitudes_conjuntos_ordenadas)

    # Calcular los porcentajes para cada conjunto
    porcentajes = [(nombres_conjuntos_nuevos[num_conjunto], longitud / total_elementos * 100) for num_conjunto, longitud in longitudes_conjuntos_ordenadas]

    # Extraer los nombres de los conjuntos y los porcentajes para el diagrama de pastel
    nombres_conjuntos = [nombre for nombre, _ in porcentajes]
    porcentajes_valores = [valor for _, valor in porcentajes]

    # Crear el diagrama de pastel
    fig, ax = plt.subplots(figsize=(10, 8))
    _, _, autopcts = ax.pie(porcentajes_valores, labels=nombres_conjuntos, autopct='%1.1f%%', startangle=140, textprops={'visible': False})
    for autopct in autopcts:
        autopct.set_visible(True)  # Mostrar los porcentajes solo para los grupos con más de 30 miembros
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Distribución de subconjuntos')

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)



    # Entrada del usuario para el tamaño mínimo del conjunto
    tamaño_mínimo = st.number_input("Introduce el tamaño mínimo del conjunto:", min_value=1, value=2, step=1)

    # Calcular las longitudes de los conjuntos con longitud >= tamaño_mínimo
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(ind) if len(conjunto) >= tamaño_mínimo]

    # Ordenar la lista de tuplas por la longitud de los conjuntos en orden descendente
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Crear un diccionario para mapear el índice del conjunto al nuevo nombre
    nombres_conjuntos_nuevos = {num_conjunto: f"Conjunto {i}" for i, (num_conjunto, _) in enumerate(longitudes_conjuntos_ordenadas)}

    st.write("Longitudes de los conjuntos con su respectivo nuevo nombre (en orden descendente):")
    for num_conjunto, longitud in longitudes_conjuntos_ordenadas:
        nombre_conjunto_nuevo = nombres_conjuntos_nuevos[num_conjunto]
        st.write(f"{nombre_conjunto_nuevo}: {longitud}")

    # Calcular el total de elementos en todos los conjuntos
    total_elementos = sum(longitud for _, longitud in longitudes_conjuntos_ordenadas)

    # Calcular los porcentajes para cada conjunto
    porcentajes = [(nombres_conjuntos_nuevos[num_conjunto], longitud / total_elementos * 100) for num_conjunto, longitud in longitudes_conjuntos_ordenadas]

    # Extraer los nombres de los conjuntos y los porcentajes para el diagrama de pastel
    nombres_conjuntos = [nombre for nombre, _ in porcentajes]
    porcentajes_valores = [valor for _, valor in porcentajes]

    # Crear el diagrama de pastel
    fig, ax = plt.subplots(figsize=(10, 8))
    _, _, autopcts = ax.pie(porcentajes_valores, labels=nombres_conjuntos, autopct='%1.1f%%', startangle=140, textprops={'visible': False})
    for autopct in autopcts:
        autopct.set_visible(True)  # Mostrar los porcentajes
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Distribución de subconjuntos')

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    import numpy as np
    
    # Calcular las longitudes de los conjuntos con longitud >= tamaño_mínimo
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(ind) if len(conjunto) >= tamaño_mínimo]

    # Ordenar la lista de tuplas por la longitud de los conjuntos en orden descendente
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Obtener los 15 conjuntos más numerosos
    conjuntos_mas_numerosos = [conjunto for i, conjunto in enumerate(ind) if i in [num_conjunto for num_conjunto, _ in longitudes_conjuntos_ordenadas[:15]]]

    # Crear DataFrames para cada uno de los 15 conjuntos más numerosos
    for i, conjunto in enumerate(conjuntos_mas_numerosos, 0):
        indices_seleccionados = list(conjunto)
        df_conjunto = datos_limpios[datos_limpios.index.isin(indices_seleccionados)]
        globals()[f"df_Conjunto_{i}"] = df_conjunto

    # Definir las columnas de interés
    columnas_interes_radar = ['C37_18', 'H11_18', 'H15A_18', 'H5_18', 'H6_18']

    # Definir los nombres de los dataframes
    nombres_dataframes = [f"df_Conjunto_{i}" for i in range(0, 15)]

    # Definir los valores para cada dataframe en las columnas de interés
    valores_dataframes = []
    for nombre_df in nombres_dataframes:
        df = eval(nombre_df)
        valores = df[columnas_interes_radar].iloc[0].tolist()  # Tomar solo la primera fila
        valores_dataframes.append(valores)

    # Colores para los gráficos
    colores = plt.cm.tab20(np.linspace(0, 1, len(nombres_dataframes)))

    # Total de pacientes
    total_pacientes = len(datos_limpios)

    # Combinar nombres de DataFrames con el número de filas
    dataframes_con_filas = zip(nombres_dataframes, valores_dataframes, colores, [len(eval(nombre)) for nombre in nombres_dataframes])

    # Ordenar la lista combinada por el número de filas
    dataframes_con_filas_ordenados = sorted(dataframes_con_filas, key=lambda x: x[3], reverse=True)

    # Crear un gráfico de radar individual para cada dataframe
    num_filas = 3
    num_columnas = 5
    fig, axs = plt.subplots(num_filas, num_columnas, figsize=(20, 10), subplot_kw=dict(polar=True))
    fig.subplots_adjust(hspace=0.7, wspace=0.4)  # Ajustar el espacio horizontal

    for i, (nombre, valores, color, num_filas_df) in enumerate(dataframes_con_filas_ordenados):
        fila = i // num_columnas
        columna = i % num_columnas
        ax = axs[fila, columna]

        valores += valores[:1]  # Para cerrar el polígono
        angulos = np.linspace(0, 2 * np.pi, len(columnas_interes_radar), endpoint=False).tolist()
        angulos += angulos[:1]  # Para cerrar el polígono
        ax.plot(angulos, valores, color=color)
        ax.fill(angulos, valores, color=color, alpha=0.25)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(columnas_interes_radar)
        ax.yaxis.grid(True)
        ax.set_title(nombre)

        # Agregar el número de filas y el porcentaje debajo del gráfico
        porcentaje = (num_filas_df / total_pacientes) * 100
        ax.text(0.5, -0.2, f"Número de filas: {num_filas_df} ({porcentaje:.2f}%)", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    import pandas as pd

    # Crear un diccionario para mapear el índice de cada fila al número de conjunto
    indice_a_conjunto = {}
    for i, conjunto in enumerate(ind):
        for indice in conjunto:
            indice_a_conjunto[indice] = i

    # Agregar una nueva columna "num_conjunto" al DataFrame 'datos_limpios' usando el diccionario
    datos_limpios['num_conjunto'] = datos_limpios.index.map(indice_a_conjunto)

    # Mostrar el DataFrame con la nueva columna
#    datos_limpios

    # Seleccionar las filas que tienen valores del 0 al 14 en la columna 'num_conjunto'
    filas_seleccionadas = datos_limpios[datos_limpios['num_conjunto'].isin(range(15))]

    # Seleccionar solo las columnas requeridas
    filas_seleccionadas = filas_seleccionadas[['C37_18', 'H11_18', 'H15A_18', 'H5_18', 'H6_18', 'num_conjunto']]

    # Crear un nuevo DataFrame con las filas seleccionadas
    nuevo_dataframe = pd.DataFrame(filas_seleccionadas)

    # Mostrar las primeras filas del nuevo DataFrame
    nuevo_dataframe



#######################

#####################333

    import pandas as pd

    # Definir las condiciones para asignar los valores a la nueva columna
    def asignar_riesgo(num_conjunto):
        if num_conjunto in [4, 13]:
            return "Riesgo considerable"
        elif num_conjunto in [3, 6, 9]:
            return "Riesgo moderado"
        elif num_conjunto in [1, 2, 5, 7, 8, 10, 11, 12, 14]:
            return "Riesgo leve"
        elif num_conjunto == 0:
            return "Sin Riesgo"
        else:
            return "No clasificado"  # Manejar cualquier otro caso

    # Agregar la nueva columna al DataFrame
    nuevo_dataframe['nivel_riesgo'] = nuevo_dataframe['num_conjunto'].apply(asignar_riesgo)

    # Mostrar las primeras filas del DataFrame con la nueva columna
    nuevo_dataframe



#######################


    # Función para calcular la diferencia entre tamaños de conjuntos
    def calcular_diferencia(lista1, lista2):
        diferencia = sum(abs(len(conj1) - len(conj2)) for conj1, conj2 in zip(lista1, lista2))
        return diferencia

    # Definir las columnas de interés
    columnas_interes = ['H15A_18', 'H11_18', 'H5_18', 'H6_18', 'C37_18']

    # Generar listas de conjuntos
    lista_1 = indiscernibility(columnas_interes, nuevo_dataframe)
    lista_2_original = indiscernibility(columnas_interes, nuevo_dataframe)

    # Obtener lista de tamaños de cada conjunto
    tamaños_lista_1 = [len(conjunto) for conjunto in lista_1]
    tamaños_lista_2_original = [len(conjunto) for conjunto in lista_2_original]

    # Inicializar variables para seguimiento de la lista más parecida
    mejor_lista = lista_2_original
    mejor_diferencia = calcular_diferencia(lista_1, lista_2_original)

    # Eliminar una por una cada columna de lista_2 y mostrar los tamaños resultantes
    for columna1 in columnas_interes:
        columnas_sin_columna1 = columnas_interes.copy()
        columnas_sin_columna1.remove(columna1)
        lista_2_sin_columna1 = indiscernibility(columnas_sin_columna1, nuevo_dataframe)
        diferencia = calcular_diferencia(lista_1, lista_2_sin_columna1)
    
        if diferencia < mejor_diferencia:
            mejor_lista = lista_2_sin_columna1
            mejor_diferencia = diferencia
    
        # Eliminar pares de columnas de lista_2 y mostrar los tamaños resultantes
        for columna2 in columnas_sin_columna1:
            if columna2 != columna1:
                columnas_sin_par = columnas_sin_columna1.copy()
                columnas_sin_par.remove(columna2)
                lista_2_sin_par = indiscernibility(columnas_sin_par, nuevo_dataframe)
                diferencia = calcular_diferencia(lista_1, lista_2_sin_par)
            
                if diferencia < mejor_diferencia:
                    mejor_lista = lista_2_sin_par
                    mejor_diferencia = diferencia

    # Mostrar la mejor lista encontrada en Streamlit
    st.write("Tamaños de conjuntos en lista_1:", tamaños_lista_1)
    st.write("Tamaños de conjuntos en la mejor lista:", [len(conjunto) for conjunto in mejor_lista])

    # Visualización con un gráfico de barras
    fig, ax = plt.subplots()
    labels = [f"Conjunto {i}" for i in range(len(tamaños_lista_1))]
    x = range(len(tamaños_lista_1))
    ax.bar(x, tamaños_lista_1, width=0.4, label='lista_1', align='center')
    ax.bar(x, [len(conjunto) for conjunto in mejor_lista], width=0.4, label='Mejor Lista', align='edge')
    ax.set_xlabel('Conjuntos')
    ax.set_ylabel('Tamaños')
    ax.set_title('Comparación de tamaños de conjuntos')
    ax.legend()

    st.pyplot(fig)

    # Obtener los valores únicos de la columna 'nivel de riesgo'
    nivel_riesgo = nuevo_dataframe['nivel_riesgo'].unique()


    # Crear una barra de selección múltiple para elegir niveles de riesgo
    niveles_seleccionados = st.multiselect(
        'Selecciona los niveles de riesgo a visualizar:',
        nivel_riesgo
    )

    # Filtrar el DataFrame según los niveles de riesgo seleccionados
    if niveles_seleccionados:
        df_filtrado = nuevo_dataframe[nuevo_dataframe['nivel_riesgo'].isin(niveles_seleccionados)]
        st.write(f"Filas con nivel de riesgo en {niveles_seleccionados}:")
        st.dataframe(df_filtrado)
    else:
        st.write("Selecciona al menos un nivel de riesgo para visualizar las filas correspondientes.")
########################


    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    # Generar listas de conjuntos (simulando lista_2)
    lista_2 = indiscernibility(['C37_18', 'H11_18', 'H5_18', 'H6_18'], datos_limpios)

    # Obtener longitudes de conjuntos
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(lista_2) if len(conjunto) >= 2]

    # Ordenar por longitud
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Obtener los 15 conjuntos más numerosos
    conjuntos_mas_numerosos = [conjunto for i, conjunto in enumerate(lista_2) if i in [num_conjunto for num_conjunto, _ in longitudes_conjuntos_ordenadas[:15]]]

    # Crear DataFrames para cada uno de los 15 conjuntos más numerosos
    dataframes_con_filas = []
    for i, conjunto in enumerate(conjuntos_mas_numerosos, 0):
        indices_seleccionados = list(conjunto)
        df_conjunto = datos_limpios[datos_limpios.index.isin(indices_seleccionados)]
        dataframes_con_filas.append((f"df_Conjunto_{i}", df_conjunto, len(df_conjunto)))

    # Definir las columnas de interés
    columnas_interes_radar = ['C37_18', 'H11_18', 'H5_18', 'H6_18']

    # Definir los valores para cada dataframe en las columnas de interés
    valores_dataframes = []
    for nombre_df, df, num_filas_df in dataframes_con_filas:
        valores = df[columnas_interes_radar].iloc[0].tolist()  # Tomar solo la primera fila
        valores_dataframes.append((nombre_df, valores, num_filas_df))

    # Colores para los gráficos
    colores = plt.cm.tab20(np.linspace(0, 1, len(valores_dataframes)))

    # Total de pacientes
    total_pacientes = len(datos_limpios)

    # Ordenar la lista combinada por el número de filas
    valores_dataframes_ordenados = sorted(valores_dataframes, key=lambda x: x[2], reverse=True)

    # Configurar Streamlit
    st.title("Visualización de Conjuntos Más Numerosos")
    st.write("Comparación de los conjuntos más numerosos en los datos.")

    # Crear un gráfico de radar individual para cada dataframe
    num_filas = 3
    num_columnas = 5
    fig, axs = plt.subplots(num_filas, num_columnas, figsize=(20, 10), subplot_kw=dict(polar=True))
    fig.subplots_adjust(hspace=0.7, wspace=0.4)  # Ajustar el espacio horizontal

    for i, (nombre, valores, num_filas_df) in enumerate(valores_dataframes_ordenados):
        fila = i // num_columnas
        columna = i % num_columnas
        ax = axs[fila, columna]

        valores += valores[:1]  # Para cerrar el polígono
        angulos = np.linspace(0, 2 * np.pi, len(columnas_interes_radar), endpoint=False).tolist()
        angulos += angulos[:1]  # Para cerrar el polígono
        color = colores[i]
        ax.plot(angulos, valores, color=color)
        ax.fill(angulos, valores, color=color, alpha=0.25)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(columnas_interes_radar)
        ax.yaxis.grid(True)
        ax.set_title(nombre)

        # Agregar el número de filas y el porcentaje debajo del gráfico
        porcentaje = (num_filas_df / total_pacientes) * 100
        ax.text(0.5, -0.2, f"Número de filas: {num_filas_df} ({porcentaje:.2f}%)", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)


######################


    # Título de la aplicación
    st.title("Filtrado de DataFrame por tamaño de conjuntos")

    # Calcular el tamaño de cada conjunto y filtrar los conjuntos con menos de 30 miembros
    conjuntos_mayores_30 = [conjunto for conjunto in ind if len(conjunto) >= 30]

    # Obtener los índices de las filas que pertenecen a los conjuntos mayores o iguales a 30 miembros
    indices_filtrados = [indice for conjunto in conjuntos_mayores_30 for indice in conjunto]

    # Filtrar el DataFrame 'datos_limpios' para mantener solo las filas con índices en 'indices_filtrados'
    datos_limpios_filtrados = datos_limpios.loc[indices_filtrados]

    # Mostrar el DataFrame filtrado en Streamlit
    st.write("DataFrame filtrado con conjuntos mayores o iguales a 30 miembros:")
    st.dataframe(datos_limpios_filtrados)

    # Mostrar el número total de conjuntos filtrados
    st.write(f"Número de conjuntos mayores o iguales a 30 miembros: {len(conjuntos_mayores_30)}")

    # Mostrar el tamaño de cada conjunto filtrado
    st.write("Tamaño de cada conjunto filtrado:")
    tamaños_conjuntos = [len(conjunto) for conjunto in conjuntos_mayores_30]
    st.write(tamaños_conjuntos)



#######################

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


    # Título de la aplicación
    st.title("Clasificador de Árbol de Decisión")

    # Definir las columnas de atributos
    columnas_atributos = ['C37_18', 'H11_18', 'H5_18', 'H6_18']

    # Separar los datos en atributos (X) y etiquetas (y)
    X = nuevo_dataframe[columnas_atributos]
    y = nuevo_dataframe['nivel_riesgo']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el clasificador de árbol de decisión
    clf = DecisionTreeClassifier()

    # Entrenar el clasificador
    clf.fit(X_train, y_train)

    # Predecir las etiquetas para los datos de prueba
    y_pred = clf.predict(X_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)

    # Mostrar la precisión del modelo en Streamlit
    st.write(f'Precisión del modelo: {accuracy:.2f}')

    # Mostrar el reporte de clasificación en Streamlit
    st.subheader("Reporte de Clasificación:")
    st.text(classification_report(y_test, y_pred))

    # Mostrar la matriz de confusión en Streamlit
    st.subheader("Matriz de Confusión:")
    st.write(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=clf.classes_, index=clf.classes_))

##################

    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Título de la aplicación
    st.title("Visualización del Árbol de Decisión")

    # Definir las columnas de atributos
    columnas_atributos = ['C37_18', 'H11_18', 'H5_18', 'H6_18']

    # Separar los datos en atributos (X) y etiquetas (y)
    X = nuevo_dataframe[columnas_atributos]
    y = nuevo_dataframe['nivel_riesgo']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el clasificador de árbol de decisión
    clf = DecisionTreeClassifier(random_state=42)

    # Entrenar el clasificador
    clf.fit(X_train, y_train)

    # Visualizar el árbol de decisión en Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Para evitar advertencias de Streamlit
    plt.figure(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=columnas_atributos, class_names=clf.classes_)
    st.pyplot()
##################

    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    # Suponiendo que 'nuevo_dataframe' ya está definido y contiene los datos necesarios

#    def asignar_nivel_riesgo(df, modelo, columnas_atributos):
#        # Hacer una copia del DataFrame para evitar modificar el original
#        df = df.copy()
    
#        # Preprocesamiento de los datos de entrada si es necesario
#        X = df[columnas_atributos]
    
#        # Predecir los niveles de riesgo para los datos de entrada
#        y_pred = modelo.predict(X)
    
        # Asignar los resultados al DataFrame en una nueva columna
#        df['Diagnóstico_árbol'] = y_pred
    
#        return df

    import streamlit as st
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    # Definir las columnas de atributos
    columnas_atributos = ['C37_18', 'H11_18', 'H5_18', 'H6_18']

    # Suponiendo que 'datos_limpios' ya está definido y contiene los datos necesarios

    # Función para asignar nivel de riesgo a una fila
    def asignar_nivel_riesgo(fila, modelo, columnas_atributos):
        X = fila[columnas_atributos].values.reshape(1, -1)
        y_pred = modelo.predict(X)
        return y_pred[0]

#    # Entrenar el modelo
#    X = datos_limpios[columnas_atributos]
#    y = datos_limpios['nivel_riesgo']
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#    clf = DecisionTreeClassifier(random_state=42)
#    clf.fit(X_train, y_train)

    # Visualizar el árbol de decisión
    #fig, ax = plt.subplots(figsize=(15, 10))
    #plot_tree(clf, filled=True, feature_names=columnas_atributos, class_names=clf.classes_, ax=ax)
    #st.pyplot(fig)

    # Aplicar la función asignar_nivel_riesgo al DataFrame datos_limpios
    #datos_limpios['Diagnóstico_árbol'] = datos_limpios.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    nuevo_dataframe['Diagnóstico_árbol'] = nuevo_dataframe.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)

    # Mostrar los resultados en Streamlit
    st.write("Resultados de asignación de nivel de riesgo:")
    st.dataframe(nuevo_dataframe[['C37_18', 'H11_18', 'H6_18', 'H5_18', 'Diagnóstico_árbol']])

    # Calcular el número de coincidencias y no coincidencias
    coincidencias = (nuevo_dataframe['nivel_riesgo'] == nuevo_dataframe['Diagnóstico_árbol']).sum()
    total_filas = len(datos_limpios)
    no_coincidencias = total_filas - coincidencias

    # Mostrar los resultados en Streamlit
    st.write(f"Número de filas en las que coinciden los valores: {coincidencias}")
    st.write(f"Número de filas en las que no coinciden los valores: {no_coincidencias}")
 
    datos_limpios['Diagnóstico_árbol'] = datos_limpios.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    # Aplicar la función asignar_nivel_riesgo al dataframe datos_limpios
    datos_limpios[['C37_18','H11_18', 'H6_18','H5_18','Diagnóstico_árbol']]

    #datos_filtrados.drop('H15A_18', axis=1, inplace=True) 
    datos_filtrados = datos_filtrados.dropna()
 
    datos_filtrados['Diagnóstico_árbol'] = datos_filtrados.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    datos_filtrados = datos_filtrados[['H11_18', 'H5_18', 'H6_18','C37_18','Diagnóstico_árbol']].dropna()
    datos_filtrados[['H11_18', 'H5_18', 'H6_18','C37_18','Diagnóstico_árbol']]

    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Suponiendo que 'datos_filtrados' ya está definido y contiene los datos necesarios

    # Agrupar por 'Diagnóstico_árbol' y contar el número de ocurrencias para nuevo_dataframe
    grupo_diagnostico_nuevo = nuevo_dataframe.groupby('Diagnóstico_árbol').size()

    # Agrupar por 'Diagnóstico_árbol' y contar el número de ocurrencias para datos_filtrados
    grupo_diagnostico_filtrados = datos_filtrados.groupby('Diagnóstico_árbol').size()

    # Crear el panel con dos subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico para nuevo_dataframe
    sns.barplot(x=grupo_diagnostico_nuevo.values, y=grupo_diagnostico_nuevo.index, palette='Dark2', ax=axes[0])
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].set_xlabel('Número de filas')
    axes[0].set_ylabel('Diagnóstico')
    axes[0].set_title('Conteo de diagnósticos (nuevo_dataframe)')

    # Gráfico para datos_filtrados
    sns.barplot(x=grupo_diagnostico_filtrados.values, y=grupo_diagnostico_filtrados.index, palette='Dark2', ax=axes[1])
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].set_xlabel('Número de filas')
    axes[1].set_ylabel('Diagnóstico')
    axes[1].set_title('Conteo de diagnósticos (datos_filtrados)')

    # Mostrar el panel con subplots en Streamlit
    st.pyplot(fig)

# Crear una nueva columna "Diagnóstico_árbol" en el dataframe "nuevo_dataframe"
#nuevo_dataframe['Diagnóstico_árbol'] = nuevo_dataframe.apply(asignar_nivel_riesgo, axis=1)

    nuevo_dataframe.shape

    datos_filtrados.shape

##################

elif option == "Relaciones de Indiscernibilidad 2021":
    
    def indiscernibility(attr, table):
        u_ind = {}  # un diccionario vacío para almacenar los elementos de la relación de indiscernibilidad (U/IND({conjunto de atributos}))
        attr_values = []  # una lista vacía para almacenar los valores de los atributos
        for i in table.index:
            attr_values = []
            for j in attr:
                attr_values.append(table.loc[i, j])  # encontrar el valor de la tabla en la fila correspondiente y el atributo deseado y agregarlo a la lista attr_values
            # convertir la lista en una cadena y verificar si ya es una clave en el diccionario
            key = ''.join(str(k) for k in attr_values)
            if key in u_ind:  # si la clave ya existe en el diccionario
                u_ind[key].add(i)
            else:  # si la clave aún no existe en el diccionario
                u_ind[key] = set()
                u_ind[key].add(i)
        # Ordenar la relación de indiscernibilidad por la longitud de cada conjunto
        u_ind_sorted = sorted(u_ind.values(), key=len, reverse=True)
        return u_ind_sorted
    
    def lower_approximation(R, X):  # Describir el conocimiento en X respecto al conocimiento en R; ambos son LISTAS DE CONJUNTOS [{},{}]
        l_approx = set()  # cambiar a [] si quieres que el resultado sea una lista de conjuntos
        for i in range(len(X)):
            for j in range(len(R)):
                if R[j].issubset(X[i]):
                    l_approx.update(R[j])  # cambiar a .append() si quieres que el resultado sea una lista de conjuntos
        return l_approx

    def upper_approximation(R, X):  # Describir el conocimiento en X respecto al conocimiento en R; ambos son LISTAS DE CONJUNTOS [{},{}]
        u_approx = set()  # cambiar a [] si quieres que el resultado sea una lista de conjuntos
        for i in range(len(X)):
            for j in range(len(R)):
                if R[j].intersection(X[i]):
                    u_approx.update(R[j])  # cambiar a .append() si quieres que el resultado sea una lista de conjuntos
        return u_approx



    st.title('Filtrar DataFrame por Columnas')

    # Crear una caja de carga de archivos
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

    if uploaded_file is not None:
        # Cargar el archivo como un DataFrame de pandas
        df = pd.read_csv(uploaded_file)

    #df.columns = df.columns.str.replace(r'(_18|_19)$', '', regex=True)

    # Mostrar el DataFrame cargado
    #df.columns = df.columns.str.replace('_18', '', regex=False)
    #df.columns = df.columns.str.replace('_18', '', regex=False).str.replace('_21', '', regex=False)
    st.write('DataFrame cargado:')
    st.dataframe(df)

    # Opcional: mostrar estadísticas básicas del DataFrame
    st.write('Descripción del DataFrame:')
    st.write(df.describe())
    st.write(df.shape)


    # Lista de verificación para seleccionar columnas
    selected_columns = st.multiselect("Selecciona las columnas para mostrar", df.columns.tolist())
        
    if selected_columns:
        # Crear dataframe reducido
        df = df[selected_columns]
            
        st.write("Dataframe reducido:")
        st.write(df)
            
        # Mostrar información del dataframe reducido
        num_rows, num_cols = df.shape
        st.write(f"Número de filas: {num_rows}")
        st.write(f"Número de columnas: {num_cols}")
            
        # Contar valores NaN por columna
        nan_counts = df.isna().sum().reset_index()
        nan_counts.columns = ["Clave", "Conteo"]
            
        st.write("Conteo de valores NaN por columna:")
        st.write(nan_counts)

######################################
        import pandas as pd
        import gdown
        import streamlit as st

        def cargar_diccionario(url, nombre):
            output = f'{nombre}.xlsx'
            gdown.download(url, output, quiet=False)

            try:
                # Intentar leer el archivo Excel
                xls = pd.ExcelFile(output)
                sheet_name = xls.sheet_names[0]  # Obtener el nombre de la primera hoja
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None, engine='openpyxl')
                st.write("Archivo xlsx cargado correctamente.")
            except Exception as e:
                st.error(f"Error al leer el archivo xlsx: {e}")
                return {}

            diccionario = {}
            for index, row in df.iterrows():
                variable = row[0]
                if variable.startswith("Pregunta"):
                    partes = variable.split(" ", 2)
                    if len(partes) < 3:
                        continue
                    codigo = partes[1].replace('.', '_')
                    explicacion = partes[2]
                    diccionario[codigo] = explicacion
            return diccionario

# URLs de los archivos en Google Drive (usando enlaces directos de descarga)
        urls = {
            '2018': 'https://drive.google.com/uc?id=17o5hDLk_RHU6kGKinJtmRGtApv927Abu',
            '2021': 'https://drive.google.com/uc?id=1K0wPIeN5gE5NizkmzBdvCw0WNGpKZFbs'
        }

# Nombres de los diccionarios
        nombres_diccionarios = list(urls.keys())

# Interfaz de selección múltiple en Streamlit
        st.title("Buscador de Variables por Año")

# Inicializar el estado de la sesión para el historial de búsquedas
        if 'historico_busquedas' not in st.session_state:
            st.session_state.historico_busquedas = pd.DataFrame(columns=['Año', 'Código', 'Explicación'])

        # Cargar el dataframe df_c (esto puede ser reemplazado con la carga real de df_c)
        df_c = df.copy()
        # Obtener la lista de nombres de columnas
        columnas = df_c.columns.tolist()

        # Barra de selección múltiple para elegir el año
        años_seleccionados = st.multiselect('Selecciona el año del diccionario', nombres_diccionarios)

        # Si se seleccionan años, cargar los diccionarios correspondientes
        diccionarios = {}
        for año in años_seleccionados:
            url = urls[año]
            diccionarios[año] = cargar_diccionario(url, f'diccionario_{año}')

        # Interfaz de búsqueda por código en los diccionarios seleccionados usando una barra desplegable
        if años_seleccionados:
            codigos_busqueda = st.multiselect("Seleccione el código de la variable:", columnas)
            if codigos_busqueda:
                for codigo_busqueda in codigos_busqueda:
                    for año, diccionario in diccionarios.items():
                        explicacion = diccionario.get(codigo_busqueda, None)
                        if explicacion:
                            st.write(f"Explicación para el código {codigo_busqueda} en {año}: {explicacion}")
                            # Agregar la búsqueda al histórico
                            nueva_fila = pd.DataFrame([[año, codigo_busqueda, explicacion]], columns=['Año', 'Código', 'Explicación'])
                            st.session_state.historico_busquedas = pd.concat([st.session_state.historico_busquedas, nueva_fila], ignore_index=True)
                        else:
                            st.write(f"No se encontró explicación para el código {codigo_busqueda} en {año}.")
                # Mostrar el histórico de búsquedas
                st.dataframe(st.session_state.historico_busquedas)
            else:
                st.write("Por favor, seleccione al menos un código de variable.")
        else:
            st.write("Por favor, selecciona al menos un año.")



######################################
        df = df[df['AGE_21'] < 100]
        # Lista de columnas a modificar
        columnas_modificar = ['H5_21', 'H6_21', 'H11_21', 'H15A_21', 'C37_21']

        # Convertir valores 6.0 o 7.0 en 1.0 en las columnas especificadas
        df[columnas_modificar] = df[columnas_modificar].replace({6.0: 1.0, 7.0: 1.0})

        # Combinar los campos de las columnas de estatura en una sola columna de estatura en metros
        df['C67_21'] = df['C67_1_21'] + df['C67_2_21'] / 100
        df = df.drop(columns=['C67_1_21', 'C67_2_21'])
        df['Indice'] = df.index



        # Eliminar filas que contengan valores 8.0 o 9.0 en cualquiera de las columnas especificadas
        df = df[~df[columnas_modificar].isin([8.0, 9.0]).any(axis=1)]

        columnas_seleccionadas = list(df.columns)
        
        # Crear widgets de selección para cada columna seleccionada
        filtros = {}
        for col in columnas_seleccionadas:
            if df[col].dtype == 'object':
                valores_unicos = df[col].unique().tolist()
                seleccion = st.multiselect(f'Seleccionar valores para {col}', valores_unicos)
                if seleccion:
                    filtros[col] = seleccion
            else:
                rango = st.slider(f'Seleccionar rango para {col}', min_value=float(df[col].min()), max_value=float(df[col].max()), value=(float(df[col].min()), float(df[col].max())), step=1.0)
                if rango:
                    filtros[col] = rango

        # Filtrar el DataFrame basado en los valores seleccionados
        df_filtrado = df.copy()
        for col, condicion in filtros.items():
            if isinstance(condicion, list):
                df_filtrado = df_filtrado[df_filtrado[col].isin(condicion)]
            else:
                df_filtrado = df_filtrado[(df_filtrado[col] >= condicion[0]) & (df_filtrado[col] <= condicion[1])]

        st.write('DataFrame Filtrado')
        st.dataframe(df_filtrado)
        st.write("Las dimensiones de la base de datos son:")
        df_filtrado.shape
        datos_filtrados=df_filtrado.copy()

     # Definir condiciones para cada grupo
        conditions = {
            "sanos": {
                'C4_21': 2.0,
                'C6_21': 2.0,
                'C12_21': 2.0,
                'C19_21': 2.0,
                'C22A_21': 2.0,
                'C26_21': 2.0,
                'C32_21': 2.0
            },
            "diabetes": {
                'C4_21': 1.0,
                'C6_21': 2.0,
                'C12_21': 2.0,
                'C19_21': 2.0,
                'C22A_21': 2.0,
                'C26_21': 2.0,
                'C32_21': 2.0
            },
            "Hipertension": {
                'C4_21': 2.0,
                'C6_21': 1.0,
                'C12_21': 2.0,
                'C19_21': 2.0,
                'C22A_21': 2.0,
                'C26_21': 2.0,
                'C32_21': 2.0
            },
            "diabetes e hipertension": {
                'C4_21': 1.0,
                'C6_21': 1.0,
                'C12_21': 2.0,
                'C19_21': 2.0,
                'C22A_21': 2.0,
                'C26_21': 2.0,
                'C32_21': 2.0
            }
        }



        # Crear una selección en Streamlit para elegir entre los conjuntos
        seleccion = st.selectbox("Selecciona un grupo", list(conditions.keys()))

        # Crear una selección múltiple en Streamlit para el valor de SEX_18
        sex_values = df_filtrado['SEX_21'].unique()
        sex_selection = st.multiselect("Selecciona el valor de SEX_21", sex_values, default=sex_values)

        # Filtrar el DataFrame en función de las condiciones seleccionadas y el valor de SEX_18
        condiciones_seleccionadas = conditions[seleccion]
        nuevo_dataframe_filtrado = df_filtrado.copy()

        # Aplicar las condiciones seleccionadas
        for columna, valor in condiciones_seleccionadas.items():
            nuevo_dataframe_filtrado = nuevo_dataframe_filtrado[nuevo_dataframe_filtrado[columna] == valor]

        # Aplicar el filtro del valor de SEX_18
        #nuevo_dataframe_filtrado = nuevo_dataframe_filtrado[nuevo_dataframe_filtrado['SEX_18'] == sex_selection]
        # Aplicar el filtro del valor de SEX_18
        nuevo_dataframe_filtrado = nuevo_dataframe_filtrado[nuevo_dataframe_filtrado['SEX_21'].isin(sex_selection)]



        # Mostrar el DataFrame filtrado
        st.dataframe(nuevo_dataframe_filtrado)
        datos_limpios = nuevo_dataframe_filtrado.dropna()
        st.write("Las dimensiones de la base de datos son:")
        st.write(datos_limpios.shape)


##########################################

    ind=indiscernibility(['C37_21', 'H11_21', 'H15A_21', 'H5_21', 'H6_21'], datos_limpios)
    

    import matplotlib.pyplot as plt


    # Calcular las longitudes de los conjuntos con longitud >= 2
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(ind) if len(conjunto) >= 2]

    # Ordenar la lista de tuplas por la longitud de los conjuntos en orden descendente
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Crear un diccionario para mapear el índice del conjunto al nuevo nombre
    nombres_conjuntos_nuevos = {num_conjunto: f"Conjunto {i}" for i, (num_conjunto, _) in enumerate(longitudes_conjuntos_ordenadas)}

    st.write("Longitudes de los conjuntos con su respectivo nuevo nombre (en orden descendente):")
    for num_conjunto, longitud in longitudes_conjuntos_ordenadas:
        nombre_conjunto_nuevo = nombres_conjuntos_nuevos[num_conjunto]
        st.write(f"{nombre_conjunto_nuevo}: {longitud}")

    # Calcular el total de elementos en todos los conjuntos
    total_elementos = sum(longitud for _, longitud in longitudes_conjuntos_ordenadas)

    # Calcular los porcentajes para cada conjunto
    porcentajes = [(nombres_conjuntos_nuevos[num_conjunto], longitud / total_elementos * 100) for num_conjunto, longitud in longitudes_conjuntos_ordenadas]

    # Extraer los nombres de los conjuntos y los porcentajes para el diagrama de pastel
    nombres_conjuntos = [nombre for nombre, _ in porcentajes]
    porcentajes_valores = [valor for _, valor in porcentajes]

    # Crear el diagrama de pastel
    fig, ax = plt.subplots(figsize=(10, 8))
    _, _, autopcts = ax.pie(porcentajes_valores, labels=nombres_conjuntos, autopct='%1.1f%%', startangle=140, textprops={'visible': False})
    for autopct in autopcts:
        autopct.set_visible(True)  # Mostrar los porcentajes solo para los grupos con más de 30 miembros
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Distribución de subconjuntos')

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)



    # Entrada del usuario para el tamaño mínimo del conjunto
    tamaño_mínimo = st.number_input("Introduce el tamaño mínimo del conjunto:", min_value=1, value=2, step=1)

    # Calcular las longitudes de los conjuntos con longitud >= tamaño_mínimo
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(ind) if len(conjunto) >= tamaño_mínimo]

    # Ordenar la lista de tuplas por la longitud de los conjuntos en orden descendente
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Crear un diccionario para mapear el índice del conjunto al nuevo nombre
    nombres_conjuntos_nuevos = {num_conjunto: f"Conjunto {i}" for i, (num_conjunto, _) in enumerate(longitudes_conjuntos_ordenadas)}

    st.write("Longitudes de los conjuntos con su respectivo nuevo nombre (en orden descendente):")
    for num_conjunto, longitud in longitudes_conjuntos_ordenadas:
        nombre_conjunto_nuevo = nombres_conjuntos_nuevos[num_conjunto]
        st.write(f"{nombre_conjunto_nuevo}: {longitud}")

    # Calcular el total de elementos en todos los conjuntos
    total_elementos = sum(longitud for _, longitud in longitudes_conjuntos_ordenadas)

    # Calcular los porcentajes para cada conjunto
    porcentajes = [(nombres_conjuntos_nuevos[num_conjunto], longitud / total_elementos * 100) for num_conjunto, longitud in longitudes_conjuntos_ordenadas]

    # Extraer los nombres de los conjuntos y los porcentajes para el diagrama de pastel
    nombres_conjuntos = [nombre for nombre, _ in porcentajes]
    porcentajes_valores = [valor for _, valor in porcentajes]

    # Crear el diagrama de pastel
    fig, ax = plt.subplots(figsize=(10, 8))
    _, _, autopcts = ax.pie(porcentajes_valores, labels=nombres_conjuntos, autopct='%1.1f%%', startangle=140, textprops={'visible': False})
    for autopct in autopcts:
        autopct.set_visible(True)  # Mostrar los porcentajes
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Distribución de subconjuntos')

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    import numpy as np
    
    # Calcular las longitudes de los conjuntos con longitud >= tamaño_mínimo
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(ind) if len(conjunto) >= tamaño_mínimo]

    # Ordenar la lista de tuplas por la longitud de los conjuntos en orden descendente
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Obtener los 15 conjuntos más numerosos
    conjuntos_mas_numerosos = [conjunto for i, conjunto in enumerate(ind) if i in [num_conjunto for num_conjunto, _ in longitudes_conjuntos_ordenadas[:15]]]

    # Crear DataFrames para cada uno de los 15 conjuntos más numerosos
    for i, conjunto in enumerate(conjuntos_mas_numerosos, 0):
        indices_seleccionados = list(conjunto)
        df_conjunto = datos_limpios[datos_limpios.index.isin(indices_seleccionados)]
        globals()[f"df_Conjunto_{i}"] = df_conjunto

    # Definir las columnas de interés
    columnas_interes_radar = ['C37_21', 'H11_21', 'H15A_21', 'H5_21', 'H6_21']

    # Definir los nombres de los dataframes
    nombres_dataframes = [f"df_Conjunto_{i}" for i in range(0, 15)]

    # Definir los valores para cada dataframe en las columnas de interés
    valores_dataframes = []
    for nombre_df in nombres_dataframes:
        df = eval(nombre_df)
        valores = df[columnas_interes_radar].iloc[0].tolist()  # Tomar solo la primera fila
        valores_dataframes.append(valores)

    # Colores para los gráficos
    colores = plt.cm.tab20(np.linspace(0, 1, len(nombres_dataframes)))

    # Total de pacientes
    total_pacientes = len(datos_limpios)

    # Combinar nombres de DataFrames con el número de filas
    dataframes_con_filas = zip(nombres_dataframes, valores_dataframes, colores, [len(eval(nombre)) for nombre in nombres_dataframes])

    # Ordenar la lista combinada por el número de filas
    dataframes_con_filas_ordenados = sorted(dataframes_con_filas, key=lambda x: x[3], reverse=True)

    # Crear un gráfico de radar individual para cada dataframe
    num_filas = 3
    num_columnas = 5
    fig, axs = plt.subplots(num_filas, num_columnas, figsize=(20, 10), subplot_kw=dict(polar=True))
    fig.subplots_adjust(hspace=0.7, wspace=0.4)  # Ajustar el espacio horizontal

    for i, (nombre, valores, color, num_filas_df) in enumerate(dataframes_con_filas_ordenados):
        fila = i // num_columnas
        columna = i % num_columnas
        ax = axs[fila, columna]

        valores += valores[:1]  # Para cerrar el polígono
        angulos = np.linspace(0, 2 * np.pi, len(columnas_interes_radar), endpoint=False).tolist()
        angulos += angulos[:1]  # Para cerrar el polígono
        ax.plot(angulos, valores, color=color)
        ax.fill(angulos, valores, color=color, alpha=0.25)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(columnas_interes_radar)
        ax.yaxis.grid(True)
        ax.set_title(nombre)

        # Agregar el número de filas y el porcentaje debajo del gráfico
        porcentaje = (num_filas_df / total_pacientes) * 100
        ax.text(0.5, -0.2, f"Número de filas: {num_filas_df} ({porcentaje:.2f}%)", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    import pandas as pd

    # Crear un diccionario para mapear el índice de cada fila al número de conjunto
    indice_a_conjunto = {}
    for i, conjunto in enumerate(ind):
        for indice in conjunto:
            indice_a_conjunto[indice] = i

    # Agregar una nueva columna "num_conjunto" al DataFrame 'datos_limpios' usando el diccionario
    datos_limpios['num_conjunto'] = datos_limpios.index.map(indice_a_conjunto)

    # Mostrar el DataFrame con la nueva columna
#    datos_limpios

    # Seleccionar las filas que tienen valores del 0 al 14 en la columna 'num_conjunto'
    filas_seleccionadas = datos_limpios[datos_limpios['num_conjunto'].isin(range(15))]

    # Seleccionar solo las columnas requeridas
    filas_seleccionadas = filas_seleccionadas[['C37_21', 'H11_21', 'H15A_21', 'H5_21', 'H6_21', 'num_conjunto']]

    # Crear un nuevo DataFrame con las filas seleccionadas
    nuevo_dataframe = pd.DataFrame(filas_seleccionadas)

    # Mostrar las primeras filas del nuevo DataFrame
    nuevo_dataframe



#######################

#####################333

    import pandas as pd

    # Definir las condiciones para asignar los valores a la nueva columna
    def asignar_riesgo(num_conjunto):
        if num_conjunto in [4, 13]:
            return "Riesgo considerable"
        elif num_conjunto in [3, 6, 9]:
            return "Riesgo moderado"
        elif num_conjunto in [1, 2, 5, 7, 8, 10, 11, 12, 14]:
            return "Riesgo leve"
        elif num_conjunto == 0:
            return "Sin Riesgo"
        else:
            return "No clasificado"  # Manejar cualquier otro caso

    # Agregar la nueva columna al DataFrame
    nuevo_dataframe['nivel_riesgo'] = nuevo_dataframe['num_conjunto'].apply(asignar_riesgo)

    # Mostrar las primeras filas del DataFrame con la nueva columna
    nuevo_dataframe



#######################


    # Función para calcular la diferencia entre tamaños de conjuntos
    def calcular_diferencia(lista1, lista2):
        diferencia = sum(abs(len(conj1) - len(conj2)) for conj1, conj2 in zip(lista1, lista2))
        return diferencia

    # Definir las columnas de interés
    columnas_interes = ['H15A_21', 'H11_21', 'H5_21', 'H6_21', 'C37_21']

    # Generar listas de conjuntos
    lista_1 = indiscernibility(columnas_interes, nuevo_dataframe)
    lista_2_original = indiscernibility(columnas_interes, nuevo_dataframe)

    # Obtener lista de tamaños de cada conjunto
    tamaños_lista_1 = [len(conjunto) for conjunto in lista_1]
    tamaños_lista_2_original = [len(conjunto) for conjunto in lista_2_original]

    # Inicializar variables para seguimiento de la lista más parecida
    mejor_lista = lista_2_original
    mejor_diferencia = calcular_diferencia(lista_1, lista_2_original)

    # Eliminar una por una cada columna de lista_2 y mostrar los tamaños resultantes
    for columna1 in columnas_interes:
        columnas_sin_columna1 = columnas_interes.copy()
        columnas_sin_columna1.remove(columna1)
        lista_2_sin_columna1 = indiscernibility(columnas_sin_columna1, nuevo_dataframe)
        diferencia = calcular_diferencia(lista_1, lista_2_sin_columna1)
    
        if diferencia < mejor_diferencia:
            mejor_lista = lista_2_sin_columna1
            mejor_diferencia = diferencia
    
        # Eliminar pares de columnas de lista_2 y mostrar los tamaños resultantes
        for columna2 in columnas_sin_columna1:
            if columna2 != columna1:
                columnas_sin_par = columnas_sin_columna1.copy()
                columnas_sin_par.remove(columna2)
                lista_2_sin_par = indiscernibility(columnas_sin_par, nuevo_dataframe)
                diferencia = calcular_diferencia(lista_1, lista_2_sin_par)
            
                if diferencia < mejor_diferencia:
                    mejor_lista = lista_2_sin_par
                    mejor_diferencia = diferencia

    # Mostrar la mejor lista encontrada en Streamlit
    st.write("Tamaños de conjuntos en lista_1:", tamaños_lista_1)
    st.write("Tamaños de conjuntos en la mejor lista:", [len(conjunto) for conjunto in mejor_lista])

    # Visualización con un gráfico de barras
    fig, ax = plt.subplots()
    labels = [f"Conjunto {i}" for i in range(len(tamaños_lista_1))]
    x = range(len(tamaños_lista_1))
    ax.bar(x, tamaños_lista_1, width=0.4, label='lista_1', align='center')
    ax.bar(x, [len(conjunto) for conjunto in mejor_lista], width=0.4, label='Mejor Lista', align='edge')
    ax.set_xlabel('Conjuntos')
    ax.set_ylabel('Tamaños')
    ax.set_title('Comparación de tamaños de conjuntos')
    ax.legend()

    st.pyplot(fig)

    # Obtener los valores únicos de la columna 'nivel de riesgo'
    nivel_riesgo = nuevo_dataframe['nivel_riesgo'].unique()


    # Crear una barra de selección múltiple para elegir niveles de riesgo
    niveles_seleccionados = st.multiselect(
        'Selecciona los niveles de riesgo a visualizar:',
        nivel_riesgo
    )

    # Filtrar el DataFrame según los niveles de riesgo seleccionados
    if niveles_seleccionados:
        df_filtrado = nuevo_dataframe[nuevo_dataframe['nivel_riesgo'].isin(niveles_seleccionados)]
        st.write(f"Filas con nivel de riesgo en {niveles_seleccionados}:")
        st.dataframe(df_filtrado)
    else:
        st.write("Selecciona al menos un nivel de riesgo para visualizar las filas correspondientes.")
########################


    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    # Generar listas de conjuntos (simulando lista_2)
    lista_2 = indiscernibility(['C37_21', 'H11_21', 'H5_21', 'H6_21'], datos_limpios)

    # Obtener longitudes de conjuntos
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(lista_2) if len(conjunto) >= 2]

    # Ordenar por longitud
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Obtener los 15 conjuntos más numerosos
    conjuntos_mas_numerosos = [conjunto for i, conjunto in enumerate(lista_2) if i in [num_conjunto for num_conjunto, _ in longitudes_conjuntos_ordenadas[:15]]]

    # Crear DataFrames para cada uno de los 15 conjuntos más numerosos
    dataframes_con_filas = []
    for i, conjunto in enumerate(conjuntos_mas_numerosos, 0):
        indices_seleccionados = list(conjunto)
        df_conjunto = datos_limpios[datos_limpios.index.isin(indices_seleccionados)]
        dataframes_con_filas.append((f"df_Conjunto_{i}", df_conjunto, len(df_conjunto)))

    # Definir las columnas de interés
    columnas_interes_radar = ['C37_21', 'H11_21', 'H5_21', 'H6_21']

    # Definir los valores para cada dataframe en las columnas de interés
    valores_dataframes = []
    for nombre_df, df, num_filas_df in dataframes_con_filas:
        valores = df[columnas_interes_radar].iloc[0].tolist()  # Tomar solo la primera fila
        valores_dataframes.append((nombre_df, valores, num_filas_df))

    # Colores para los gráficos
    colores = plt.cm.tab20(np.linspace(0, 1, len(valores_dataframes)))

    # Total de pacientes
    total_pacientes = len(datos_limpios)

    # Ordenar la lista combinada por el número de filas
    valores_dataframes_ordenados = sorted(valores_dataframes, key=lambda x: x[2], reverse=True)

    # Configurar Streamlit
    st.title("Visualización de Conjuntos Más Numerosos")
    st.write("Comparación de los conjuntos más numerosos en los datos.")

    # Crear un gráfico de radar individual para cada dataframe
    num_filas = 3
    num_columnas = 5
    fig, axs = plt.subplots(num_filas, num_columnas, figsize=(20, 10), subplot_kw=dict(polar=True))
    fig.subplots_adjust(hspace=0.7, wspace=0.4)  # Ajustar el espacio horizontal

    for i, (nombre, valores, num_filas_df) in enumerate(valores_dataframes_ordenados):
        fila = i // num_columnas
        columna = i % num_columnas
        ax = axs[fila, columna]

        valores += valores[:1]  # Para cerrar el polígono
        angulos = np.linspace(0, 2 * np.pi, len(columnas_interes_radar), endpoint=False).tolist()
        angulos += angulos[:1]  # Para cerrar el polígono
        color = colores[i]
        ax.plot(angulos, valores, color=color)
        ax.fill(angulos, valores, color=color, alpha=0.25)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(columnas_interes_radar)
        ax.yaxis.grid(True)
        ax.set_title(nombre)

        # Agregar el número de filas y el porcentaje debajo del gráfico
        porcentaje = (num_filas_df / total_pacientes) * 100
        ax.text(0.5, -0.2, f"Número de filas: {num_filas_df} ({porcentaje:.2f}%)", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)


######################


    # Título de la aplicación
    st.title("Filtrado de DataFrame por tamaño de conjuntos")

    # Calcular el tamaño de cada conjunto y filtrar los conjuntos con menos de 30 miembros
    conjuntos_mayores_30 = [conjunto for conjunto in ind if len(conjunto) >= 30]

    # Obtener los índices de las filas que pertenecen a los conjuntos mayores o iguales a 30 miembros
    indices_filtrados = [indice for conjunto in conjuntos_mayores_30 for indice in conjunto]

    # Filtrar el DataFrame 'datos_limpios' para mantener solo las filas con índices en 'indices_filtrados'
    datos_limpios_filtrados = datos_limpios.loc[indices_filtrados]

    # Mostrar el DataFrame filtrado en Streamlit
    st.write("DataFrame filtrado con conjuntos mayores o iguales a 30 miembros:")
    st.dataframe(datos_limpios_filtrados)

    # Mostrar el número total de conjuntos filtrados
    st.write(f"Número de conjuntos mayores o iguales a 30 miembros: {len(conjuntos_mayores_30)}")

    # Mostrar el tamaño de cada conjunto filtrado
    st.write("Tamaño de cada conjunto filtrado:")
    tamaños_conjuntos = [len(conjunto) for conjunto in conjuntos_mayores_30]
    st.write(tamaños_conjuntos)



#######################

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


    # Título de la aplicación
    st.title("Clasificador de Árbol de Decisión")

    # Definir las columnas de atributos
    columnas_atributos = ['C37_21', 'H11_21', 'H5_21', 'H6_21']

    # Separar los datos en atributos (X) y etiquetas (y)
    X = nuevo_dataframe[columnas_atributos]
    y = nuevo_dataframe['nivel_riesgo']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el clasificador de árbol de decisión
    clf = DecisionTreeClassifier()

    # Entrenar el clasificador
    clf.fit(X_train, y_train)

    # Predecir las etiquetas para los datos de prueba
    y_pred = clf.predict(X_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)

    # Mostrar la precisión del modelo en Streamlit
    st.write(f'Precisión del modelo: {accuracy:.2f}')

    # Mostrar el reporte de clasificación en Streamlit
    st.subheader("Reporte de Clasificación:")
    st.text(classification_report(y_test, y_pred))

    # Mostrar la matriz de confusión en Streamlit
    st.subheader("Matriz de Confusión:")
    st.write(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=clf.classes_, index=clf.classes_))

##################

    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Título de la aplicación
    st.title("Visualización del Árbol de Decisión")

    # Definir las columnas de atributos
    columnas_atributos = ['C37_21', 'H11_21', 'H5_21', 'H6_21']

    # Separar los datos en atributos (X) y etiquetas (y)
    X = nuevo_dataframe[columnas_atributos]
    y = nuevo_dataframe['nivel_riesgo']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el clasificador de árbol de decisión
    clf = DecisionTreeClassifier(random_state=42)

    # Entrenar el clasificador
    clf.fit(X_train, y_train)

    # Visualizar el árbol de decisión en Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Para evitar advertencias de Streamlit
    plt.figure(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=columnas_atributos, class_names=clf.classes_)
    st.pyplot()
##################

    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    # Suponiendo que 'nuevo_dataframe' ya está definido y contiene los datos necesarios

#    def asignar_nivel_riesgo(df, modelo, columnas_atributos):
#        # Hacer una copia del DataFrame para evitar modificar el original
#        df = df.copy()
    
#        # Preprocesamiento de los datos de entrada si es necesario
#        X = df[columnas_atributos]
    
#        # Predecir los niveles de riesgo para los datos de entrada
#        y_pred = modelo.predict(X)
    
        # Asignar los resultados al DataFrame en una nueva columna
#        df['Diagnóstico_árbol'] = y_pred
    
#        return df

    import streamlit as st
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    # Definir las columnas de atributos
    columnas_atributos = ['C37_21', 'H11_21', 'H5_21', 'H6_21']

    # Suponiendo que 'datos_limpios' ya está definido y contiene los datos necesarios

    # Función para asignar nivel de riesgo a una fila
    def asignar_nivel_riesgo(fila, modelo, columnas_atributos):
        X = fila[columnas_atributos].values.reshape(1, -1)
        y_pred = modelo.predict(X)
        return y_pred[0]

#    # Entrenar el modelo
#    X = datos_limpios[columnas_atributos]
#    y = datos_limpios['nivel_riesgo']
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#    clf = DecisionTreeClassifier(random_state=42)
#    clf.fit(X_train, y_train)

    # Visualizar el árbol de decisión
    #fig, ax = plt.subplots(figsize=(15, 10))
    #plot_tree(clf, filled=True, feature_names=columnas_atributos, class_names=clf.classes_, ax=ax)
    #st.pyplot(fig)

    # Aplicar la función asignar_nivel_riesgo al DataFrame datos_limpios
    #datos_limpios['Diagnóstico_árbol'] = datos_limpios.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    nuevo_dataframe['Diagnóstico_árbol'] = nuevo_dataframe.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)

    # Mostrar los resultados en Streamlit
    st.write("Resultados de asignación de nivel de riesgo:")
    st.dataframe(nuevo_dataframe[['C37_21', 'H11_21', 'H6_21', 'H5_21', 'Diagnóstico_árbol']])

    # Calcular el número de coincidencias y no coincidencias
    coincidencias = (nuevo_dataframe['nivel_riesgo'] == nuevo_dataframe['Diagnóstico_árbol']).sum()
    total_filas = len(datos_limpios)
    no_coincidencias = total_filas - coincidencias

    # Mostrar los resultados en Streamlit
    st.write(f"Número de filas en las que coinciden los valores: {coincidencias}")
    st.write(f"Número de filas en las que no coinciden los valores: {no_coincidencias}")
 
    datos_limpios['Diagnóstico_árbol'] = datos_limpios.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    # Aplicar la función asignar_nivel_riesgo al dataframe datos_limpios
    datos_limpios[['C37_21','H11_21', 'H6_21','H5_21','Diagnóstico_árbol']]

    #datos_filtrados.drop('H15A_18', axis=1, inplace=True) 
    datos_filtrados = datos_filtrados.dropna()
 
    datos_filtrados['Diagnóstico_árbol'] = datos_filtrados.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    datos_filtrados = datos_filtrados[['H11_21', 'H5_21', 'H6_21','C37_21','Diagnóstico_árbol']].dropna()
    datos_filtrados[['H11_21', 'H5_21', 'H6_21','C37_21','Diagnóstico_árbol']]

    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Suponiendo que 'datos_filtrados' ya está definido y contiene los datos necesarios

    # Agrupar por 'Diagnóstico_árbol' y contar el número de ocurrencias para nuevo_dataframe
    grupo_diagnostico_nuevo = nuevo_dataframe.groupby('Diagnóstico_árbol').size()

    # Agrupar por 'Diagnóstico_árbol' y contar el número de ocurrencias para datos_filtrados
    grupo_diagnostico_filtrados = datos_filtrados.groupby('Diagnóstico_árbol').size()

    # Crear el panel con dos subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico para nuevo_dataframe
    sns.barplot(x=grupo_diagnostico_nuevo.values, y=grupo_diagnostico_nuevo.index, palette='Dark2', ax=axes[0])
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].set_xlabel('Número de filas')
    axes[0].set_ylabel('Diagnóstico')
    axes[0].set_title('Conteo de diagnósticos (nuevo_dataframe)')

    # Gráfico para datos_filtrados
    sns.barplot(x=grupo_diagnostico_filtrados.values, y=grupo_diagnostico_filtrados.index, palette='Dark2', ax=axes[1])
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].set_xlabel('Número de filas')
    axes[1].set_ylabel('Diagnóstico')
    axes[1].set_title('Conteo de diagnósticos (datos_filtrados)')

    # Mostrar el panel con subplots en Streamlit
    st.pyplot(fig)

# Crear una nueva columna "Diagnóstico_árbol" en el dataframe "nuevo_dataframe"
#nuevo_dataframe['Diagnóstico_árbol'] = nuevo_dataframe.apply(asignar_nivel_riesgo, axis=1)

    nuevo_dataframe.shape

    datos_filtrados.shape





##################



elif option == "Equipo de trabajo":

    st.subheader("Equipo de Trabajo")

    # Información del equipo
    equipo = [{
            "nombre": "Dr. Santiago Arceo Díaz",
            "foto": "ArceoS.jpg",
            "reseña": "Licenciado en Física, Maestro en Física y Doctor en Ciencias (Astrofísica). Posdoctorante de la Universidad de Colima y profesor del Tecnológico Nacional de México Campus Colima. Cuenta con el perfil deseable, pertenece al núcleo académico y es colaborador del cuerpo académico Tecnologías Emergentes y Desarrollo Web de la Maestría Sistemas Computacionales. Ha dirigido tesis de la Maestría en Sistemas Computacionales y en la Maestría en Arquitectura Sostenible y Gestión Urbana.",
            "CV": "https://scholar.google.com.mx/citations?user=3xPPTLoAAAAJ&hl=es", "contacto": "pv_sarceo@ucol.mx"},
        {
            "nombre": "José Ramón González",
            "foto": "JR.jpeg",
            "reseña": "Estudiante de la facultad de medicina en la Universidad de Colima, cursando el servicio social en investigación en el Centro Universitario de Investigaciones Biomédicas, bajo el proyecto Aplicación de un software basado en modelos predictivos como herramienta de apoyo en el diagnóstico de sarcopenia en personas adultas mayores a partir de parámetros antropométricos.", "CV": "https://scholar.google.com.mx/citations?user=3xPPTLoAAAAJ&hl=es", "contacto": "jgonzalez90@ucol.mx"},
        {
            "nombre": "Dra. Xochitl Angélica Rosío Trujillo Trujillo",
            "foto": "DraXochilt.jpg",
            "reseña": "Bióloga, Maestra y Doctora en Ciencias Fisiológicas con especialidad en Fisiología. Es Profesora-Investigadora de Tiempo Completo de la Universidad de Colima. Cuenta con perfil deseable y es miembro del Sistema Nacional de Investigadores en el nivel 3. Su línea de investigación es en Biomedicina en la que cuenta con una producción científica de más de noventa artículos en revistas internacionales, varios capítulos de libro y dos libros. Imparte docencia y ha formado a más de treinta estudiantes de licenciatura y de posgrado en programas académicos adscritos al Sistema Nacional de Posgrado del CONAHCYT.",
        "CV": "https://portal.ucol.mx/cuib/XochitlTrujillo.htm", "contacto": "rosio@ucol.mx"},
            {
            "nombre": "Dr. Miguel Huerta Viera",
            "foto": "DrHuerta.jpg",
            "reseña": "Doctor en Ciencias con especialidad en Fisiología y Biofísica. Es Profesor-Investigador Titular “C” del Centro Universitario de Investigaciones Biomédicas de la Universidad de Colima. Es miembro del Sistema Nacional de Investigadores en el nivel 3 emérito. Su campo de investigación es la Biomedicina, con énfasis en la fisiología y biofísica del sistema neuromuscular y la fisiopatología de la diabetes mellitus. Ha publicado más de cien artículos revistas indizadas al Journal of Citation Reports y ha graduado a más de 40 Maestros y Doctores en Ciencias en programas SNP-CONAHCyT.",
            "CV": "https://portal.ucol.mx/cuib/dr-miguel-huerta.htm", "contacto": "huertam@ucol.mx"},
       {
            "nombre": "Dr. Jaime Alberto Bricio Barrios",
            "foto":  "BricioJ.jpg",
            "reseña": "Licenciado en Nutrición, Maestro en Ciencias Médicas, Maestro en Seguridad Alimentaria y Doctor en Ciencias Médicas. Profesor e Investigador de Tiempo Completo de la Facultad de Medicina en la Universidad de Colima. miembro del Sistema Nacional de Investigadores en el nivel 1. Miembro fundador de la asociación civil DAYIN (Desarrollo de Ayuda con Investigación)",
    "CV": "https://scholar.google.com.mx/citations?hl=es&user=ugl-bksAAAAJ", "contacto": "jbricio@ucol.mx"},      
            {
            "nombre": "Mtra. Elena Elsa Bricio Barrios",
            "foto": "BricioE.jpg",
            "reseña": "Química Metalúrgica, Maestra en Ciencias en Ingeniería Química y doctorante en Ingeniería Química. Actualmente es profesora del Tecnológico Nacional de México Campus Colima. Cuenta con el perfil deseable, es miembro del cuerpo académico Tecnologías Emergentes y Desarrollo Web y ha codirigido tesis de la Maestría en Sistemas Computacionales.",
    "CV": "https://scholar.google.com.mx/citations?hl=es&user=TGZGewEAAAAJ", "contacto": "elena.bricio@colima.tecnm.mx"}
    ]

    #       Establecer la altura deseada para las imágenes
    altura_imagen = 150  # Cambia este valor según tus preferencias

    # Mostrar información de cada miembro del equipo
    for miembro in equipo:
        st.subheader(miembro["nombre"])
        img = st.image(miembro["foto"], caption=f"Foto de {miembro['nombre']}", use_column_width=False, width=altura_imagen)
        st.write(f"Correo electrónico: {miembro['contacto']}")
        st.write(f"Reseña profesional: {miembro['reseña']}")
        st.write(f"CV: {miembro['CV']}")

    # Información de contacto
    st.subheader("Información de Contacto")
    st.write("Si deseas ponerte en contacto con nuestro equipo, puedes enviar un correo a santiagoarceodiaz@ucol.mx")

