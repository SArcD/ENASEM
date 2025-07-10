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
#option = st.sidebar.selectbox("Seleccione una pestaña", ["Introducción", "Filtrar datos", "Buscador de variables", "Buscador de datos", "Relaciones de Indiscernibilidad 2018", "Relaciones de Indiscernibilidad 2021", "Equipo de trabajo"])
option = st.sidebar.selectbox("Seleccione una pestaña", ["Introducción", "Filtrar datos", "Buscador de variables", "Relaciones de Indiscernibilidad 2018", "Relaciones de Indiscernibilidad 2021", "Equipo de trabajo"])

if option == "Introducción":
    #
    st.title("Analizador ENASEM-RS")
    st.header("Sobre el envejecimiento en México")

    st.markdown("""
<div style="text-align: justify">
Al igual que para otros países, la tendencia actual para la distribución por grupos de edad en  
<a href="https://www.inegi.org.mx/temas/estructura/" target="_blank"><strong>México</strong></a> indica que en el futuro cercano la población de personas adultas mayores será considerablemente superior que la de personas jóvenes. De acuerdo con un estudio publicado por el <a href="https://www.cepal.org/es/enfoques/panorama-envejecimiento-tendencias-demograficas-america-latina-caribe" target="_blank"><strong>Centro Latinoamericano y Caribeño de Demografía (CELADE) – CEPAL</strong></a>, esto podría ocurrir en el año 2027 y, si la tendencia continúa, para el año 2085 la población de personas adultas mayores podría llegar a 48 millones (<a href="https://www.cepal.org/es/enfoques/panorama-envejecimiento-tendencias-demograficas-america-latina-caribe" target="_blank"><strong>CEPAL, 2023</strong></a>). Debido a lo anterior, las <a href="https://www.gob.mx/inapam/articulos/calidad-de-vida-para-un-envejecimiento-saludable?idiom=es" target="_blank"><strong>estrategias de prevención de enfermedades y calidad de vida para un envejecimiento saludable</strong></a> se volverán cada vez más relevantes.
</div>
""", unsafe_allow_html=True)


    
    st.subheader("**La Encuesta Nacional Sobre Envejecimiento en México**")     
    st.markdown("""
<div style="text-align: justify;">
<a href="https://enasem.org/DataProducts/ImputedData_Esp.aspx" target="_blank"><strong>La ENASEM (Encuesta Nacional Sobre Envejecimiento en México)</strong></a> es uno de los estudios de mayor escala en la recolección de información sobre el estado de salud de las personas adultas mayores. Este estudio longitudinal, desarrollado por el 
<a href="https://www.inegi.org.mx/" target="_blank"><strong>Instituto Nacional de Estadística y Geografía (INEGI)</strong></a>, en colaboración con el 
<a href="https://www.utmb.edu/" target="_blank"><strong>Centro Médico de la Universidad de Texas (UTMB)</strong></a>, el 
<a href="https://www.inger.gob.mx/" target="_blank"><strong>Instituto Nacional de Geriatría (INGER)</strong></a> y el 
<a href="https://www.insp.mx/" target="_blank"><strong>Instituto Nacional de Salud Pública (INSP)</strong></a>, tiene como objetivo actualizar y dar seguimiento a la información estadística recabada en los levantamientos sobre la población de 50 años y más en México, con representación urbana y rural.
</div>
""", unsafe_allow_html=True)

    
    st.markdown("""
<div style="text-align: justify;">
<strong>La ENASEM</strong> forma parte de una familia global de estudios longitudinales que tratan de entender el proceso de envejecimiento humano bajo distintas condiciones de vida. En Estados Unidos se lleva a cabo el 
<a href="https://hrs.isr.umich.edu/" target="_blank"><strong>“Health and Retirement Study (HRS)”</strong></a>, en Brasil el 
<a href="https://www.elsi.cpqrr.fiocruz.br/" target="_blank"><strong>“Estudo Longitudinal da Saúde dos Idosos Brasileiros (ELSI-Brasil)”</strong></a> y en la Unión Europea, 
<a href="https://www.share-project.org/" target="_blank"><strong>“The Survey of Health, Ageing and Retirement in Europe (SHARE)”</strong></a>. La información recabada es fundamental para la creación de estrategias que permitan la mitigación de condiciones debilitantes para las personas adultas mayores, tales como los síndromes geriátricos.
</div>
""", unsafe_allow_html=True)

    st.subheader("**Los síndromes geriátricos**")     
    st.markdown("""
<div style="text-align: justify;">
Es un conjunto de cuadros clínicos, signos y síntomas frecuentes en personas adultas mayores, sobre todo después de los 65 años. Estos tienen que ver más con la interacción entre el desgaste causado por el envejecimiento y múltiples patologías que con enfermedades en sí mismas. La consecuencia principal de esto es la reducción progresiva de la capacidad funcional y el deterioro progresivo de la salud, así como el incremento de la polifarmacia. Típicamente los <a href="https://postgradomedicina.com/sindromes-geriatricos-causas-tratamiento/" target="_blank"><strong>síndromes geriátricos</strong></a> tienen una alta prevalencia en la población de personas adultas mayores, y suele acentuarse si coinciden con <a href="https://www.who.int/es/news-room/fact-sheets/detail/noncommunicable-diseases" target="_blank"><strong>enfermedades crónicas</strong></a> o lo padecen personas institucionalizadas. Generan un deterioro progresivo de la autonomía, capacidad funcional e incrementan la necesidad de cuidados específicos. Su aparición puede agravar los daños que ya causan otras comorbilidades y requieren un tratamiento integral (como cuidados <a href="https://medlineplus.gov/spanish/nutritionforolderadults.html" target="_blank"><strong>nutricionales</strong></a>, <a href="https://www.gob.mx/inapam/es/articulos/gerontologia-una-respuesta-al-envejecimiento?idiom=es" target="_blank"><strong>médicos</strong></a> y <a href="https://www.gob.mx/inapam/articulos/salud-mental-en-personas-mayores?idiom=es" target="_blank"><strong>psicológicos</strong></a>). Algunos de los síndromes geriátricos más comunes son el <a href="https://mimocare.net/blog/deterioro-cognitivo-en-el-adulto-mayor/" target="_blank"><strong>deterioro cognitivo</strong></a>, la <a href="https://www.imss.gob.mx/sites/all/statics/guiasclinicas/479GRR_0.pdf" target="_blank"><strong>fragilidad</strong></a> y la <a href="https://www.gob.mx/salud/articulos/que-es-la-sarcopenia" target="_blank"><strong>sarcopenia</strong></a>.
</div>
""", unsafe_allow_html=True)


    
    st.subheader("Sarcopenia")
    st.markdown("""
<div style="text-align: justify;">
La <a href="https://www.who.int/health-topics/ageing#tab=tab_1" target="_blank"><strong>sarcopenia</strong></a> es uno de los <a href="https://postgradomedicina.com/sindromes-geriatricos-causas-tratamiento/" target="_blank"><strong>síndromes geriátricos</strong></a> más comunes. Su definición tradicional implica una alteración progresiva y generalizada del músculo esquelético, caracterizada por una pérdida acelerada de masa y función muscular. La incidencia prolongada de sarcopenia en personas adultas mayores puede correlacionarse con la aparición de <a href="https://iris.who.int/handle/10665/186463" target="_blank"><strong>deterioro funcional</strong></a>, <a href="https://www.ncbi.nlm.nih.gov/books/NBK560761/" target="_blank"><strong>caídas</strong></a>, <a href="https://www.imss.gob.mx/sites/all/statics/guiasclinicas/479GRR_0.pdf" target="_blank"><strong>fragilidad</strong></a> y un aumento en la mortalidad (<a href="https://dialnet.unirioja.es/servlet/articulo?codigo=8551376" target="_blank"><strong>Montero-Errasquín & Cruz-Jentoft, 2022</strong></a>). Además, la sarcopenia incrementa la predisposición a <a href="https://doi.org/10.1016/j.arr.2011.03.003" target="_blank"><strong>comorbilidades</strong></a>, añadiendo una capa adicional de complejidad a la gestión de la salud en el contexto geriátrico (<a href="https://pubmed.ncbi.nlm.nih.gov/30312372/" target="_blank"><strong>Cruz-Jentoft et al., 2019</strong></a>).
</div>
""", unsafe_allow_html=True)


    
    st.subheader("Comorbilidades asociadas a la sarcopenia")

    tab1, tab2, = st.tabs(["Diabetes Mellitus 2", "Hipertensión arterial"])
        
    with tab1:
        st.header("Diabetes Mellitus Tipo 2")

        st.markdown("""
<div style="text-align: justify;">
La <a href="https://www.mayoclinic.org/es/diseases-conditions/type-2-diabetes/symptoms-causes/syc-20351193" target="_blank"><strong>diabetes mellitus tipo 2</strong></a> es una enfermedad crónica que ocurre cuando el páncreas no produce suficiente insulina o cuando el organismo no utiliza de forma eficaz la insulina disponible. La insulina es una hormona esencial que regula los niveles de glucosa en sangre. La hiperglucemia —es decir, el aumento sostenido de glucosa en sangre— es un efecto común de la diabetes no controlada y, con el tiempo, puede provocar daños graves en muchos sistemas del cuerpo, especialmente en los nervios y vasos sanguíneos.
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div style="text-align: justify;">
En 2014, se estimaba que en la región de las Américas el <strong>8,3%</strong> de los adultos mayores de 18 años tenía diabetes (<strong>8,5%</strong> a nivel mundial). Para 2019, la diabetes fue la causa directa de aproximadamente <strong>284,000 muertes</strong> en la región y se calcula que el <strong>44%</strong> de todas las muertes por diabetes ocurrieron antes de los 70 años. A nivel mundial, la cifra fue de <strong>1,5 millones de muertes</strong>, de las cuales casi la mitad ocurrieron antes de los 70 años (<a href="https://www.paho.org/es/noticias/11-11-2022-numero-personas-con-diabetes-americas-se-ha-triplicado-tres-decadas-segun" target="_blank"><strong>OPS, 2021</strong></a>). Con el tiempo, la diabetes puede dañar el corazón, los vasos sanguíneos, los ojos, los riñones y los nervios.
</div>
""", unsafe_allow_html=True)


        st.subheader("Impacto en la salud")
        st.markdown(""" <div style="text-align: justify;"> 
            
- Los adultos con diabetes tienen un riesgo dos o tres veces mayor de sufrir ataques cardíacos y accidentes cerebrovasculares.

- Combinado con un flujo sanguíneo reducido, la neuropatía (daño a los nervios) en los pies aumenta la posibilidad de úlceras en el pie, infección y eventual necesidad de amputación de una extremidad.

- La retinopatía diabética es una causa importante de ceguera y se produce como resultado del daño acumulado a largo plazo en los pequeños vasos sanguíneos de la retina. Cerca de 1 millón de personas son ciegas debido a la diabetes.

- La diabetes es una de las principales causas de insuficiencia renal.

https://www.paho.org/es/temas/diabetes
.""",  unsafe_allow_html=True)
               
                    
    with tab2:
            st.header("Hipertensión arterial")
            st.markdown(""" <div style="text-align: justify;"> La hipertensión arterial, definida como presión arterial sistólica igual o superior a 140 mmHg o presión arterial diastólica igual o superior a 90 mmHg, es uno de los factores de riesgo más importantes para las enfermedades cardiovasculares y la enfermedad renal crónica. La presión arterial es un rasgo multifacético, afectado por la nutrición, el medio ambiente y el comportamiento a lo largo del curso de la vida, incluida la nutrición y el crecimiento fetal y la infancia, la adiposidad, los componentes específicos de la dieta, especialmente la ingesta de sodio y potasio, el consumo de alcohol, el tabaquismo y la actividad física, la contaminación del aire, el plomo, el ruido, el estrés psicosocial y el uso de medicamentos para bajar la presión arterial.""",  unsafe_allow_html=True)
            
            st.subheader("Impacto en la salud")
            st.markdown(""" <div style="text-align: justify;">
            La hipertensión es un trastorno médico grave que puede incrementar el riesgo de enfermedades cardiovasculares, cerebrales, renales y otras. Esta importante causa de defunción prematura en todo el mundo afecta a más de uno de cada cuatro hombres y una de cada cinco mujeres, o sea, más de 1000 millones de personas. La carga de morbilidad por hipertensión es desproporcionadamente alta en los países de ingresos bajos y medianos, en los que se registran dos terceras partes de los casos, debido en gran medida al aumento de los factores de riesgo entre esas poblaciones en los últimos decenios. 
            https://www.paho.org/es/enlace/hipertension
            """,  unsafe_allow_html=True)

elif option == "Filtrar datos":
    st.header("Extracción de datos a partir de la ENASEM")
    st.markdown(""" En esta sección puede cargar algunos de los conjuntos de datos de la ENASEM (ya sea de las ediciones de 2018 o de 2021). En el menú desplegable puede seleccionar el archivo a cargar. </div> """,  unsafe_allow_html=True)
    st.write("")  # Esto agrega un espacio en blanco

    # Menú desplegable para elegir el archivo
    selected_file = st.selectbox("**Selecciona un archivo CSV**", list(file_urls.keys()))

    if selected_file:
        # Cargar el archivo seleccionado
        data = load_csv_from_drive(file_urls[selected_file])
        
        st.write(f"**Archivo seleccionado:** {selected_file}")
        st.write(data)
        
        # Lista de verificación para seleccionar columnas
        st.markdown(""" <div style="text-align: justify;"> A continuación puede generar una base de datos a partir de las columnas que seleccione del menú desplegable. Una vez seleccionadas podrá visualizar la base de datos y descargarla en formato .csv o .xlsx al presionar cualquiera de los botones de descarga. </div> """,  unsafe_allow_html=True)
        st.write("")  # Esto agrega un espacio en blanco
        selected_columns = st.multiselect("**Selecciona las columnas para mostrar**", data.columns.tolist())
        
        if selected_columns:
            # Crear dataframe reducido
            reduced_data = data[selected_columns]
            st.write("")  # Esto agrega un espacio en blanco
            st.write("**Base de datos con las columnas seleccionadas:**")
            st.dataframe(reduced_data, use_container_width=True)

            with st.expander("**Información adicional**"):
                # Mostrar información del dataframe reducido
                num_rows, num_cols = reduced_data.shape
                st.write(f"**Número de filas**: {num_rows}")
                st.write(f"**Número de columnas**: {num_cols}")
            
                # Contar valores NaN por columna
                nan_counts = reduced_data.isna().sum().reset_index()
                nan_counts.columns = ["Clave", "Conteo"]
            
                st.write("**Conteo de valores NaN por columna:**")
                st.write(nan_counts)

            # Botón para descargar el dataframe reducido en formato csv
            csv_data = convert_df_to_csv(reduced_data)
            st.download_button(
                label="**Descargar Dataframe en formato CSV**",
                data=csv_data,
                file_name="dataframe_reducido.csv",
                mime="text/csv"
            )

            xlsx_data = convert_df_to_xlsx(reduced_data)
            st.download_button(
                label="**Descargar Dataframe en formato XLSX**",
                data=xlsx_data,
                file_name="dataframe_reducido.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


    st.subheader("Unir dataframes")

    st.markdown("""<div style="text-align: justify;"> En esta sección puede unir dos archivos .csv para formar una base de datos mas grande (recuerde seleccionar archivos que correspondan al mismo año). La base de datos se mostrará abajo, así como información sobre el conteo de filas con columnas vacías </div> """,  unsafe_allow_html=True)
    st.write("")  # Esto agrega un espacio en blanco
    # Seleccionar dos archivos CSV para unir
    selected_files = st.multiselect("**Selecciona dos archivos CSV para unir**", list(file_urls.keys()), default=None, max_selections=2)

    if len(selected_files) == 2:
        # Cargar los dos archivos seleccionados
        df1 = load_csv_from_drive(file_urls[selected_files[0]])
        df2 = load_csv_from_drive(file_urls[selected_files[1]])
        
        if df1 is not None and df2 is not None:
            # Unir los dataframes usando la columna 'CUNICAH'
            merged_data = pd.merge(df1, df2, on='CUNICAH', how='inner')
            
            st.write("**Base de datos unida**:")
            st.dataframe(merged_data, use_container_width=True)

            with st.expander("**Información adicional**"):
                # Mostrar información del dataframe reducido
                num_rows, num_cols = merged_data.shape
                st.write(f"**Número de filas**: {num_rows}")
                st.write(f"**Número de columnas**: {num_cols}")
            
                # Contar valores NaN por columna
                nan_counts = merged_data.isna().sum().reset_index()
                nan_counts.columns = ["Clave", "Conteo"]
            
                st.write("**Conteo de valores NaN por columna:**")
                st.write(nan_counts)
            

            # Botón para descargar el dataframe reducido en formato csv
            csv_data = convert_df_to_csv(merged_data)
            st.download_button(
                label="**Descargar Dataframe en formato CSV**",
                data=csv_data,
                file_name="dataframe_unificado.csv",
                mime="text/csv"
            )
            
            # Botón para descargar el dataframe unido en formato CSV
            csv_data = convert_df_to_csv(merged_data)
            st.download_button(
                label="**Descargar Dataframe unido en formato CSV**",
                data=csv_data,
                file_name="dataframe_unido.csv",
                mime="text/csv"
            )

        st.subheader("Selección de columnas")
        st.markdown("""<div style="text-align: justify;"> A continuación puede generar una base de datos a partir de las columnas que seleccione del menú desplegable. Una vez seleccionadas podrá visualizar la base de datos y descargarla en formato .csv o .xlsx al presionar cualquiera de los botones de descarga. </div> """,  unsafe_allow_html=True)
    # Seleccionar dos archivos CSV para unir
        # Lista de verificación para seleccionar columnas
        st.write("")  # Esto agrega un espacio en blanco
        selected_columns = st.multiselect("**Selecciona las columnas para mostrar**", merged_data.columns.tolist())
        
        if selected_columns:
            # Crear dataframe reducido
            reduced_merged_data = merged_data[selected_columns]
            
            st.write("**Base de datos:**")
            st.dataframe(reduced_merged_data, use_container_width=True)

            with st.expander("**Información adicional**"):
                # Mostrar información del dataframe reducido
                num_rows, num_cols = reduced_merged_data.shape
                st.write(f"**Número de filas**: {num_rows}")
                st.write(f"**Número de columnas**: {num_cols}")
            
                # Contar valores NaN por columna
                nan_counts = reduced_merged_data.isna().sum().reset_index()
                nan_counts.columns = ["Clave", "Conteo"]
            
                st.write("**Conteo de valores NaN por columna:**")
                st.write(nan_counts)

            
            # Botón para descargar el dataframe reducido en formato csv
            csv_data = convert_df_to_csv(reduced_merged_data)
            st.download_button(
                label="**Descargar Dataframe en formato CSV**",
                data=csv_data,
                file_name="dataframe_unificado_reducido.csv",
                mime="text/csv"
            )

            xlsx_data = convert_df_to_xlsx(reduced_merged_data)
            st.download_button(
                label="**Descargar Dataframe en formato XLSX**",
                data=xlsx_data,
                file_name="dataframe_unificado_reducido.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
elif option == "Buscador de variables":


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
    st.title("Buscador de Variables por año")
    st.markdown("""<div style="text-align: justify;"> En esta sección puede visualizar la explicación de cualquiera de las claves para las variables de la ENASEM (ya sea de la edición 2018 o 2021). Primero, use el menú desplegable para seleccionar el año del diccionario a consultar. </div> """,  unsafe_allow_html=True)
    # Inicializar el estado de la sesión para el historial de búsquedas
    if 'historico_busquedas' not in st.session_state:
        st.session_state.historico_busquedas = pd.DataFrame(columns=['Año', 'Código', 'Explicación'])

    # Barra de selección múltiple para elegir el año
    años_seleccionados = st.multiselect('**Selecciona el año del diccionario**', nombres_diccionarios)

    # Si se seleccionan años, cargar los diccionarios correspondientes
    diccionarios = {}
    for año in años_seleccionados:
        url = urls[año]
        diccionarios[año] = cargar_diccionario(url, f'diccionario_{año}')

    # Interfaz de búsqueda por código en los diccionarios seleccionados
    if años_seleccionados:
        codigo_busqueda = st.text_input("**Ingrese el código de la variable (por ejemplo, AA21_21):**")
        if codigo_busqueda:
            for año, diccionario in diccionarios.items():
                explicacion = diccionario.get(codigo_busqueda, None)
                if explicacion:
                    st.write(f"**Explicación para el código {codigo_busqueda} en {año}**: {explicacion}")
                    # Agregar la búsqueda al histórico
                    nueva_fila = pd.DataFrame([[año, codigo_busqueda, explicacion]], columns=['Año', 'Código', 'Explicación'])
                    st.session_state.historico_busquedas = pd.concat([st.session_state.historico_busquedas, nueva_fila], ignore_index=True)
                else:
                    st.write(f"**No se encontró explicación para el código {codigo_busqueda} en {año}.**")
            # Mostrar el histórico de búsquedas
            st.dataframe(st.session_state.historico_busquedas, use_container_width=True)
        else:
            st.write("**Por favor, ingrese un código de variable.**")
    else:
        st.write("**Por favor, selecciona al menos un año.**")


#elif option == "Buscador de datos":

#    st.title('Filtrar DataFrame por Columnas')

#    # Crear una caja de carga de archivos
#    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

#    if uploaded_file is not None:
#        # Cargar el archivo como un DataFrame de pandas
#        df = pd.read_csv(uploaded_file)

#    # Mostrar el DataFrame cargado
#    st.write('DataFrame cargado:')
#    st.dataframe(df)

#    # Opcional: mostrar estadísticas básicas del DataFrame
#    st.write('Descripción del DataFrame:')
#    st.write(df.describe())

#    # Lista de verificación para seleccionar columnas
#    selected_columns = st.multiselect("Selecciona las columnas para mostrar", df.columns.tolist())
        
#    if selected_columns:
#        # Crear dataframe reducido
#        df = df[selected_columns]
            
#        st.write("Dataframe reducido:")
#        st.write(df)
            
#        # Mostrar información del dataframe reducido
#        num_rows, num_cols = df.shape
#        st.write(f"Número de filas: {num_rows}")
#        st.write(f"Número de columnas: {num_cols}")
            
#        # Contar valores NaN por columna
#        nan_counts = df.isna().sum().reset_index()
#        nan_counts.columns = ["Clave", "Conteo"]
            
#        st.write("Conteo de valores NaN por columna:")
#        st.write(nan_counts)

    
#        columnas_seleccionadas = list(df.columns)
        
        
#        # Crear widgets de selección para cada columna seleccionada
#        filtros = {}
#        for col in columnas_seleccionadas:
#            if df[col].dtype == 'object':
#                valores_unicos = df[col].unique().tolist()
#                seleccion = st.multiselect(f'Seleccionar valores para {col}', valores_unicos)
#                if seleccion:
#                    filtros[col] = seleccion
#            else:
#                rango = st.slider(f'Seleccionar rango para {col}', min_value=float(df[col].min()), max_value=float(df[col].max()), value=(float(df[col].min()), float(df[col].max())), step=1.0)
#                if rango:
#                    filtros[col] = rango

#        # Filtrar el DataFrame basado en los valores seleccionados
#        df_filtrado = df.copy()
#        for col, condicion in filtros.items():
#            if isinstance(condicion, list):
#                df_filtrado = df_filtrado[df_filtrado[col].isin(condicion)]
#            else:
#                df_filtrado = df_filtrado[(df_filtrado[col] >= condicion[0]) & (df_filtrado[col] <= condicion[1])]

#        st.write('DataFrame Filtrado')
        #st.dataframe(df_filtrado)
##################

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



    st.title('Estimación del nivel de riesgo por sarcopenia')

    st.markdown(
    """
    <div style="text-align: justify;">
        En esta sección se calcula el <strong>riesgo de padecer sarcopenia</strong> a partir de las respuestas de las y los participantes de la <a href="https://enasem.org/Home/index_esp.aspx"><strong>Encuesta Nacional Sobre Salud y Envejecimiento</strong></a> (esta sección analiza los datos de la <strong>Edición 2018</strong>, puede acceder a las otras ediciones en la barra lateral izquierda). Esto se hace partiendo de la identificación de las preguntas de la encuesta que guarden la mayor similitud posible con las que contiene el cuestionario <a href="https://nutricionemocional.es/sites/default/files/tests_frailsarcf_web_2.pdf"><strong>SARC-F</strong></a>.
    </div>
    """,
    unsafe_allow_html=True
    )
    
    st.write("")


    st.markdown("""
    <div style="text-align: justify;"> 
    El proceso se realiza en las siguientes fases:
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    1. **Depuración de datos**: se eliminan datos de pacientes que no cumplan con los [**criterios de inclusión**](#criterios_de_inclusión) o presenten registros      incompletos. Además, se definen 5 cuestionamientos de la ENASEM que guardan similitud con los que conforman el test *SARC-F* y se crea una submuestra de         participantes que hayan contestado a estos cuestionamientos.
    """)

    st.markdown("""
    2. **Clasificación de participantes**: Usando la [**teoría de conjuntos rugosos**](https://shre.ink/DjYi), se divide la base de datos en una colección de     subconjuntos de pacientes que hayan contestado idénticamente a las preguntas clave (a estos subconjuntos se les llama relaciones de indiscernibilidad).
    """)

    st.markdown("""
    3. **Obtención de reglas de decisión**: Se entrena un modelo de árbol de decisión para determinar un conjunto de reglas que permitan clasificar a los     pacientes de la base de datos (aún aquellos que inicialmente no tenían respuestas completas en todas las preguntas de interés).
    """)



    
    st.subheader("Sección 1: Carga y depuración de datos")

    st.markdown(
    """
    <div style="text-align: justify;">
        Por favor, cargue un archivo correspondiente a las secciones <strong>conjunto_de_datos_sect_a_c_d_f_e_pc_h_i_enasem_2018</strong>. El archivo debe estar     en formato CSV y si se carga correctamente podrá visualizarse en el recuadro de abajo.
    </div>
    """,
    unsafe_allow_html=True
    )


    
    # Crear una caja de carga de archivos
    uploaded_file = st.file_uploader("**Elige un archivo CSV**", type="csv")

    if uploaded_file is not None:
        # Cargar el archivo como un DataFrame de pandas
        df = pd.read_csv(uploaded_file)

    #df.columns = df.columns.str.replace(r'(_18|_19)$', '', regex=True)

    # Mostrar el DataFrame cargado
    #df.columns = df.columns.str.replace('_18', '', regex=False)
    #df.columns = df.columns.str.replace('_18', '', regex=False).str.replace('_21', '', regex=False)
    # Aplicar un gradiente de color
    # Establecer el fondo de la tabla a blanco utilizando Styler
    #df_styled = df.style.applymap(lambda x: 'background-color: white; color: black')

    # Mostrar el dataframe estilizado en Streamlit
    st.dataframe(df, use_container_width=True)
    with st.expander("**Resumen de la base cargada**"):
        st.write(f'*La base seleccionada contiene **{df.shape[0]}** filas y **{df.shape[1]}** columnas.*')

    st.markdown(
    """
    <div style="text-align: justify;">
        Cada <strong>fila</strong> corresponde a las respuestas de un o una participante de la ENASEM, y cada <strong>columna</strong> corresponde a una pregunta en particular de las <strong>secciones de la "a" a la "i"</strong>. Si desea revisar el significado de las claves de las preguntas, consulte la sección de "Buscador de variables". Los registros vacíos (aquellos que muestren un <strong>None</strong>), los que contengan respuestas <strong>"8" o "9"</strong> (<strong>"No sabe"</strong> y <strong>"No quiere contestar"</strong>), y los que tengan <strong>(999)</strong> <strong>se eliminarán en la depuración</strong>.
    </div>
    """,
    unsafe_allow_html=True)


    


    st.write('<a id="criterios_de_inclusión"></a>', unsafe_allow_html=True)
    with st.expander("**Criterios de inclusión**"):
        st.markdown("""
    Los criterios de inclusión utilizados fueron:
    - Participantes de ambos sexos con 60 años o mas.
    - Participantes sin extremidades faltantes.
    - Participantes sin diagnóstico confirmado de cáncer u otros padecimientos o comorbilidades (a excepción de los que se describen en la sección de [variables de interés](#variables_de_interés)).
    """)

    st.write('<a id="variables_de_interés"></a>', unsafe_allow_html=True)
    st.subheader("Selección de variables de interés")
    
    st.markdown(
    """
        En esta sección puede elegir entre dos posibles listas de variables de interés:
        - La **selección estándar**: contiene una lista de variables que puede encontrarse en el apéndice (vea la parte final de esta página).
        
        - **Lista personalizada**: Si selecciona esta opción, aparecerá una barra en la que puede elegir entre las variables contenidas en la base de datos para su búsqueda. **Nota:** para que el código funcione correctamente, su lista debe incluir a las siguientes variables.""")

    

    # Ejemplo de contenido colapsable
    with st.expander("Listado de variables incluidas en la **selección estándar**"):
    
        st.write("""
                 
- **Código AGE_18**: Edad en años cumplidos.

- **Código SEX_18**: Sexo.

- **Código C4_18**: ¿Alguna vez le ha dicho un doctor o personal médico que usted tiene hipertensión o presión alta?

- **Código C6_18**: ¿Alguna vez le ha dicho un doctor o personal médico que usted tiene diabetes?

- **Código C12_18**: ¿Alguna vez le ha dicho un doctor o personal médico que usted tiene cáncer?

- **Código C19_18**: ¿Alguna vez le ha dicho un doctor o personal médico que usted tiene alguna enfermedad respiratoria, tal como asma o enfisema?

- **Código C22A_18**: ¿Alguna vez le ha dicho un doctor o personal médico que usted ha tenido un ataque/infarto al corazón?

- **Código C26_18**: ¿Alguna vez le ha dicho un doctor o personal médico que usted ha tenido una embolia cerebral, derrame cerebral o isquemia cerebral transitoria?

- **Código C32_18**: ¿Alguna vez le ha dicho un doctor o personal médico que usted tiene artritis o reumatismo?

- **Código C37_18**: ¿Se ha caído en los últimos dos años?

- **Código C49_1_18**: Estas preguntas se refieren a cómo se ha sentido usted durante la semana pasada. Para cada pregunta, por favor dígame, ¿la mayor parte del tiempo se ha sentido deprimido?

- **Código C49_2_18**: Estas preguntas se refieren a cómo se ha sentido usted durante la semana pasada. Para cada pregunta, por favor dígame, ¿la mayor parte del tiempo ha sentido que todo lo que hacía era un esfuerzo?

- **Código C49_8_18**: Estas preguntas se refieren a cómo se ha sentido usted durante la semana pasada. Para cada pregunta, por favor dígame, ¿la mayor parte del tiempo se ha sentido cansado?

- **Código C64_18**: ¿Comparado con hace dos años, usted...?

- **Código C66_18**: ¿Como cuántos kilos pesa usted ahora?

- **Código C67_1_18**: ¿Como cuánto mide usted sin zapatos? - Metros

- **Código C67_2_18**: ¿Como cuánto mide usted sin zapatos? - Centímetros

- **Código C68E_18**: Durante los últimos dos años, ¿ha tenido alguno de los siguientes problemas o molestias frecuentemente? - Fatiga severa o agotamiento serio

- **Código C68G_18**: Durante los últimos dos años, ¿ha tenido alguno de los siguientes problemas o molestias frecuentemente? - Pérdida involuntaria de orina, al hacer cosas como toser, estornudar, recoger cosas o hacer ejercicio

- **Código C68H_18**: Durante los últimos dos años, ¿ha tenido alguno de los siguientes problemas o molestias frecuentemente? - Pérdida involuntaria de orina, cuando tenía urgencia de orinar pero no pudo llegar al baño a tiempo

- **Código C69A_18**: ¿Cómo evaluaría la fuerza de su mano (la que utiliza más)?, ¿diría que es...?

- **Código C69B_18**: ¿Qué tan seguido tiene usted dificultad en mantener su equilibrio/balance?, ¿diría que...?

- **Código C71A_18**: ¿Le falta alguna extremidad o parte de sus piernas o brazos debido a un accidente o enfermedad?

- **Código C76_18**: En los últimos 12 meses, ¿cuánto efecto cree usted que el estrés ha tenido sobre su salud?

- **Código H1_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene usted dificultad en caminar varias cuadras?

- **Código H4_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en estar sentado(a) por dos horas?

- **Código H5_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en levantarse de una silla después de haber estado sentado(a) durante largo tiempo?

- **Código H6_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en subir varios pisos de escaleras sin descansar?

- **Código H8_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en inclinar su cuerpo, arrodillarse, agacharse o ponerse en cuclillas?

- **Código H9_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en subir o extender los brazos más arriba de los hombros?

- **Código H10_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud ¿tiene alguna dificultad en jalar o empujar objetos grandes como un sillón?

- **Código H11_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en levantar o transportar objetos que pesan más de 5 kilos, como una bolsa pesada de alimentos?

- **Código H12_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en recoger una moneda de 1 peso de la mesa?

- **Código H13_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene usted dificultad para vestirse, incluyendo ponerse los zapatos y los calcetines?

- **Código H15A_18**: Por favor dígame si tiene alguna dificultad con cada una de las actividades que le voy a mencionar. Si usted no hace ninguna de las siguientes actividades, simplemente dígamelo. No incluya dificultades que cree que durarán menos de tres meses. Debido a un problema de salud ¿usted tiene dificultad para caminar de un lado a otro de un cuarto?

- **Código H15B_18**: ¿Usa usted equipo o aparatos, tales como bastón, caminador o silla de ruedas para caminar de un lado a otro de un cuarto?

- **Código H15D_18**: ¿Alguien le ayuda a usted para caminar de un lado a otro de un cuarto?

- **Código H16A_18**: Por favor dígame si tiene alguna dificultad con cada una de las actividades que le voy a mencionar. Si usted no hace ninguna de las siguientes actividades, simplemente dígamelo. No incluya dificultades que cree que durarán menos de tres meses. Debido a un problema de salud ¿usted tiene dificultad para bañarse en una tina o regadera?

- **Código H16D_18**: ¿Alguien le ayuda a usted para bañarse en una tina o regadera?

- **Código H17A_18**: Por favor dígame si tiene alguna dificultad con cada una de las actividades que le voy a mencionar. Si usted no hace ninguna de las siguientes actividades, simplemente dígamelo. No incluya dificultades que cree que durarán menos de tres meses. Debido a un problema de salud ¿usted tiene dificultad al comer, por ejemplo para cortar su comida?

- **Código H17D_18**: ¿Alguien le ayuda a usted al comer, por ejemplo para cortar su comida?

- **Código H18A_18**: Por favor dígame si tiene alguna dificultad con cada una de las actividades que le voy a mencionar. Si usted no hace ninguna de las siguientes actividades, simplemente dígamelo. No incluya dificultades que cree que durarán menos de tres meses. Debido a un problema de salud ¿usted tiene dificultad al acostarse y levantarse de la cama?

- **Código H18D_18**: ¿Alguien le ayuda a usted al acostarse y levantarse de la cama?

- **Código H19A_18**: Por favor dígame si tiene alguna dificultad con cada una de las actividades que le voy a mencionar. Si usted no hace ninguna de las siguientes actividades, simplemente dígamelo. No incluya dificultades que cree que durarán menos de tres meses. Debido a un problema de salud ¿usted tiene dificultad al usar el excusado, incluyendo subirse y bajarse o ponerse en cuclillas?

- **Código H19D_18**: ¿Alguien le ayuda a usted al usar el excusado, incluyendo subirse y bajarse o ponerse en cuclillas?
    """)


    # Opcional: mostrar estadísticas básicas del DataFrame
    #st.write(f'**Descripción de la base de datos:** La base seleccionada contiene **{df.shape[0]}** filas y **{df.shape[1]}** columnas.')
    df['Indice'] = df.index

    # Lista predefinida "selección estándar"
    seleccion_estandar = ['Indice',"AGE_18", 'SEX_18', 'C4_18', 'C6_18', 'C12_18', 'C19_18', 'C22A_18', 'C26_18', "C32_18", 'C37_18',
                      "C49_1_18", 'C49_2_18', 'C49_8_18', 'C64_18', 'C66_18', 'C67_1_18', 'C67_2_18', 'C68E_18', 'C68G_18',
                      'C68H_18', 'C69A_18', 'C69B_18', 'C71A_18', 'C76_18', 'H1_18', 'H4_18', 'H5_18', 'H6_18', 'H8_18',
                      'H9_18', 'H10_18', 'H11_18', 'H12_18', 'H13_18', 'H15A_18', 'H15B_18', 'H15D_18', 'H16A_18', 'H16D_18',
                      'H17A_18', 'H17D_18', 'H18A_18', 'H18D_18', 'H19A_18', 'H19D_18']
###

# Opción para seleccionar columnas
    opcion_seleccion = st.radio("¿Cómo quieres seleccionar las columnas?", ("Usar selección estándar", "Usar lista personalizada"))

    if opcion_seleccion == "Usar selección estándar":
        selected_columns = seleccion_estandar
    else:
        selected_columns = st.multiselect("Usar lista personalizada", df.columns.tolist())

    # Lista de verificación para seleccionar columnas
    #selected_columns = st.multiselect("Selecciona las columnas para mostrar", df.columns.tolist())
        
    if selected_columns:
        # Crear dataframe reducido
        df = df[selected_columns]
            
        st.write("Base de datos con la **lista de variables seleccionada:**")
        st.write(df)
            
        # Mostrar información del dataframe reducido
        num_rows, num_cols = df.shape
        #st.write(f"Número de filas: {num_rows}")
        #st.write(f"Número de columnas: {num_cols}")
        # Opcional: mostrar estadísticas básicas del DataFrame
        st.write(f'**Descripción de la base de datos:** La base seleccionada contiene **{df.shape[0]}** filas y **{df.shape[1]}** columnas.')


        # Contar valores NaN por columna
        nan_counts = df.isna().sum().reset_index()
        nan_counts.columns = ["Clave", "Conteo de filas vacias"]
            
        #st.write("Conteo de valores NaN por columna:")
        #st.write(nan_counts)
        #st.dataframe(nan_counts, use_container_width=True)

####################################

        import pandas as pd
        import gdown
        import streamlit as st

        # Función para cargar el diccionario
        def cargar_diccionario(url, nombre):
            output = f'{nombre}.xlsx'
            gdown.download(url, output, quiet=False)

            try:
                # Intentar leer el archivo Excel
                xls = pd.ExcelFile(output)
                sheet_name = xls.sheet_names[0]  # Obtener el nombre de la primera hoja
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None, engine='openpyxl')
                #st.write("Archivo xlsx cargado correctamente.")
            except Exception as e:
                st.error(f"Error al leer el archivo xlsx: {e}")
                return {}

            diccionario = {}
            for index, row in df.iterrows():
                variable = row[0]
                if isinstance(variable, str) and variable.startswith("Pregunta"):
                    partes = variable.split(" ", 2)
                    if len(partes) < 3:
                        continue
                    codigo = partes[1].replace('.', '_')
                    explicacion = partes[2]
                    diccionario[codigo] = explicacion

            # Agregar explicaciones adicionales
            diccionario.update({
                "Indice": "El número de fila en la base de datos",
                "AGE_18": "Edad en años cumplidos",
                "SEX_18": "Sexo"
            })

            return diccionario

        # URL del archivo de 2018 en Google Drive
        url_2018 = 'https://drive.google.com/uc?id=17o5hDLk_RHU6kGKinJtmRGtApv927Abu'

        # Interfaz en Streamlit
        st.write("Estas son las variables que seleccionó (la primera columna corresponde a la **Clave**, la segunda es el **conteo de filas vacías** de cada variable, la tercera es la **explicación** de la clave para esa variable).")

        # Inicializar el estado de la sesión para el historial de búsquedas
        if 'historico_busquedas' not in st.session_state:
            st.session_state.historico_busquedas = pd.DataFrame(columns=['Año', 'Código', 'Explicación'])

        # Cargar el diccionario de 2018
        diccionario_2018 = cargar_diccionario(url_2018, 'diccionario_2018')

        df_c=df.copy()
        # Obtener la lista de nombres de columnas
        columnas = df_c.columns.tolist()

        # Contar valores NaN por columna
        nan_counts = df_c.isna().sum().reset_index()
        nan_counts.columns = ["Clave", "Conteo"]

        # Agregar explicaciones al DataFrame nan_counts
        nan_counts['Explicación'] = nan_counts['Clave'].map(diccionario_2018)

        # Mostrar el DataFrame con el conteo de valores NaN y explicaciones
        #st.write("Conteo de valores NaN por columna con explicaciones:")
        st.dataframe(nan_counts, use_container_width=True)




######################################

        # Filtro inicial
        df = df[df['AGE_18'] < 100]

        # Lista de columnas a modificar
        columnas_modificar = ['C4_18', 'C6_18', 'C12_18', 'C19_18', 'C22A_18', 'C26_18', "C32_18", 'C37_18',
                      "C49_1_18", 'C49_2_18', 'C49_8_18', 'C64_18', 'C66_18', 'C68E_18', 'C68G_18',
                      'C68H_18', 'C69A_18', 'C69B_18', 'C71A_18', 'C76_18', 'H1_18', 'H4_18', 'H5_18', 'H6_18', 'H8_18',
                      'H9_18', 'H10_18', 'H11_18', 'H12_18', 'H13_18', 'H15A_18', 'H15B_18', 'H15D_18', 'H16A_18', 'H16D_18',
                      'H17A_18', 'H17D_18', 'H18A_18', 'H18D_18', 'H19A_18', 'H19D_18']

        # Convertir valores 6.0 o 7.0 en 1.0 en las columnas especificadas
        df[columnas_modificar] = df[columnas_modificar].replace({6.0: 1.0, 7.0: 1.0})

        # Combinar los campos de las columnas de estatura en una sola columna de estatura en metros
        df['C67_18'] = df['C67_1_18'] + df['C67_2_18'] / 100
        df = df.drop(columns=['C67_1_18', 'C67_2_18'])

        # Eliminar filas que contengan valores 8.0 o 9.0 en cualquiera de las columnas especificadas
        df = df[~df[columnas_modificar].isin([8.0, 9.0, 999, 9.99]).any(axis=1)]
        df = df[~df['C67_18'].isin([9.99, 8.88])]
        columnas_seleccionadas = list(df.columns)

        # Crear widgets de selección para cada columna seleccionada en la barra lateral
        filtros = {}
        for col in columnas_seleccionadas:
            if df[col].dtype == 'object':
                valores_unicos = df[col].unique().tolist()
                seleccion = st.sidebar.multiselect(f'Seleccionar valores para {col}', valores_unicos)
                if seleccion:
                    filtros[col] = seleccion
            else:
                rango = st.sidebar.slider(f'Seleccionar rango para {col}', min_value=float(df[col].min()), max_value=float(df[col].max()), value=(float(df[col].min()), float(df[col].max())), step=1.0)
                if rango:
                    filtros[col] = rango

        # Filtrar el DataFrame basado en los valores seleccionados
        df_filtrado = df.copy()
        for col, condicion in filtros.items():
            if isinstance(condicion, list):
                df_filtrado = df_filtrado[df_filtrado[col].isin(condicion)]
            else:
                df_filtrado = df_filtrado[(df_filtrado[col] >= condicion[0]) & (df_filtrado[col] <= condicion[1])]

        #st.write('DataFrame Filtrado')
        st.markdown("Aquí puede ver la **base de datos depurada:**")
        st.dataframe(df_filtrado, use_container_width=True)
        #st.write("Las dimensiones de la base de datos son:")
        #st.write(df_filtrado.shape)
        st.write(f'La base depurada contiene **{df_filtrado.shape[0]}** filas y **{df_filtrado.shape[1]}** columnas.')
        datos_filtrados = df_filtrado.copy()

#######################################


     # Definir condiciones para cada grupo
        conditions = {
            "Ninguna": {
                'C4_18': 2.0,
                'C6_18': 2.0,
                'C12_18': 2.0,
                'C19_18': 2.0,
                'C22A_18': 2.0,
                'C26_18': 2.0,
                'C32_18': 2.0
            },
            "Diabetes": {
                'C4_18': 1.0,
                'C6_18': 2.0,
                'C12_18': 2.0,
                'C19_18': 2.0,
                'C22A_18': 2.0,
                'C26_18': 2.0,
                'C32_18': 2.0
            },
            "Hipertensión": {
                'C4_18': 2.0,
                'C6_18': 1.0,
                'C12_18': 2.0,
                'C19_18': 2.0,
                'C22A_18': 2.0,
                'C26_18': 2.0,
                'C32_18': 2.0
            },
            "Hipertensión y Diabetes": {
                'C4_18': 1.0,
                'C6_18': 1.0,
                'C12_18': 2.0,
                'C19_18': 2.0,
                'C22A_18': 2.0,
                'C26_18': 2.0,
                'C32_18': 2.0
            }
        }


        st.markdown(
            """
            Aquí pude seleccionar entre hacer un análisis utilizando datos de la muestra completa o aislar un grupo de interés. Los grupos disponibles son:

            - **Ninguna:** Declara **no tener diagnóstico de Diabetes o Hipertensión**.

            - **Diabetes:** Declara tener diagnóstico **negativo en Hipertensión** pero **positivo en Diabetes**.

            - **Hipertensión:** Declara tener diagnóstico **positivo en Hipertensión**  pero **negativo en Diabetes**.

            - **Hipertensión y Diabetes:** Declara tener diagnóstico **positivo tanto en Hipertensión como en Diabetes**.
            
            """
            )

        # Crear una selección en Streamlit para elegir entre los conjuntos
        seleccion = st.selectbox("**Seleccione un grupo**", list(conditions.keys()))

        # Crear una selección múltiple en Streamlit para el valor de SEX_18
        sex_values = df_filtrado['SEX_18'].unique()
        st.write("""Aquí pude seleccionar entre hacer un análisis utilizando datos de la muestra completa o sobre un solo sexo. La Clave numérica **"1.0"** corresponde a las **mujeres** y la **Clave "2.0"** corresponde a los **hombres.**
        """)
        sex_selection = st.multiselect("**Seleccione el sexo de la muestra** (puede seleccionar ambos si quiere analizar la muestra completa)", sex_values, default=sex_values)

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
        nuevo_dataframe_filtrado['Comorbilidad'] = seleccion


        st.write("**La base de datos a analizar se muestra aquí:**")
        # Mostrar el DataFrame filtrado
        #st.dataframe(nuevo_dataframe_filtrado, use_container_width=True)
        datos_limpios = nuevo_dataframe_filtrado.copy()
        datos_limpios = datos_limpios.dropna()
        st.dataframe(datos_limpios, use_container_width=True)
        st.write(f'La base depurada contiene **{datos_limpios.shape[0]}** filas y **{datos_limpios.shape[1]}** columnas.')
        #datos_limpios.shape

        # Botón para descargar el dataframe reducido en formato csv
        csv_data = convert_df_to_csv(datos_limpios)
        st.download_button(
            label="Descargar Dataframe en formato CSV",
            data=csv_data,
            file_name="Base depurada 2018.csv",
            mime="text/csv"
            )

        xlsx_data = convert_df_to_xlsx(datos_limpios)
        st.download_button(
            label="Descargar Dataframe en formato XLSX",
            data=xlsx_data,
            file_name="Base depurada 2018.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

##########################################
    st.subheader("Sección 2: Clasificación de participantes")
    st.markdown(
        """
        En esta sección se utiliza la [**teoría de conjuntos rugosos**](https://shre.ink/DjYi) para agrupar a las y los participantes de la encuesta cuyas respuestas fueron idénticas. Esto se logra mediante el cálculo de las [**relaciones de indiscerbibilidad**](https://en.wikipedia.org/wiki/Rough_set). El gáfico de representa a las particiones de pacientes con respuestas idénticas en las preguntas (**'C37_18', 'H11_18', 'H15A_18', 'H5_18', 'H6_18'**).


"""
)
    with st.expander("**Preguntas similares a las de SARC-F**"):    
        st.write("""
                 
- **Código C37_18**: ¿Se ha caído en los últimos dos años?

- **Código H5_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en levantarse de una silla después de haber estado sentado(a) durante largo tiempo?

- **Código H6_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en subir varios pisos de escaleras sin descansar?

- **Código H11_18**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en levantar o transportar objetos que pesan más de 5 kilos, como una bolsa pesada de alimentos?

- **Código H15A_18**: Por favor dígame si tiene alguna dificultad con cada una de las actividades que le voy a mencionar. Si usted no hace ninguna de las siguientes actividades, simplemente dígamelo. No incluya dificultades que cree que durarán menos de tres meses. Debido a un problema de salud ¿usted tiene dificultad para caminar de un lado a otro de un cuarto?

    """)


    
    ind=indiscernibility(['C37_18', 'H11_18', 'H15A_18', 'H5_18', 'H6_18'], datos_limpios)
    

    import matplotlib.pyplot as plt


    # Calcular las longitudes de los conjuntos con longitud >= 2
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(ind) if len(conjunto) >= 2]

    # Ordenar la lista de tuplas por la longitud de los conjuntos en orden descendente
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Crear un diccionario para mapear el índice del conjunto al nuevo nombre
    nombres_conjuntos_nuevos = {num_conjunto: f"Conjunto {i}" for i, (num_conjunto, _) in enumerate(longitudes_conjuntos_ordenadas)}

    #st.write("Longitudes de los conjuntos con su respectivo nuevo nombre (en orden descendente):")
    #for num_conjunto, longitud in longitudes_conjuntos_ordenadas:
    #    nombre_conjunto_nuevo = nombres_conjuntos_nuevos[num_conjunto]
    #    st.write(f"{nombre_conjunto_nuevo}: {longitud}")

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
    # Agregar el tamaño de la muestra total como texto
    ax.annotate(f'Tamaño de la muestra total a partir de los datos de 2018: {total_elementos}', 
            xy=(0.5, -0.05), 
            xycoords='axes fraction', 
            ha='center', 
            fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Distribución de subconjuntos')

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)
    st.write("Seleccione en el recuadro el **número mínimo de miembros** de los conjuntos que considerará. Esto simplifica el análisis. **Típicamente se define un número mínimo de 30 miembros** (o de 10 miembros si la muestra total es demasiado pequeña).")


    # Entrada del usuario para el tamaño mínimo del conjunto
    tamaño_mínimo = st.number_input("**Defina el número mínimo de miebros que tendrán los conjuntos a considerar:**", min_value=1, value=2, step=1)

    # Calcular las longitudes de los conjuntos con longitud >= tamaño_mínimo
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(ind) if len(conjunto) >= tamaño_mínimo]

    # Ordenar la lista de tuplas por la longitud de los conjuntos en orden descendente
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Crear un diccionario para mapear el índice del conjunto al nuevo nombre
    nombres_conjuntos_nuevos = {num_conjunto: f"**Conjunto {i}**" for i, (num_conjunto, _) in enumerate(longitudes_conjuntos_ordenadas)}


##
    st.write("Estos son los conjuntos considerados y su número de miembros:")
#    for num_conjunto, longitud in longitudes_conjuntos_ordenadas:
#        nombre_conjunto_nuevo = nombres_conjuntos_nuevos[num_conjunto]
#        st.write(f"{nombre_conjunto_nuevo}: {longitud}")
##
#st.write("Longitudes de los conjuntos con su respectivo nuevo nombre (en orden descendente):")

    #  Crear una fila con columnas
    cols = st.columns(3)

    # Rellenar cada columna con datos
    for i, (num_conjunto, longitud) in enumerate(longitudes_conjuntos_ordenadas):
        nombre_conjunto_nuevo = nombres_conjuntos_nuevos[num_conjunto]
        cols[i % 3].write(f"{nombre_conjunto_nuevo}: {longitud}")


    # Calcular el total de elementos en todos los conjuntos
    total_elementos = sum(longitud for _, longitud in longitudes_conjuntos_ordenadas)

    # Calcular los porcentajes para cada conjunto
    porcentajes = [(nombres_conjuntos_nuevos[num_conjunto], longitud / total_elementos * 100) for num_conjunto, longitud in longitudes_conjuntos_ordenadas]

    # Extraer los nombres de los conjuntos y los porcentajes para el diagrama de pastel
    nombres_conjuntos = [nombre for nombre, _ in porcentajes]
    porcentajes_valores = [valor for _, valor in porcentajes]

    # Crear el diagrama de pastel
    #fig, ax = plt.subplots(figsize=(10, 8))
    #_, _, autopcts = ax.pie(porcentajes_valores, labels=nombres_conjuntos, autopct='%1.1f%%', startangle=140, textprops={'visible': False})
    #for autopct in autopcts:
    #    autopct.set_visible(True)  # Mostrar los porcentajes
    #ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    #ax.set_title('Distribución de subconjuntos')

    # Mostrar el gráfico en Streamlit
    #st.pyplot(fig)

    import numpy as np
    
    # Calcular las longitudes de los conjuntos con longitud >= tamaño_mínimo
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(ind) if len(conjunto) >= tamaño_mínimo]

    # Ordenar la lista de tuplas por la longitud de los conjuntos en orden descendente
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)
    
    num_conjuntos = st.number_input("Define el número de conjuntos para vizualizar los perfiles que los caracterizan:", min_value=1, max_value=len(ind), value=15)

    # Obtener los 15 conjuntos más numerosos
    conjuntos_mas_numerosos = [conjunto for i, conjunto in enumerate(ind) if i in [num_conjunto for num_conjunto, _ in longitudes_conjuntos_ordenadas[:num_conjuntos]]]

    # Crear DataFrames para cada uno de los 15 conjuntos más numerosos
    for i, conjunto in enumerate(conjuntos_mas_numerosos, 0):
        indices_seleccionados = list(conjunto)
        df_conjunto = datos_limpios[datos_limpios.index.isin(indices_seleccionados)]
        globals()[f"df_Conjunto_{i}"] = df_conjunto

    # Definir las columnas de interés
    columnas_interes_radar = ['C37_18', 'H11_18', 'H15A_18', 'H5_18', 'H6_18']

    # Definir los nombres de los dataframes
    nombres_dataframes = [f"df_Conjunto_{i}" for i in range(0, num_conjuntos)]

    # Definir los valores para cada dataframe en las columnas de interés
    valores_dataframes = []
    for nombre_df in nombres_dataframes:
        df = eval(nombre_df)
        valores = df[columnas_interes_radar].iloc[0].tolist()  # Tomar solo la primera fila
        valores_dataframes.append(valores)

    st.write("A continuación se muestran los perfiles de las relaciones de indiscernibilidad, en términos de los cinco ítems de la encuesta que guardan mayor similitud con los del test SARC-F. Los **gráficos de radar** muestran las respuestas a estos ítems:")
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
    filas_seleccionadas = filas_seleccionadas[['C37_18', 'H11_18', 'H15A_18', 'H5_18', 'H6_18', 'num_conjunto', 'Comorbilidad']]

    # Crear un nuevo DataFrame con las filas seleccionadas
    nuevo_dataframe = pd.DataFrame(filas_seleccionadas)

    # Mostrar las primeras filas del nuevo DataFrame
    #st.dataframe(nuevo_dataframe, use_container_width=True)



#######################

#####################333

#    import pandas as pd

#    # Definir las condiciones para asignar los valores a la nueva columna
#    def asignar_riesgo(num_conjunto):
#        if num_conjunto in [6, 14]:
#            return "Riesgo considerable"
#        elif num_conjunto in [3, 5, 10]:
#            return "Riesgo moderado"
#        elif num_conjunto in [1, 2, 4, 7, 8, 9, 11, 12, 13]:
#            return "Riesgo leve"
#        elif num_conjunto == 0:
#            return "Sin Riesgo"
#        else:
#            return "No clasificado"  # Manejar cualquier otro caso
    with st.expander("**Determinación de un nivel de riesgo**"):
        st.markdown("""
                    Ya que las preguntas de la ENASEM no son idénticas a las del cuestionario SARC-F (las respuestas de ese cuestionario permiten establecer una escala de intensidad de dificultad para realizar ciertas actividades, mientras que la ENASEM solo permiten contestar *si* o *no*), se definió un criterio alternativo mediante el cual pudiera establecerse un nivel de riesgo de padecer sarcopenia. Los niveles de riesgo definidos son:

                    - **Sin riesgo:** No se manifiesta tener dificultad en ninguno de los 5 cuestionamientos (o caídas recientes, en el caso de esa pregunta).

                    - **Riesgo leve:** Se manifiesta tener dificultades en uno o dos de los cuestionamientos (o dificultad en uno y caídas recientes, en el caso de esa pregunta).

                    - **Riesgo moderado:** Se manifiesta tener dificultades simultaneas en tres de los cuestionamientos (o dos cuestinamientos y caidas recientes).

                    - **Riesgo severo:** Se manifiesta tener dificultades en cuatro o cinco de de los cuestionamientos (o en cuatro de ellos y caidas recientes).
                    """)

    
#    # Función para determinar el nivel de riesgo
#    def asignar_nivel_riesgo(row):
#        valores = row[['C37_18', 'H11_18', 'H15A_18', 'H5_18', 'H6_18']]
#        if all(valores == 2.0):
#            return "Sin Riesgo"
#        cuenta_1_0 = (valores == 1.0).sum()
#        if cuenta_1_0 == 1 or cuenta_1_0 == 2:
#            return "Riesgo leve"
#        elif cuenta_1_0 == 3:
#            return "Riesgo moderado"
#        elif cuenta_1_0 >= 4:
#            return "Riesgo severo"
#    # Función para determinar el nivel de riesgo

    def asignar_nivel_riesgo(row):
        valores = row[['C37_18', 'H11_18', 'H15A_18', 'H5_18', 'H6_18']]
    
    # Contar cuántas columnas individualmente tienen el valor 1.0
        cuenta_1_0 = (valores == 1.0).sum()
    
    # Verificar las condiciones para cada nivel de riesgo
        if all(valores == 2.0):
            return "Sin Riesgo"
        elif cuenta_1_0 == 1 or cuenta_1_0 == 2:
            return "Riesgo leve"
        elif cuenta_1_0 == 3:
            return "Riesgo moderado"
        elif cuenta_1_0 == 4 or cuenta_1_0 == 5:
            return "Riesgo severo"
        else:
            return "Nivel de riesgo no determinado"  # En caso de que ninguna condición coincida, lo cual no debería ocurrir.




    # Aplicar la función a cada fila del DataFrame
    nuevo_dataframe['nivel_riesgo'] = nuevo_dataframe.apply(asignar_nivel_riesgo, axis=1)

    # Mostrar el DataFrame con el nivel de riesgo asignado
    #st.write(nuevo_dataframe)


    # Agregar la nueva columna al DataFrame
    #nuevo_dataframe['nivel_riesgo'] = nuevo_dataframe['num_conjunto'].apply(asignar_riesgo)
    
    st.write("Debajo puede ver y descargar la base de datos de las respuestas que se usan para estimar el nivel de riesgo de sarcopenia, la etiqueta que indica al conjunto que corresponde cada paciente y el nivel de riesgo asociado.")
    
    st.dataframe(nuevo_dataframe, use_container_width=True)

    # Botón para descargar el dataframe reducido en formato csv
    csv_data = convert_df_to_csv(nuevo_dataframe)
    st.download_button(
        label="Descargar Dataframe en formato CSV",
        data=csv_data,
        file_name="Base de relaciones de indiscernibilidad y nivel de riesgo_2018.csv",
        mime="text/csv"
        )

    xlsx_data = convert_df_to_xlsx(nuevo_dataframe)
    st.download_button(
        label="Descargar Dataframe en formato XLSX",
        data=xlsx_data,
        file_name="Base de relaciones de indiscernibilidad y nivel de riesgo_2018.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )




#######################


#    # Función para calcular la diferencia entre tamaños de conjuntos
#    def calcular_diferencia(lista1, lista2):
#        diferencia = sum(abs(len(conj1) - len(conj2)) for conj1, conj2 in zip(lista1, lista2))
#        return diferencia

    # Definir las columnas de interés
#    columnas_interes = ['H15A_18', 'H11_18', 'H5_18', 'H6_18', 'C37_18']

    # Generar listas de conjuntos
#    lista_1 = indiscernibility(columnas_interes, nuevo_dataframe)
#    lista_2_original = indiscernibility(columnas_interes, nuevo_dataframe)

    # Obtener lista de tamaños de cada conjunto
#    tamaños_lista_1 = [len(conjunto) for conjunto in lista_1]
#    tamaños_lista_2_original = [len(conjunto) for conjunto in lista_2_original]

    # Inicializar variables para seguimiento de la lista más parecida
#    mejor_lista = lista_2_original
#    mejor_diferencia = calcular_diferencia(lista_1, lista_2_original)

    # Eliminar una por una cada columna de lista_2 y mostrar los tamaños resultantes
#    for columna1 in columnas_interes:
#        columnas_sin_columna1 = columnas_interes.copy()
#        columnas_sin_columna1.remove(columna1)
#        lista_2_sin_columna1 = indiscernibility(columnas_sin_columna1, nuevo_dataframe)
#        diferencia = calcular_diferencia(lista_1, lista_2_sin_columna1)
    
#        if diferencia < mejor_diferencia:
#            mejor_lista = lista_2_sin_columna1
#            mejor_diferencia = diferencia
    
#        # Eliminar pares de columnas de lista_2 y mostrar los tamaños resultantes
#        for columna2 in columnas_sin_columna1:
#            if columna2 != columna1:
#                columnas_sin_par = columnas_sin_columna1.copy()
#                columnas_sin_par.remove(columna2)
#                lista_2_sin_par = indiscernibility(columnas_sin_par, nuevo_dataframe)
#                diferencia = calcular_diferencia(lista_1, lista_2_sin_par)
            
#                if diferencia < mejor_diferencia:
#                    mejor_lista = lista_2_sin_par
#                    mejor_diferencia = diferencia

    # Mostrar la mejor lista encontrada en Streamlit
#    st.write("Tamaños de conjuntos en lista_1:", tamaños_lista_1)
#    st.write("Tamaños de conjuntos en la mejor lista:", [len(conjunto) for conjunto in mejor_lista])

    # Visualización con un gráfico de barras
    #fig, ax = plt.subplots()
    #labels = [f"Conjunto {i}" for i in range(len(tamaños_lista_1))]
    #x = range(len(tamaños_lista_1))
    #ax.bar(x, tamaños_lista_1, width=0.4, label='lista_1', align='center')
    #ax.bar(x, [len(conjunto) for conjunto in mejor_lista], width=0.4, label='Mejor Lista', align='edge')
    #ax.set_xlabel('Conjuntos')
    #ax.set_ylabel('Tamaños')
    #ax.set_title('Comparación de tamaños de conjuntos')
    #ax.legend()

   # st.pyplot(fig)


    st.subheader("Sección 3: Identificación de un reducto")

    st.markdown("""
                Un **reducto** corresponde a una lista reducidad de preguntas que puede crear la misma clasificación de pacientes que la lista completa. En esta sección se identifica un [reducto](https://www.researchgate.net/publication/262252197_The_Concept_of_Reducts_in_Pawlak_Three-Step_Rough_Set_Analysis) para la lista de preguntas 'C37_18', 'H11_18', 'H15A_18', 'H5_18', 'H6_18'. Se realiza una clasificación en la que de forma progresiva se van quitando preguntas de la lista original y se compara la partición que crea con la que logra la lista completa. El reducto corresponde a la lista de preguntas que genere una partición lo mas parecida posible a la de la lista original.
""")


    # Calcular la diferencia absoluta total de tamaños
    def diferencia_total(lista_a, lista_b):
        min_len = min(len(lista_a), len(lista_b))
        return sum(abs(lista_a[i] - lista_b[i]) for i in range(min_len))

    # Ejemplo de listas de conjuntos
    lista_1 = indiscernibility(['H15A_18', 'H11_18', 'H5_18', 'H6_18', 'C37_18'], nuevo_dataframe)
    lista_2 = indiscernibility(['H15A_18', 'H11_18', 'H5_18', 'H6_18', 'C37_18'], nuevo_dataframe)

    # Mostrar tamaño de los conjuntos originales
#    st.write("Tamaño de lista_1:", len(lista_1))
#    st.write("Tamaño de lista_2:", len(lista_2))

    # Obtener lista de tamaños de cada conjunto
    tamaños_lista_1 = [len(conjunto) for conjunto in lista_1]
    tamaños_lista_2 = [len(conjunto) for conjunto in lista_2]

#    st.write("Tamaños de conjuntos en lista_1:", tamaños_lista_1)
#    st.write("Tamaños de conjuntos en lista_2:", tamaños_lista_2)

    # Inicializar variables para el seguimiento de la mejor coincidencia
    mejor_similitud = float('inf')
    mejor_lista = None

    # Expander para mostrar tamaños resultantes al eliminar columnas
    with st.expander("**Búsqueda del reducto**"):
        st.write("La partición creada por la lista completa se nombró como *lista_1*. La *lista_2* corresponde a copias de *lista_1* en la que se van quitando progresivamente ciertas preguntas (primero una sola, luego 2, luego 3 etc.). El **reducto** corresponde a la lista de preguntas que genere una partición igual, o lo más parecida posible a la de *lista_1*." )
        #st.write(f'La base depurada contiene **{datos_limpios.shape[0]}** filas y **{datos_limpios.shape[1]}** columnas.')
        st.write(f'Las particiones creadas usando las cinco preguntas de interés son: {len(lista_1)}')
        longitudes = [len(conjunto) for conjunto in lista_1]
        st.text("Longitudes de los conjuntos: " + ', '.join(map(str, longitudes)))

        for columna1 in ['H15A_18', 'H11_18', 'H5_18', 'H6_18', 'C37_18']:
            columnas_sin_columna1 = ['H15A_18', 'H11_18', 'H5_18', 'H6_18', 'C37_18']
            columnas_sin_columna1.remove(columna1)
            lista_2_sin_columna1 = indiscernibility(columnas_sin_columna1, nuevo_dataframe)
            tamaños_lista_2_sin_columna1 = [len(conjunto) for conjunto in lista_2_sin_columna1]
            st.write(f"**Tamaño de lista_2 sin** {columna1}: {len(lista_2_sin_columna1)}")
            st.text("Longitudes de los conjuntos: " + ', '.join(map(str, tamaños_lista_2_sin_columna1)))
            #st.write("Tamaños de conjuntos en lista_2:", tamaños_lista_2_sin_columna1)

            # Comparar similitud
            similitud = diferencia_total(tamaños_lista_1, tamaños_lista_2_sin_columna1)
            if similitud < mejor_similitud:
                mejor_similitud = similitud
                mejor_lista = columnas_sin_columna1.copy()

            # Eliminar pares de columnas de lista_2 y mostrar los tamaños resultantes
            for columna2 in columnas_sin_columna1:
                if columna2 != columna1:
                    columnas_sin_par = columnas_sin_columna1.copy()
                    columnas_sin_par.remove(columna2)
                    lista_2_sin_par = indiscernibility(columnas_sin_par, nuevo_dataframe)
                    tamaños_lista_2_sin_par = [len(conjunto) for conjunto in lista_2_sin_par]
                    st.write(f"**Tamaño de lista_2 sin** {columna1} **y** {columna2}: {len(lista_2_sin_par)}")
                    st.text("Longitudes de los conjuntos: " + ', '.join(map(str, tamaños_lista_2_sin_par)))
                    #st.write("Tamaños de conjuntos en lista_2:", tamaños_lista_2_sin_par)

                    # Comparar similitud
                    similitud = diferencia_total(tamaños_lista_1, tamaños_lista_2_sin_par)
                    if similitud < mejor_similitud:
                        mejor_similitud = similitud
                        mejor_lista = columnas_sin_par.copy()

    # Mostrar la lista de columnas que guarda mayor similitud con lista_1
    #st.write("El reducto es:")
    #st.text("**El reducto es: **" + ', '.join(map(str, mejor_lista)))
    st.markdown("**El reducto es: **" + ', '.join(map(str, mejor_lista)))

    
    # Obtener los valores únicos de la columna 'nivel de riesgo'
    nivel_riesgo = nuevo_dataframe['nivel_riesgo'].unique()


#    # Crear una barra de selección múltiple para elegir niveles de riesgo
#    niveles_seleccionados = st.multiselect(
#        'En este recuadro puede seleccionar un subgrupo de pacientes que comparten el mismo nivel de riesgo',
#        nivel_riesgo
#    )

#    # Filtrar el DataFrame según los niveles de riesgo seleccionados
#    if niveles_seleccionados:
#        df_filtrado = nuevo_dataframe[nuevo_dataframe['nivel_riesgo'].isin(niveles_seleccionados)]
#        st.write(f"Filas con nivel de riesgo en {niveles_seleccionados}:")
#        st.dataframe(df_filtrado, use_container_width=True)
#    else:
#        st.write("Selecciona al menos un nivel de riesgo para visualizar las filas correspondientes.")
########################


    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    # Generar listas de conjuntos (simulando lista_2)
    lista_2 = indiscernibility(['C37_18', 'H11_18', 'H5_18', 'H6_18'], datos_limpios)
    #st.write(len(lista_2))
    # Obtener longitudes de conjuntos
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(lista_2) if len(conjunto) >= 2]

    # Ordenar por longitud
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Obtener los 15 conjuntos más numerosos
    conjuntos_mas_numerosos = [conjunto for i, conjunto in enumerate(lista_2) if i in [num_conjunto for num_conjunto, _ in longitudes_conjuntos_ordenadas[: 15]]]

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
    #st.title("Visualización de Conjuntos Más Numerosos")
    st.write("Perfiles de los pacientes construidos usando el reducto")

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
    #st.title("Filtrado de DataFrame por tamaño de conjuntos")

    # Calcular el tamaño de cada conjunto y filtrar los conjuntos con menos de 30 miembros
    conjuntos_mayores_30 = [conjunto for conjunto in ind if len(conjunto) >= 30]

    # Obtener los índices de las filas que pertenecen a los conjuntos mayores o iguales a 30 miembros
    indices_filtrados = [indice for conjunto in conjuntos_mayores_30 for indice in conjunto]

    # Filtrar el DataFrame 'datos_limpios' para mantener solo las filas con índices en 'indices_filtrados'
    datos_limpios_filtrados = datos_limpios.loc[indices_filtrados]

    # Mostrar el DataFrame filtrado en Streamlit
    #st.write("DataFrame filtrado con conjuntos mayores o iguales a 30 miembros:")
    #st.dataframe(datos_limpios_filtrados)

    # Mostrar el número total de conjuntos filtrados
    #st.write(f"Número de conjuntos mayores o iguales a 30 miembros: {len(conjuntos_mayores_30)}")

    # Mostrar el tamaño de cada conjunto filtrado
    #st.write("Tamaño de cada conjunto filtrado:")
    tamaños_conjuntos = [len(conjunto) for conjunto in conjuntos_mayores_30]
    #st.write(tamaños_conjuntos)

    with st.expander("Aproximaciones"):
        st.write('El reducto puede generar particiones que no coinciden del todo con las que crea la lista completa. Esto puede hacer que los participantes queden clasificados con niveles de riesgo distintos por el reducto y por la lista completa. En esta sección se muestra la listas de elementos que no son clasificados de la misma manera. Si las listas están vacias significa que las clasificacones coinciden. Si aparecen números, estos corresponden a los índices de los participantes que son clasificados en niveles de riesgo distintos.')
        X_No_indices = [set(nuevo_dataframe[nuevo_dataframe['nivel_riesgo'] == 'Sin Riesgo'].index.tolist())]
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        L=lower_approximation(R, X_No_indices)
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        upper_approximation(R,  X_No_indices)
        U=upper_approximation(R,  X_No_indices)
        U-L

        X_leve_indices = [set(nuevo_dataframe[nuevo_dataframe['nivel_riesgo'] == 'Riesgo leve'].index.tolist())]
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        L=lower_approximation(R, X_leve_indices)
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        upper_approximation(R,  X_leve_indices)
        U=upper_approximation(R,  X_leve_indices)
        U-L

        X_moderado_indices = [set(nuevo_dataframe[nuevo_dataframe['nivel_riesgo'] == 'Riesgo moderado'].index.tolist())]
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        L=lower_approximation(R, X_moderado_indices)
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        upper_approximation(R,  X_moderado_indices)
        U=upper_approximation(R,  X_moderado_indices)
        U-L

        X_severo_indices = [set(nuevo_dataframe[nuevo_dataframe['nivel_riesgo'] == 'Riesgo severo'].index.tolist())]
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        L=lower_approximation(R, X_severo_indices)
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        upper_approximation(R,  X_severo_indices)
        U=upper_approximation(R,  X_severo_indices)
        U-L


#######################

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


    # Título de la aplicación
    st.subheader("Creación de un modelo de árbol de decisión")

    st.markdown("""
    En esta sección se obtiene un modelo de **arbol de decisión** que permite determinar las reglas de clasificación, a partir de las preguntas que conforman el reducto, y que produce la misma partición que la lista completa. El modelo de árbol que se obtiene permite clasificar incluso a aquellos participantes que no hayan contestado a los cinco cuestionamientos que se necesitan para asignar un nivel de riesgo de padecer sarcopenia. Además, permite establecer una gearquía sobre la importancia relativa que cada pregunta tiene para determinar un nivel de riesgo.
""")    

    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Definir las columnas de atributos
    columnas_atributos = ['C37_18', 'H11_18', 'H5_18', 'H6_18']

    # Separar los datos en atributos (X) y etiquetas (y)
    X = nuevo_dataframe[columnas_atributos]
    y = nuevo_dataframe['nivel_riesgo']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    # Crear el clasificador de árbol de decisión
    clf = DecisionTreeClassifier()

    # Entrenar el clasificador
    clf.fit(X_train, y_train)

    # Predecir las etiquetas para los datos de prueba
    y_pred = clf.predict(X_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)

    # Visualizar el árbol de decisión en Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Para evitar advertencias de Streamlit
    plt.figure(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=columnas_atributos, class_names=clf.classes_)
    st.pyplot()


    with st.expander("**Métricas de evaluación de la precisión del modelo**"):
        # Mostrar la precisión del modelo en Streamlit
        st.write(f'Precisión del modelo: {accuracy:.2f}')

        # Mostrar el reporte de clasificación en Streamlit
        st.subheader("Reporte de Clasificación:")
        st.text(classification_report(y_test, y_pred))

        # Mostrar la matriz de confusión en Streamlit
        st.subheader("Matriz de Confusión:")
        st.write(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=clf.classes_, index=clf.classes_))

##################


#    # Definir las columnas de atributos
#    columnas_atributos = ['C37_18', 'H11_18', 'H5_18', 'H6_18']

#    # Separar los datos en atributos (X) y etiquetas (y)
#    X = nuevo_dataframe[columnas_atributos]
#    y = nuevo_dataframe['nivel_riesgo']

#    # Dividir los datos en conjuntos de entrenamiento y prueba
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#    # Crear el clasificador de árbol de decisión
#    clf = DecisionTreeClassifier(random_state=42)

    # Entrenar el clasificador
#    clf.fit(X_train, y_train)


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
    import seaborn as sns
    import streamlit as st
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    # Definir las columnas de atributos
    #columnas_atributos = ['C37_18', 'H11_18', 'H5_18', 'H6_18']

    # Suponiendo que 'datos_limpios' ya está definido y contiene los datos necesarios

    # Función para asignar nivel de riesgo a una fila
    def asignar_nivel_riesgo(fila, modelo, columnas_atributos):
        X = fila[columnas_atributos].values.reshape(1, -1)
        y_pred = clf.predict(X)
        return y_pred[0]

    # Aplicar la función asignar_nivel_riesgo al DataFrame datos_limpios
    #datos_limpios['Diagnóstico_árbol'] = datos_limpios.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    nuevo_dataframe['Diagnóstico_árbol'] = nuevo_dataframe.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)

    # Mostrar los resultados en Streamlit
    #st.write("Resultados de asignación de nivel de riesgo:")
    #st.dataframe(nuevo_dataframe[['C37_18', 'H11_18', 'H6_18', 'H5_18', 'Diagnóstico_árbol']], use_container_width=True)

    # Calcular el número de coincidencias y no coincidencias
    coincidencias = (nuevo_dataframe['nivel_riesgo'] == nuevo_dataframe['Diagnóstico_árbol']).sum()
    total_filas = len(nuevo_dataframe)
    no_coincidencias = total_filas - coincidencias

    # Mostrar los resultados en Streamlit
    st.write(f"Número de filas en las que coinciden los valores: {coincidencias}")
    st.write(f"Número de filas en las que no coinciden los valores: {no_coincidencias}")
 
    datos_limpios['Diagnóstico_árbol'] = datos_limpios.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    # Aplicar la función asignar_nivel_riesgo al dataframe datos_limpios
    #datos_limpios[['C37_18','H11_18', 'H6_18','H5_18','Diagnóstico_árbol']]

    #datos_filtrados.drop('H15A_18', axis=1, inplace=True) 
    datos_filtrados = datos_limpios_filtrados.dropna()
 
    datos_filtrados['Diagnóstico_árbol'] = datos_filtrados.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    datos_filtrados = datos_filtrados[['H11_18', 'H5_18', 'H6_18','C37_18','Diagnóstico_árbol']].dropna()
    #datos_filtrados[['H11_18', 'H5_18', 'H6_18','C37_18','Diagnóstico_árbol']]

    nuevo_dataframe_filtrado['Diagnóstico_árbol'] = nuevo_dataframe_filtrado.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    #df_filtrado = df.dropna()
    st.dataframe(nuevo_dataframe_filtrado, use_container_width=True)

    st.write(f'La base seleccionada contiene **{datos_filtrados.shape[0]}** filas y **{datos_filtrados.shape[1]}** columnas.')


    # Botón para descargar el dataframe reducido en formato csv
    csv_data = convert_df_to_csv(nuevo_dataframe_filtrado)
    st.download_button(
        label="Descargar Dataframe en formato CSV",
        data=csv_data,
        file_name="dataframe_reducido.csv",
        mime="text/csv"
        )

    xlsx_data = convert_df_to_xlsx(nuevo_dataframe_filtrado)
    st.download_button(
        label="Descargar Dataframe en formato XLSX",
        data=xlsx_data,
        file_name="dataframe_reducido.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown("""
    El modelo de árbol, obtenido a partir del reducto, permite estimar el nivel de riesgo aún en el caso de pacientes que no hayan contestado la encuesta completa. En la Figura debajo se comparan los tamaños de los conjuntos de pacientes en cada nivel de riesgo de acuerdo a la lista completa (izquierda) con la del reducto (derecha).
""")

    # Suponiendo que 'datos_filtrados' ya está definido y contiene los datos necesarios

    # Agrupar por 'Diagnóstico_árbol' y contar el número de ocurrencias para nuevo_dataframe
    grupo_diagnostico_nuevo = nuevo_dataframe.groupby('Diagnóstico_árbol').size()

    # Agrupar por 'Diagnóstico_árbol' y contar el número de ocurrencias para datos_filtrados
    grupo_diagnostico_filtrados = nuevo_dataframe_filtrado.groupby('Diagnóstico_árbol').size()

    # Crear el panel con dos subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico para nuevo_dataframe
    sns.barplot(x=grupo_diagnostico_nuevo.values, y=grupo_diagnostico_nuevo.index, palette='Dark2', ax=axes[0])
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].set_xlabel('Número de filas')
    axes[0].set_ylabel('Diagnóstico')
    axes[0].set_title('Conteo de diagnósticos (Lista completa)')

    # Gráfico para datos_filtrados
    sns.barplot(x=grupo_diagnostico_filtrados.values, y=grupo_diagnostico_filtrados.index, palette='Dark2', ax=axes[1])
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].set_xlabel('Número de filas')
    axes[1].set_ylabel('Diagnóstico')
    axes[1].set_title('Conteo de diagnósticos (Reducto)')

    # Mostrar el panel con subplots en Streamlit
    st.pyplot(fig)

# Crear una nueva columna "Diagnóstico_árbol" en el dataframe "nuevo_dataframe"
#nuevo_dataframe['Diagnóstico_árbol'] = nuevo_dataframe.apply(asignar_nivel_riesgo, axis=1)

    #nuevo_dataframe.shape

    #datos_filtrados.shape

    #df_filtrado.shape



    # Configurar la interfaz de usuario de Streamlit
    st.title("Comparación por comorbilidad")

    st.markdown("""
    En esta sección puede compararse la distribución de niveles de riesgo para pacientes con diversas comorbilidades. En su verisón actual (01-06-2024) se comparan 4 grupos: pacientes sanos, pacientes con diabetes, pacientes con hipertensión y pacientes con diabetes e hipertensión. Para visualizar los resultados es necesario que carge los archivos de participantes de la encuesta con cada tipo de comorbilidad o sanos en los que ya se halla estimado un nivel de riesgo por sarcopenia (estos archivos se obtienen al correr los pasos previos a esta sección, usando el archivo **conjunto_de_datos_sect_a_c_d_f_e_pc_h_i_enasem_2018**.)
""")
    # Cargar hasta 4 archivos
    archivos = st.file_uploader("Cargar archivos (máximo 4 archivos, CSV o Excel)", 
                                type=["csv", "xlsx"], 
                                accept_multiple_files=True, 
                               key="archivos_uploader")

    # Verificar que no se carguen más de 4 archivos
    if len(archivos) > 4:
        st.error("Por favor, carga un máximo de 4 archivos.")
    else:
        # Inicializar una lista para almacenar los dataframes
        dfs = []

        # Procesar cada archivo
        for i, archivo in enumerate(archivos):
            if archivo.name.endswith('.csv'):
                df = pd.read_csv(archivo)
                #st.write(f"Archivo {archivo.name} cargado correctamente como df_{i+1}:")
                st.dataframe(df)
            elif archivo.name.endswith('.xlsx'):
                xls = pd.ExcelFile(archivo)
                sheet_name = xls.sheet_names[0]  # Obtener el nombre de la primera hoja
                df = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl')
                #st.write(f"Archivo {archivo.name} cargado correctamente como df_{i+1}:")
                #st.dataframe(df)
        
            # Asignar el dataframe a una variable dinámica
            globals()[f'df_{i+1}'] = df
            dfs.append(df)

        # Mostrar los nombres de los dataframes creados
        #for i in range(len(dfs)):
        #    st.write(f"df_{i+1} creado.")
        # Concatenar los dataframes verticalmente
    
    # Crear la columna 'comorbilidad' con el valor 'negativo'
    #df_1['Comorbilidad'] = 'Ninguna'
    #df_2['Comorbilidad'] = 'Diabetes'
    #df_3['Comorbilidad'] = 'Hipertensión'
    #df_4['Comorbilidad'] = 'Hipertensión y Diabetes'

    datos_concatenados = pd.concat([df_1, df_2, df_3, df_4], ignore_index=True)

    # Mostrar el dataframe resultante
    st.dataframe(datos_concatenados, use_container_width=True)

    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st


    # Contar el número de ocurrencias de cada valor en la columna "Diagnóstico_árbol"
    diagnosticos_counts = datos_concatenados['Diagnóstico_árbol'].value_counts()

    # Crear el diagrama de pastel
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(diagnosticos_counts, labels=diagnosticos_counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Porcentaje por evaluación de riesgo de Sarcopenia')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Agregar leyenda de texto
    plt.text(-1.3, -1.1, "Muestra de la ENASEM de 2018 de voluntarios mayores de 60 años", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)



#    # Crear figura y ejes para los subplots
#    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

#    # Iterar sobre cada nivel de riesgo y crear un gráfico de pastel
#    for i, nivel in enumerate(nivel_riesgo):
#        # Filtrar el dataframe para obtener solo las filas con el nivel de riesgo actual
#        filtro_riesgo = datos_concatenados[datos_concatenados['Diagnóstico_árbol'] == nivel]
    
#        # Contar el número de ocurrencias de cada valor en la columna "Comorbilidad"
#        comorbilidad_counts = filtro_riesgo['Comorbilidad'].value_counts()
    
#        # Crear el gráfico de pastel en el subplot correspondiente
#        ax = axs[i // 2, i % 2]
#        ax.pie(comorbilidad_counts, labels=comorbilidad_counts.index, autopct='%1.1f%%', startangle=140)
#        ax.set_title(f'Porcentaje de comorbilidades - {nivel}')
#        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

#    # Agregar leyenda de texto en el subplot inferior izquierdo
#    fig.text(0.5, 0.02, "Muestra de la ENASEM de 2021 de voluntarios mayores de 60 años", ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

#    # Ajustar diseño de subplots
#    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

#    # Mostrar el gráfico en Streamlit
#    st.pyplot(fig)





    st.title("Análisis de Comorbilidades y Diagnóstico de Sarcopenia")

    # Agrupar por "Diagnóstico árbol" y "Comorbilidad" y contar el número de filas en cada grupo
    grupo_diagnostico_comorbilidad = datos_concatenados.groupby(['Diagnóstico_árbol', 'Comorbilidad']).size().unstack(fill_value=0)

    # Calcular el porcentaje para cada subgrupo
    grupo_diagnostico_comorbilidad_porcentaje = grupo_diagnostico_comorbilidad.div(grupo_diagnostico_comorbilidad.sum(axis=1), axis=0) * 100

    # Ordenar las barras según las categorías "Sin Riesgo", "Riesgo leve", "Riesgo moderado" y "Riesgo considerable"
    orden_categorias = ["Sin Riesgo", "Riesgo leve", "Riesgo moderado", "Riesgo severo"]
    grupo_diagnostico_comorbilidad_porcentaje = grupo_diagnostico_comorbilidad_porcentaje.reindex(orden_categorias)

    # Crear el diagrama de barras con colores específicos para cada tipo de comorbilidad
    fig, ax = plt.subplots(figsize=(10, 6))
    grupo_diagnostico_comorbilidad_porcentaje.plot(kind='bar', stacked=True, ax=ax, color=['green', 'orange', 'red', 'blue'])

    # Configurar el gráfico 
    plt.title('Muestra de la ENASEM de 2018 de voluntarios mayores de 60 años')
    plt.xlabel('Diagnóstico árbol')
    plt.ylabel('Porcentaje')
    plt.xticks(rotation=45)
    plt.legend(title='Comorbilidad', bbox_to_anchor=(1.01, 0.89), loc='center left')

    # Mostrar los porcentajes numéricos dentro de las barras
    for i, (index, row) in enumerate(grupo_diagnostico_comorbilidad_porcentaje.iterrows()):
        acumulado = 0
        for j, value in enumerate(row):
            if value != 0:  # Mostrar el valor solo si no es cero
                altura_fraccion = acumulado + value / 2
                ax.text(i, altura_fraccion, f'{value:.1f}%', ha='center', va='center', color='white')
                acumulado += value

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)


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



    st.title('Estimación del nivel de riesgo por sarcopenia')

    st.write("""En esta sección se calcula el **riesgo de padecer sarcopenia** a partir de las respuestas de las y los participantes de la **Encuesta Nacional Sobre Salud y Envejecimiento**. Esto se hace partiendo de la identificación de las preguntas de la encuesta que guarden la mayor similitud posible con las que contiene el cuestionario **SARC-F**.
             """) 
    st.markdown("""
    1. **Depuración de datos**: se eliminan datos de pacientes que no cumplan con los criterios de inclusión o presenten registros incompletos. Además se definen 5 cuestionamientos de la ENASEM que guardan similitud con SARC-F y se crea una submuestra de participantes que hayan contestado a estos cuestionamientos.

    2. **Clasificación de participantes**: Usando la teoría de conjuntos rugosos, se divide la base de datos en una colección de subconjuntos de pacientes que hayan contestado idénticamente a las preguntas clave (a estos subconjuntos se les llama relaciones de indiscernibilidad).

    3. **Obtención de reglas de decisión**: Se entrena un modelo de árbol de decisión para determinar un conjunto de reglas que permitan clasificar a los pacientes de la base de datos (aún aquellos que inicialmente no tenían respuestas completas en todas las preguntas de interés).
""")
    
    st.subheader("Carga y depuración de datos")
    st.write("""
    Por favor, cargue un archivo correspodiente a las secciones **conjunto_de_datos_sect_a_c_d_f_e_pc_h_i_enasem_2021**. El archivo debe estar en formato csv y debe aparecer debajo, si se cargo correctamente. 
""")
    # Crear una caja de carga de archivos
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

    if uploaded_file is not None:
        # Cargar el archivo como un DataFrame de pandas
        df = pd.read_csv(uploaded_file)

    #df.columns = df.columns.str.replace(r'(_21|_19)$', '', regex=True)

    # Mostrar el DataFrame cargado
    #df.columns = df.columns.str.replace('_21', '', regex=False)
    #df.columns = df.columns.str.replace('_21', '', regex=False).str.replace('_21', '', regex=False)
    st.write('Cada **fila** corresponde a las respuestas de una o un participante de la ENASEM y cada **columna** corresponde a un pregunta en particular de las **secciones "a" a "i"** (si quiere revisar el significado de las claves de las preguntas revise la sección siguiente). Los registros vacíos (aquellos que muestren un **None**), los que contengan repuestas **"8" o "9"** (**"No sabe"** y **"No quiere contestar"** y los que tengan (**999**) **se eliminarán en la depuración**).')
    st.dataframe(df)

    st. subheader("Selección de variables de interés")
    st.write("""
             En esta sección puede elegir entre dos posibles listas de variables de interés:
             - La **selección estándar**: contiene una lista de variables que pude econtrarse en el apéndice (vea la parte final de esta página).

             - **Lista personalizada**: Si selecciona esta opción aparecerá una barra en la que puede elegir entre las variables contenida en la base de datos para su búsqueda (**Nota:** para que el código funcione correctamente, su lista debe incluir a las siguientes variables: )
             """)

    # Ejemplo de contenido colapsable
    with st.expander("Listado de variables incluidas en la **selección estándar**"):
    
        st.write("""
                 
- **Código AGE_21**: Edad en años cumplidos.

- **Código SEX_21**: Sexo.

- **Código C4_21**: ¿Alguna vez le ha dicho un doctor o personal médico que usted tiene hipertensión o presión alta?

- **Código C6_21**: ¿Alguna vez le ha dicho un doctor o personal médico que usted tiene diabetes?

- **Código C12_21**: ¿Alguna vez le ha dicho un doctor o personal médico que usted tiene cáncer?

- **Código C19_21**: ¿Alguna vez le ha dicho un doctor o personal médico que usted tiene alguna enfermedad respiratoria, tal como asma o enfisema?

- **Código C22A_21**: ¿Alguna vez le ha dicho un doctor o personal médico que usted ha tenido un ataque/infarto al corazón?

- **Código C26_21**: ¿Alguna vez le ha dicho un doctor o personal médico que usted ha tenido una embolia cerebral, derrame cerebral o isquemia cerebral transitoria?

- **Código C32_21**: ¿Alguna vez le ha dicho un doctor o personal médico que usted tiene artritis o reumatismo?

- **Código C37_21**: ¿Se ha caído en los últimos dos años?

- **Código C49_1_21**: Estas preguntas se refieren a cómo se ha sentido usted durante la semana pasada. Para cada pregunta, por favor dígame, ¿la mayor parte del tiempo se ha sentido deprimido?

- **Código C49_2_21**: Estas preguntas se refieren a cómo se ha sentido usted durante la semana pasada. Para cada pregunta, por favor dígame, ¿la mayor parte del tiempo ha sentido que todo lo que hacía era un esfuerzo?

- **Código C49_8_21**: Estas preguntas se refieren a cómo se ha sentido usted durante la semana pasada. Para cada pregunta, por favor dígame, ¿la mayor parte del tiempo se ha sentido cansado?

- **Código C64_21**: ¿Comparado con hace dos años, usted...?

- **Código C66_21**: ¿Como cuántos kilos pesa usted ahora?

- **Código C67_1_21**: ¿Como cuánto mide usted sin zapatos? - Metros

- **Código C67_2_21**: ¿Como cuánto mide usted sin zapatos? - Centímetros

- **Código C68E_21**: Durante los últimos dos años, ¿ha tenido alguno de los siguientes problemas o molestias frecuentemente? - Fatiga severa o agotamiento serio

- **Código C68G_21**: Durante los últimos dos años, ¿ha tenido alguno de los siguientes problemas o molestias frecuentemente? - Pérdida involuntaria de orina, al hacer cosas como toser, estornudar, recoger cosas o hacer ejercicio

- **Código C68H_21**: Durante los últimos dos años, ¿ha tenido alguno de los siguientes problemas o molestias frecuentemente? - Pérdida involuntaria de orina, cuando tenía urgencia de orinar pero no pudo llegar al baño a tiempo

- **Código C69A_21**: ¿Cómo evaluaría la fuerza de su mano (la que utiliza más)?, ¿diría que es...?

- **Código C69B_21**: ¿Qué tan seguido tiene usted dificultad en mantener su equilibrio/balance?, ¿diría que...?

- **Código C71A_21**: ¿Le falta alguna extremidad o parte de sus piernas o brazos debido a un accidente o enfermedad?

- **Código C76_21**: En los últimos 12 meses, ¿cuánto efecto cree usted que el estrés ha tenido sobre su salud?

- **Código H1_21**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene usted dificultad en caminar varias cuadras?

- **Código H4_21**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en estar sentado(a) por dos horas?

- **Código H5_21**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en levantarse de una silla después de haber estado sentado(a) durante largo tiempo?

- **Código H6_21**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en subir varios pisos de escaleras sin descansar?

- **Código H8_21**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en inclinar su cuerpo, arrodillarse, agacharse o ponerse en cuclillas?

- **Código H9_21**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en subir o extender los brazos más arriba de los hombros?

- **Código H10_21**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud ¿tiene alguna dificultad en jalar o empujar objetos grandes como un sillón?

- **Código H11_21**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en levantar o transportar objetos que pesan más de 5 kilos, como una bolsa pesada de alimentos?

- **Código H12_21**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene alguna dificultad en recoger una moneda de 1 peso de la mesa?

- **Código H13_21**: Dígame por favor si usted tiene alguna dificultad en hacer cada una de las actividades diarias que le voy a mencionar. No incluya dificultades que cree que durarán menos de tres meses. Debido a problemas de salud, ¿tiene usted dificultad para vestirse, incluyendo ponerse los zapatos y los calcetines?

- **Código H15A_21**: Por favor dígame si tiene alguna dificultad con cada una de las actividades que le voy a mencionar. Si usted no hace ninguna de las siguientes actividades, simplemente dígamelo. No incluya dificultades que cree que durarán menos de tres meses. Debido a un problema de salud ¿usted tiene dificultad para caminar de un lado a otro de un cuarto?

- **Código H15B_21**: ¿Usa usted equipo o aparatos, tales como bastón, caminador o silla de ruedas para caminar de un lado a otro de un cuarto?

- **Código H15D_21**: ¿Alguien le ayuda a usted para caminar de un lado a otro de un cuarto?

- **Código H16A_21**: Por favor dígame si tiene alguna dificultad con cada una de las actividades que le voy a mencionar. Si usted no hace ninguna de las siguientes actividades, simplemente dígamelo. No incluya dificultades que cree que durarán menos de tres meses. Debido a un problema de salud ¿usted tiene dificultad para bañarse en una tina o regadera?

- **Código H16D_21**: ¿Alguien le ayuda a usted para bañarse en una tina o regadera?

- **Código H17A_21**: Por favor dígame si tiene alguna dificultad con cada una de las actividades que le voy a mencionar. Si usted no hace ninguna de las siguientes actividades, simplemente dígamelo. No incluya dificultades que cree que durarán menos de tres meses. Debido a un problema de salud ¿usted tiene dificultad al comer, por ejemplo para cortar su comida?

- **Código H17D_21**: ¿Alguien le ayuda a usted al comer, por ejemplo para cortar su comida?

- **Código H18A_21**: Por favor dígame si tiene alguna dificultad con cada una de las actividades que le voy a mencionar. Si usted no hace ninguna de las siguientes actividades, simplemente dígamelo. No incluya dificultades que cree que durarán menos de tres meses. Debido a un problema de salud ¿usted tiene dificultad al acostarse y levantarse de la cama?

- **Código H18D_21**: ¿Alguien le ayuda a usted al acostarse y levantarse de la cama?

- **Código H19A_21**: Por favor dígame si tiene alguna dificultad con cada una de las actividades que le voy a mencionar. Si usted no hace ninguna de las siguientes actividades, simplemente dígamelo. No incluya dificultades que cree que durarán menos de tres meses. Debido a un problema de salud ¿usted tiene dificultad al usar el excusado, incluyendo subirse y bajarse o ponerse en cuclillas?

- **Código H19D_21**: ¿Alguien le ayuda a usted al usar el excusado, incluyendo subirse y bajarse o ponerse en cuclillas?
    """)


    # Opcional: mostrar estadísticas básicas del DataFrame
    st.write(f'**Descripción de la base de datos:** La base seleccionada contiene **{df.shape[0]}** filas y **{df.shape[1]}** columnas.')
    df['Indice'] = df.index

    # Lista predefinida "selección estándar"
    seleccion_estandar = ['Indice',"AGE_21", 'SEX_21', 'C4_21', 'C6_21', 'C12_21', 'C19_21', 'C22A_21', 'C26_21', "C32_21", 'C37_21',
                      "C49_1_21", 'C49_2_21', 'C49_8_21', 'C64_21', 'C66_21', 'C67_1_21', 'C67_2_21', 'C68E_21', 'C68G_21',
                      'C68H_21', 'C69A_21', 'C69B_21', 'C71A_21', 'C76_21', 'H1_21', 'H4_21', 'H5_21', 'H6_21', 'H8_21',
                      'H9_21', 'H10_21', 'H11_21', 'H12_21', 'H13_21', 'H15A_21', 'H15B_21', 'H15D_21', 'H16A_21', 'H16D_21',
                      'H17A_21', 'H17D_21', 'H18A_21', 'H18D_21', 'H19A_21', 'H19D_21']
###

# Opción para seleccionar columnas
    opcion_seleccion = st.radio("¿Cómo quieres seleccionar las columnas?", ("Usar selección estándar", "Usar lista personalizada"))

    if opcion_seleccion == "Usar selección estándar":
        selected_columns = seleccion_estandar
    else:
        selected_columns = st.multiselect("Usar lista personalizada", df.columns.tolist())

    # Lista de verificación para seleccionar columnas
    #selected_columns = st.multiselect("Selecciona las columnas para mostrar", df.columns.tolist())
        
    if selected_columns:
        # Crear dataframe reducido
        df = df[selected_columns]
            
        st.write("Base de datos con la **lista de variables seleccionada:**")
        st.write(df)
            
        # Mostrar información del dataframe reducido
        num_rows, num_cols = df.shape
        #st.write(f"Número de filas: {num_rows}")
        #st.write(f"Número de columnas: {num_cols}")
        # Opcional: mostrar estadísticas básicas del DataFrame
        st.write(f'**Descripción de la base de datos:** La base seleccionada contiene **{df.shape[0]}** filas y **{df.shape[1]}** columnas.')


        # Contar valores NaN por columna
        nan_counts = df.isna().sum().reset_index()
        nan_counts.columns = ["Clave", "Conteo de filas vacias"]
            
        #st.write("Conteo de valores NaN por columna:")
        #st.write(nan_counts)
        #st.dataframe(nan_counts, use_container_width=True)

####################################

        import pandas as pd
        import gdown
        import streamlit as st

        # Función para cargar el diccionario
        def cargar_diccionario(url, nombre):
            output = f'{nombre}.xlsx'
            gdown.download(url, output, quiet=False)

            try:
                # Intentar leer el archivo Excel
                xls = pd.ExcelFile(output)
                sheet_name = xls.sheet_names[0]  # Obtener el nombre de la primera hoja
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None, engine='openpyxl')
                #st.write("Archivo xlsx cargado correctamente.")
            except Exception as e:
                st.error(f"Error al leer el archivo xlsx: {e}")
                return {}

            diccionario = {}
            for index, row in df.iterrows():
                variable = row[0]
                if isinstance(variable, str) and variable.startswith("Pregunta"):
                    partes = variable.split(" ", 2)
                    if len(partes) < 3:
                        continue
                    codigo = partes[1].replace('.', '_')
                    explicacion = partes[2]
                    diccionario[codigo] = explicacion

            # Agregar explicaciones adicionales
            diccionario.update({
                "Indice": "El número de fila en la base de datos",
                "AGE_21": "Edad en años cumplidos",
                "SEX_21": "Sexo"
            })

            return diccionario

        # URL del archivo de 2021 en Google Drive
        url_2021 = 'https://drive.google.com/uc?id=17o5hDLk_RHU6kGKinJtmRGtApv927Abu'

        # Interfaz en Streamlit
        st.write("Estas son las variables que seleccionó (la primera columna corresponde a la **Clave**, la segunda es el **conteo de filas vacías** de cada variable, la tercera es la **explicación** de la clave para esa variable).")

        # Inicializar el estado de la sesión para el historial de búsquedas
        if 'historico_busquedas' not in st.session_state:
            st.session_state.historico_busquedas = pd.DataFrame(columns=['Año', 'Código', 'Explicación'])

        # Cargar el diccionario de 2021
        diccionario_2021 = cargar_diccionario(url_2021, 'diccionario_2021')

        df_c=df.copy()
        # Obtener la lista de nombres de columnas
        columnas = df_c.columns.tolist()

        # Contar valores NaN por columna
        nan_counts = df_c.isna().sum().reset_index()
        nan_counts.columns = ["Clave", "Conteo"]

        # Agregar explicaciones al DataFrame nan_counts
        nan_counts['Explicación'] = nan_counts['Clave'].map(diccionario_2021)

        # Mostrar el DataFrame con el conteo de valores NaN y explicaciones
        #st.write("Conteo de valores NaN por columna con explicaciones:")
        st.dataframe(nan_counts, use_container_width=True)




######################################

        # Filtro inicial
        df = df[df['AGE_21'] < 100]

        # Lista de columnas a modificar
        columnas_modificar = ['C4_21', 'C6_21', 'C12_21', 'C19_21', 'C22A_21', 'C26_21', "C32_21", 'C37_21',
                      "C49_1_21", 'C49_2_21', 'C49_8_21', 'C64_21', 'C66_21', 'C68E_21', 'C68G_21',
                      'C68H_21', 'C69A_21', 'C69B_21', 'C71A_21', 'C76_21', 'H1_21', 'H4_21', 'H5_21', 'H6_21', 'H8_21',
                      'H9_21', 'H10_21', 'H11_21', 'H12_21', 'H13_21', 'H15A_21', 'H15B_21', 'H15D_21', 'H16A_21', 'H16D_21',
                      'H17A_21', 'H17D_21', 'H18A_21', 'H18D_21', 'H19A_21', 'H19D_21']

        # Convertir valores 6.0 o 7.0 en 1.0 en las columnas especificadas
        df[columnas_modificar] = df[columnas_modificar].replace({6.0: 1.0, 7.0: 1.0})

        # Combinar los campos de las columnas de estatura en una sola columna de estatura en metros
        df['C67_21'] = df['C67_1_21'] + df['C67_2_21'] / 100
        df = df.drop(columns=['C67_1_21', 'C67_2_21'])

        # Eliminar filas que contengan valores 8.0 o 9.0 en cualquiera de las columnas especificadas
        df = df[~df[columnas_modificar].isin([8.0, 9.0, 999, 9.99]).any(axis=1)]
        df = df[~df['C67_21'].isin([9.99, 8.88])]
        columnas_seleccionadas = list(df.columns)

        # Crear widgets de selección para cada columna seleccionada en la barra lateral
        filtros = {}
        for col in columnas_seleccionadas:
            if df[col].dtype == 'object':
                valores_unicos = df[col].unique().tolist()
                seleccion = st.sidebar.multiselect(f'Seleccionar valores para {col}', valores_unicos)
                if seleccion:
                    filtros[col] = seleccion
            else:
                rango = st.sidebar.slider(f'Seleccionar rango para {col}', min_value=float(df[col].min()), max_value=float(df[col].max()), value=(float(df[col].min()), float(df[col].max())), step=1.0)
                if rango:
                    filtros[col] = rango

        # Filtrar el DataFrame basado en los valores seleccionados
        df_filtrado = df.copy()
        for col, condicion in filtros.items():
            if isinstance(condicion, list):
                df_filtrado = df_filtrado[df_filtrado[col].isin(condicion)]
            else:
                df_filtrado = df_filtrado[(df_filtrado[col] >= condicion[0]) & (df_filtrado[col] <= condicion[1])]

        #st.write('DataFrame Filtrado')
        st.dataframe(df_filtrado, use_container_width=True)
        #st.write("Las dimensiones de la base de datos son:")
        st.write(df_filtrado.shape)
        datos_filtrados = df_filtrado.copy()

#######################################


     # Definir condiciones para cada grupo
        conditions = {
            "Ninguna": {
                'C4_21': 2.0,
                'C6_21': 2.0,
                'C12_21': 2.0,
                'C19_21': 2.0,
                'C22A_21': 2.0,
                'C26_21': 2.0,
                'C32_21': 2.0
            },
            "Diabetes": {
                'C4_21': 1.0,
                'C6_21': 2.0,
                'C12_21': 2.0,
                'C19_21': 2.0,
                'C22A_21': 2.0,
                'C26_21': 2.0,
                'C32_21': 2.0
            },
            "Hipertensión": {
                'C4_21': 2.0,
                'C6_21': 1.0,
                'C12_21': 2.0,
                'C19_21': 2.0,
                'C22A_21': 2.0,
                'C26_21': 2.0,
                'C32_21': 2.0
            },
            "Hipertensión y Diabetes": {
                'C4_21': 1.0,
                'C6_21': 1.0,
                'C12_21': 2.0,
                'C19_21': 2.0,
                'C22A_21': 2.0,
                'C26_21': 2.0,
                'C32_21': 2.0
            }
        }


        st.markdown(
            """
            Aquí pude seleccionar entre hacer un análisis utilizando datos de la muestra completa o aislar un grupo de interés. Los grupos disponibles son:

            - **Ninguna:** Declara **no tener diagnóstico de Diabetes o Hipertensión**.

            - **Diabetes:** Declara tener diagnóstico **negativo en Hipertensión** pero **positivo en Diabetes**.

            - **Hipertensión:** Declara tener diagnóstico **positivo en Hipertensión**  pero **negativo en Diabetes**.

            - **Hipertensión y Diabetes:** Declara tener diagnóstico **positivo tanto en Hipertensión como en Diabetes**.
            
            """
            )

        # Crear una selección en Streamlit para elegir entre los conjuntos
        seleccion = st.selectbox("**Seleccione un grupo**", list(conditions.keys()))

        # Crear una selección múltiple en Streamlit para el valor de SEX_21
        sex_values = df_filtrado['SEX_21'].unique()
        st.write("""Aquí pude seleccionar entre hacer un análisis utilizando datos de la muestra completa o sobre un solo sexo. La Clave numérica **"1.0"** corresponde a las **mujeres** y la **Clave "2.0"** corresponde a los **hombres.**
        """)
        sex_selection = st.multiselect("**Seleccione el sexo de la muestra** (puede seleccionar ambos si quiere analizar la muestra completa)", sex_values, default=sex_values)

        # Filtrar el DataFrame en función de las condiciones seleccionadas y el valor de SEX_21
        condiciones_seleccionadas = conditions[seleccion]
        nuevo_dataframe_filtrado = df_filtrado.copy()

        # Aplicar las condiciones seleccionadas
        for columna, valor in condiciones_seleccionadas.items():
            nuevo_dataframe_filtrado = nuevo_dataframe_filtrado[nuevo_dataframe_filtrado[columna] == valor]

        # Aplicar el filtro del valor de SEX_21
        #nuevo_dataframe_filtrado = nuevo_dataframe_filtrado[nuevo_dataframe_filtrado['SEX_21'] == sex_selection]
        # Aplicar el filtro del valor de SEX_21
        nuevo_dataframe_filtrado = nuevo_dataframe_filtrado[nuevo_dataframe_filtrado['SEX_21'].isin(sex_selection)]
        nuevo_dataframe_filtrado['Comorbilidad'] = seleccion


        st.write("**")
        # Mostrar el DataFrame filtrado
        st.dataframe(nuevo_dataframe_filtrado, use_container_width=True)
        datos_limpios = nuevo_dataframe_filtrado.copy()
        datos_limpios = datos_limpios.dropna()
        st.write(f'La base depurada contiene **{datos_limpios.shape[0]}** filas y **{datos_limpios.shape[1]}** columnas.')
        st.dataframe(datos_limpios, use_container_width=True)
        datos_limpios.shape

##########################################
    st.subheader("Clasificación de participantes.")
    st.markdown(
        """
        En esta sección se utiliza la **Teoría de conjuntos rugosos** para agrupar a las y los participantes de la encuesta cuyas respuestas fueron idénticas. Esto se logra mediante el cálculo de las **relaciones de indiscerbibilidad** El gáfico de representa a las particiones de pacientes con respuestas idénticas en las preguntas (**'C37_21', 'H11_21', 'H15A_21', 'H5_21', 'H6_21'**).

"""
)
    ind=indiscernibility(['C37_21', 'H11_21', 'H15A_21', 'H5_21', 'H6_21'], datos_limpios)
    

    import matplotlib.pyplot as plt


    # Calcular las longitudes de los conjuntos con longitud >= 2
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(ind) if len(conjunto) >= 2]

    # Ordenar la lista de tuplas por la longitud de los conjuntos en orden descendente
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Crear un diccionario para mapear el índice del conjunto al nuevo nombre
    nombres_conjuntos_nuevos = {num_conjunto: f"Conjunto {i}" for i, (num_conjunto, _) in enumerate(longitudes_conjuntos_ordenadas)}

    #st.write("Longitudes de los conjuntos con su respectivo nuevo nombre (en orden descendente):")
    #for num_conjunto, longitud in longitudes_conjuntos_ordenadas:
    #    nombre_conjunto_nuevo = nombres_conjuntos_nuevos[num_conjunto]
    #    st.write(f"{nombre_conjunto_nuevo}: {longitud}")

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
    # Agregar el tamaño de la muestra total como texto
    ax.annotate(f'Tamaño de la muestra total: {total_elementos}', 
            xy=(0.5, -0.05), 
            xycoords='axes fraction', 
            ha='center', 
            fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Distribución de subconjuntos')

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)
    st.write("Seleccione en el recuadro el **número mínimo de miembros** de los conjuntos que considerará. Esto simplifica el análisis. **Típicamente se define un número mínimo de 30 miembros** (o de 10 miembros si la muestra total es demasiado pequeña o si las relaciones de indiscernibilidad creadas tienen, en su mayoría, menos de 30 miembros).")


    # Entrada del usuario para el tamaño mínimo del conjunto
    tamaño_mínimo = st.number_input("**Defina el número mínimo de miebros que tendrán los conjuntos a considerar:**", min_value=1, value=2, step=1)

    # Calcular las longitudes de los conjuntos con longitud >= tamaño_mínimo
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(ind) if len(conjunto) >= tamaño_mínimo]

    # Ordenar la lista de tuplas por la longitud de los conjuntos en orden descendente
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Crear un diccionario para mapear el índice del conjunto al nuevo nombre
    nombres_conjuntos_nuevos = {num_conjunto: f"**Conjunto {i}**" for i, (num_conjunto, _) in enumerate(longitudes_conjuntos_ordenadas)}


##
    st.write("Estos son los conjuntos considerados y su número de miembros:")
#    for num_conjunto, longitud in longitudes_conjuntos_ordenadas:
#        nombre_conjunto_nuevo = nombres_conjuntos_nuevos[num_conjunto]
#        st.write(f"{nombre_conjunto_nuevo}: {longitud}")
##
#st.write("Longitudes de los conjuntos con su respectivo nuevo nombre (en orden descendente):")

    #  Crear una fila con columnas
    cols = st.columns(3)

    # Rellenar cada columna con datos
    for i, (num_conjunto, longitud) in enumerate(longitudes_conjuntos_ordenadas):
        nombre_conjunto_nuevo = nombres_conjuntos_nuevos[num_conjunto]
        cols[i % 3].write(f"{nombre_conjunto_nuevo}: {longitud}")


    # Calcular el total de elementos en todos los conjuntos
    total_elementos = sum(longitud for _, longitud in longitudes_conjuntos_ordenadas)

    # Calcular los porcentajes para cada conjunto
    porcentajes = [(nombres_conjuntos_nuevos[num_conjunto], longitud / total_elementos * 100) for num_conjunto, longitud in longitudes_conjuntos_ordenadas]

    # Extraer los nombres de los conjuntos y los porcentajes para el diagrama de pastel
    nombres_conjuntos = [nombre for nombre, _ in porcentajes]
    porcentajes_valores = [valor for _, valor in porcentajes]

    # Crear el diagrama de pastel
    #fig, ax = plt.subplots(figsize=(10, 8))
    #_, _, autopcts = ax.pie(porcentajes_valores, labels=nombres_conjuntos, autopct='%1.1f%%', startangle=140, textprops={'visible': False})
    #for autopct in autopcts:
    #    autopct.set_visible(True)  # Mostrar los porcentajes
    #ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    #ax.set_title('Distribución de subconjuntos')

    # Mostrar el gráfico en Streamlit
    #st.pyplot(fig)

    import numpy as np
    
    # Calcular las longitudes de los conjuntos con longitud >= tamaño_mínimo
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(ind) if len(conjunto) >= tamaño_mínimo]

    # Ordenar la lista de tuplas por la longitud de los conjuntos en orden descendente
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)
    
    num_conjuntos = st.number_input("Define el número de conjuntos para vizualizar los perfiles que los caracterizan:", min_value=1, max_value=len(ind), value=15)

    # Obtener los 15 conjuntos más numerosos
    conjuntos_mas_numerosos = [conjunto for i, conjunto in enumerate(ind) if i in [num_conjunto for num_conjunto, _ in longitudes_conjuntos_ordenadas[:num_conjuntos]]]

    # Crear DataFrames para cada uno de los 15 conjuntos más numerosos
    for i, conjunto in enumerate(conjuntos_mas_numerosos, 0):
        indices_seleccionados = list(conjunto)
        df_conjunto = datos_limpios[datos_limpios.index.isin(indices_seleccionados)]
        globals()[f"df_Conjunto_{i}"] = df_conjunto

    # Definir las columnas de interés
    columnas_interes_radar = ['C37_21', 'H11_21', 'H15A_21', 'H5_21', 'H6_21']

    # Definir los nombres de los dataframes
    nombres_dataframes = [f"df_Conjunto_{i}" for i in range(0, num_conjuntos)]

    # Definir los valores para cada dataframe en las columnas de interés
    valores_dataframes = []
    for nombre_df in nombres_dataframes:
        df = eval(nombre_df)
        valores = df[columnas_interes_radar].iloc[0].tolist()  # Tomar solo la primera fila
        valores_dataframes.append(valores)

    st.write("A continuación se muestran los perfiles de las relaciones de indiscernibilidad, en términos de los cinco ítems de la encuesta que guardan mayor similitud con los del test SARC-F. Los **gráficos de radar** muestran las respuestas a estos ítems:")
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
    filas_seleccionadas = filas_seleccionadas[['C37_21', 'H11_21', 'H15A_21', 'H5_21', 'H6_21', 'num_conjunto', 'Comorbilidad']]

    # Crear un nuevo DataFrame con las filas seleccionadas
    nuevo_dataframe = pd.DataFrame(filas_seleccionadas)

    # Mostrar las primeras filas del nuevo DataFrame
    #st.dataframe(nuevo_dataframe, use_container_width=True)



#######################

#####################333

#    import pandas as pd

#    # Definir las condiciones para asignar los valores a la nueva columna
#    def asignar_riesgo(num_conjunto):
#        if num_conjunto in [6, 14]:
#            return "Riesgo considerable"
#        elif num_conjunto in [3, 5, 10]:
#            return "Riesgo moderado"
#        elif num_conjunto in [1, 2, 4, 7, 8, 9, 11, 12, 13]:
#            return "Riesgo leve"
#        elif num_conjunto == 0:
#            return "Sin Riesgo"
#        else:
#            return "No clasificado"  # Manejar cualquier otro caso
    with st.expander("Determinación de un nivel de riesgo"):
        st.markdown("""
                    Ya que las preguntas de la ENASEM no son idénticas a las del cuestionario SARC-F (las respuestas de ese cuestionario permiten establecer una escala de intensidad de dificultad para realizar ciertas actividades, mientras que la ENASEM solo permiten contestar *si* o *no*), se definió un criterio alternativo mediante el cual pudiera establecerse un nivel de riesgo de padecer sarcopenia. Los niveles de riesgo definidos son:

                    - **Sin riesgo:** No se manifiesta tener dificultad en ninguno de los 5 cuestionamientos (o caídas recientes, en el caso de esa pregunta).

                    - **Riesgo leve:** Se manifiesta tener dificultades en uno o dos de los cuestionamientos (o dificultad en uno y caídas recientes, en el caso de esa pregunta).

                    - **Riesgo moderado:** Se manifiesta tener dificultades simultaneas en tres de los cuestionamientos (o dos cuestinamientos y caidas recientes).

                    - **Riesgo severo:** Se manifiesta tener dificultades en cuatro o cinco de de los cuestionamientos (o en cuatro de ellos y caidas recientes).
                    """)

    
#    # Función para determinar el nivel de riesgo
#    def asignar_nivel_riesgo(row):
#        valores = row[['C37_21', 'H11_21', 'H15A_21', 'H5_21', 'H6_21']]
#        if all(valores == 2.0):
#            return "Sin Riesgo"
#        cuenta_1_0 = (valores == 1.0).sum()
#        if cuenta_1_0 == 1 or cuenta_1_0 == 2:
#            return "Riesgo leve"
#        elif cuenta_1_0 == 3:
#            return "Riesgo moderado"
#        elif cuenta_1_0 >= 4:
#            return "Riesgo severo"
#    # Función para determinar el nivel de riesgo

    def asignar_nivel_riesgo(row):
        valores = row[['C37_21', 'H11_21', 'H15A_21', 'H5_21', 'H6_21']]
    
    # Contar cuántas columnas individualmente tienen el valor 1.0
        cuenta_1_0 = (valores == 1.0).sum()
    
    # Verificar las condiciones para cada nivel de riesgo
        if all(valores == 2.0):
            return "Sin Riesgo"
        elif cuenta_1_0 == 1 or cuenta_1_0 == 2:
            return "Riesgo leve"
        elif cuenta_1_0 == 3:
            return "Riesgo moderado"
        elif cuenta_1_0 == 4 or cuenta_1_0 == 5:
            return "Riesgo severo"
        else:
            return "Nivel de riesgo no determinado"  # En caso de que ninguna condición coincida, lo cual no debería ocurrir.




    # Aplicar la función a cada fila del DataFrame
    nuevo_dataframe['nivel_riesgo'] = nuevo_dataframe.apply(asignar_nivel_riesgo, axis=1)

    # Mostrar el DataFrame con el nivel de riesgo asignado
    #st.write(nuevo_dataframe)


    # Agregar la nueva columna al DataFrame
    #nuevo_dataframe['nivel_riesgo'] = nuevo_dataframe['num_conjunto'].apply(asignar_riesgo)
    
    st.write("Debajo puede ver y descargar la base de datos de las respuestas que se usan para estimar el nivel de riesgo de sarcopenia, la etiqueta que indica al conjunto que corresponde cada paciente y el nivel de riesgo asociado.")
    
    st.dataframe(nuevo_dataframe, use_container_width=True)

    # Botón para descargar el dataframe reducido en formato csv
    csv_data = convert_df_to_csv(nuevo_dataframe)
    st.download_button(
        label="Descargar Dataframe en formato CSV",
        data=csv_data,
        file_name="Base de relaciones de indiscernibilidad y nivel de riesgo_2021.csv",
        mime="text/csv"
        )

    xlsx_data = convert_df_to_xlsx(nuevo_dataframe)
    st.download_button(
        label="Descargar Dataframe en formato XLSX",
        data=xlsx_data,
        file_name="Base de relaciones de indiscernibilidad y nivel de riesgo_2021.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )




#######################


#    # Función para calcular la diferencia entre tamaños de conjuntos
#    def calcular_diferencia(lista1, lista2):
#        diferencia = sum(abs(len(conj1) - len(conj2)) for conj1, conj2 in zip(lista1, lista2))
#        return diferencia

    # Definir las columnas de interés
#    columnas_interes = ['H15A_21', 'H11_21', 'H5_21', 'H6_21', 'C37_21']

    # Generar listas de conjuntos
#    lista_1 = indiscernibility(columnas_interes, nuevo_dataframe)
#    lista_2_original = indiscernibility(columnas_interes, nuevo_dataframe)

    # Obtener lista de tamaños de cada conjunto
#    tamaños_lista_1 = [len(conjunto) for conjunto in lista_1]
#    tamaños_lista_2_original = [len(conjunto) for conjunto in lista_2_original]

    # Inicializar variables para seguimiento de la lista más parecida
#    mejor_lista = lista_2_original
#    mejor_diferencia = calcular_diferencia(lista_1, lista_2_original)

    # Eliminar una por una cada columna de lista_2 y mostrar los tamaños resultantes
#    for columna1 in columnas_interes:
#        columnas_sin_columna1 = columnas_interes.copy()
#        columnas_sin_columna1.remove(columna1)
#        lista_2_sin_columna1 = indiscernibility(columnas_sin_columna1, nuevo_dataframe)
#        diferencia = calcular_diferencia(lista_1, lista_2_sin_columna1)
    
#        if diferencia < mejor_diferencia:
#            mejor_lista = lista_2_sin_columna1
#            mejor_diferencia = diferencia
    
#        # Eliminar pares de columnas de lista_2 y mostrar los tamaños resultantes
#        for columna2 in columnas_sin_columna1:
#            if columna2 != columna1:
#                columnas_sin_par = columnas_sin_columna1.copy()
#                columnas_sin_par.remove(columna2)
#                lista_2_sin_par = indiscernibility(columnas_sin_par, nuevo_dataframe)
#                diferencia = calcular_diferencia(lista_1, lista_2_sin_par)
            
#                if diferencia < mejor_diferencia:
#                    mejor_lista = lista_2_sin_par
#                    mejor_diferencia = diferencia

    # Mostrar la mejor lista encontrada en Streamlit
#    st.write("Tamaños de conjuntos en lista_1:", tamaños_lista_1)
#    st.write("Tamaños de conjuntos en la mejor lista:", [len(conjunto) for conjunto in mejor_lista])

    # Visualización con un gráfico de barras
    #fig, ax = plt.subplots()
    #labels = [f"Conjunto {i}" for i in range(len(tamaños_lista_1))]
    #x = range(len(tamaños_lista_1))
    #ax.bar(x, tamaños_lista_1, width=0.4, label='lista_1', align='center')
    #ax.bar(x, [len(conjunto) for conjunto in mejor_lista], width=0.4, label='Mejor Lista', align='edge')
    #ax.set_xlabel('Conjuntos')
    #ax.set_ylabel('Tamaños')
    #ax.set_title('Comparación de tamaños de conjuntos')
    #ax.legend()

   # st.pyplot(fig)


    st.subheader("Identificación de un reducto")

    st.markdown("""
                Un **reducto** corresponde a una lista reducidad de preguntas que puede crear la misma clasificación de pacientes que la lista completa. En esta sección se identifica un reducto para la lista de preguntas 'C37_21', 'H11_21', 'H15A_21', 'H5_21', 'H6_21'. Se realiza una clasificación en la que de forma progresiva se van quitando preguntas de la lista original y se compara la partición que crea con la que logra la lista completa. El reducto corresponde a la lista de preguntas que genere una partición lo mas parecida posible a la de la lista original.
""")


    # Calcular la diferencia absoluta total de tamaños
    def diferencia_total(lista_a, lista_b):
        min_len = min(len(lista_a), len(lista_b))
        return sum(abs(lista_a[i] - lista_b[i]) for i in range(min_len))

    # Ejemplo de listas de conjuntos
    lista_1 = indiscernibility(['H15A_21', 'H11_21', 'H5_21', 'H6_21', 'C37_21'], nuevo_dataframe)
    lista_2 = indiscernibility(['H15A_21', 'H11_21', 'H5_21', 'H6_21', 'C37_21'], nuevo_dataframe)

    # Mostrar tamaño de los conjuntos originales
#    st.write("Tamaño de lista_1:", len(lista_1))
#    st.write("Tamaño de lista_2:", len(lista_2))

    # Obtener lista de tamaños de cada conjunto
    tamaños_lista_1 = [len(conjunto) for conjunto in lista_1]
    tamaños_lista_2 = [len(conjunto) for conjunto in lista_2]

#    st.write("Tamaños de conjuntos en lista_1:", tamaños_lista_1)
#    st.write("Tamaños de conjuntos en lista_2:", tamaños_lista_2)

    # Inicializar variables para el seguimiento de la mejor coincidencia
    mejor_similitud = float('inf')
    mejor_lista = None

    # Expander para mostrar tamaños resultantes al eliminar columnas
    with st.expander("**Búsqueda del reducto**"):
        st.write("La partición creada por la lista completa se nombró como *lista_1*. La *lista_2* corresponde a copias de *lista_1* en la que se van quitando progresivamente ciertas preguntas (primero una sola, luego 2, luego 3 etc.). El **reducto** corresponde a la lista de preguntas que genere una partición igual, o lo más parecida posible a la de *lista_1*." )
        #st.write(len(lista_1))
        for columna1 in ['H15A_21', 'H11_21', 'H5_21', 'H6_21', 'C37_21']:
            columnas_sin_columna1 = ['H15A_21', 'H11_21', 'H5_21', 'H6_21', 'C37_21']
            columnas_sin_columna1.remove(columna1)
            lista_2_sin_columna1 = indiscernibility(columnas_sin_columna1, nuevo_dataframe)
            tamaños_lista_2_sin_columna1 = [len(conjunto) for conjunto in lista_2_sin_columna1]
            st.write(f"Tamaño de lista_2 sin {columna1}: {len(lista_2_sin_columna1)}")
            st.write("Tamaños de conjuntos en lista_2:", tamaños_lista_2_sin_columna1)

            # Comparar similitud
            similitud = diferencia_total(tamaños_lista_1, tamaños_lista_2_sin_columna1)
            if similitud < mejor_similitud:
                mejor_similitud = similitud
                mejor_lista = columnas_sin_columna1.copy()

            # Eliminar pares de columnas de lista_2 y mostrar los tamaños resultantes
            for columna2 in columnas_sin_columna1:
                if columna2 != columna1:
                    columnas_sin_par = columnas_sin_columna1.copy()
                    columnas_sin_par.remove(columna2)
                    lista_2_sin_par = indiscernibility(columnas_sin_par, nuevo_dataframe)
                    tamaños_lista_2_sin_par = [len(conjunto) for conjunto in lista_2_sin_par]
                    st.write(f"Tamaño de lista_2 sin {columna1} y {columna2}: {len(lista_2_sin_par)}")
                    st.write("Tamaños de conjuntos en lista_2:", tamaños_lista_2_sin_par)

                    # Comparar similitud
                    similitud = diferencia_total(tamaños_lista_1, tamaños_lista_2_sin_par)
                    if similitud < mejor_similitud:
                        mejor_similitud = similitud
                        mejor_lista = columnas_sin_par.copy()

    # Mostrar la lista de columnas que guarda mayor similitud con lista_1
    st.write("El reducto es:")
    st.write(mejor_lista)
    #st.write(len(lista_1))
    #st.write(len(lista_2))




    # Obtener los valores únicos de la columna 'nivel de riesgo'
    nivel_riesgo = nuevo_dataframe['nivel_riesgo'].unique()


#    # Crear una barra de selección múltiple para elegir niveles de riesgo
#    niveles_seleccionados = st.multiselect(
#        'En este recuadro puede seleccionar un subgrupo de pacientes que comparten el mismo nivel de riesgo',
#        nivel_riesgo
#    )

#    # Filtrar el DataFrame según los niveles de riesgo seleccionados
#    if niveles_seleccionados:
#        df_filtrado = nuevo_dataframe[nuevo_dataframe['nivel_riesgo'].isin(niveles_seleccionados)]
#        st.write(f"Filas con nivel de riesgo en {niveles_seleccionados}:")
#        st.dataframe(df_filtrado, use_container_width=True)
#    else:
#        st.write("Selecciona al menos un nivel de riesgo para visualizar las filas correspondientes.")
########################


    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    # Generar listas de conjuntos (simulando lista_2)
    lista_2 = indiscernibility(['C37_21', 'H11_21', 'H5_21', 'H6_21'], datos_limpios)
    #st.write(len(lista_2))
    # Obtener longitudes de conjuntos
    longitudes_conjuntos = [(i, len(conjunto)) for i, conjunto in enumerate(lista_2) if len(conjunto) >= 2]

    # Ordenar por longitud
    longitudes_conjuntos_ordenadas = sorted(longitudes_conjuntos, key=lambda x: x[1], reverse=True)

    # Obtener los 15 conjuntos más numerosos
    conjuntos_mas_numerosos = [conjunto for i, conjunto in enumerate(lista_2) if i in [num_conjunto for num_conjunto, _ in longitudes_conjuntos_ordenadas[: 15]]]

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
    #st.title("Visualización de Conjuntos Más Numerosos")
    st.write("Perfiles de los pacientes construidos usando el reducto")

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
    #st.title("Filtrado de DataFrame por tamaño de conjuntos")

    # Calcular el tamaño de cada conjunto y filtrar los conjuntos con menos de 30 miembros
    conjuntos_mayores_30 = [conjunto for conjunto in ind if len(conjunto) >= 30]

    # Obtener los índices de las filas que pertenecen a los conjuntos mayores o iguales a 30 miembros
    indices_filtrados = [indice for conjunto in conjuntos_mayores_30 for indice in conjunto]

    # Filtrar el DataFrame 'datos_limpios' para mantener solo las filas con índices en 'indices_filtrados'
    datos_limpios_filtrados = datos_limpios.loc[indices_filtrados]

    # Mostrar el DataFrame filtrado en Streamlit
    #st.write("DataFrame filtrado con conjuntos mayores o iguales a 30 miembros:")
    #st.dataframe(datos_limpios_filtrados)

    # Mostrar el número total de conjuntos filtrados
    #st.write(f"Número de conjuntos mayores o iguales a 30 miembros: {len(conjuntos_mayores_30)}")

    # Mostrar el tamaño de cada conjunto filtrado
    #st.write("Tamaño de cada conjunto filtrado:")
    tamaños_conjuntos = [len(conjunto) for conjunto in conjuntos_mayores_30]
    #st.write(tamaños_conjuntos)

    with st.expander("Aproximaciones"):
        X_No_indices = [set(nuevo_dataframe[nuevo_dataframe['nivel_riesgo'] == 'Sin Riesgo'].index.tolist())]
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        L=lower_approximation(R, X_No_indices)
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        upper_approximation(R,  X_No_indices)
        U=upper_approximation(R,  X_No_indices)
        U-L

        X_leve_indices = [set(nuevo_dataframe[nuevo_dataframe['nivel_riesgo'] == 'Riesgo leve'].index.tolist())]
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        L=lower_approximation(R, X_leve_indices)
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        upper_approximation(R,  X_leve_indices)
        U=upper_approximation(R,  X_leve_indices)
        U-L

        X_moderado_indices = [set(nuevo_dataframe[nuevo_dataframe['nivel_riesgo'] == 'Riesgo moderado'].index.tolist())]
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        L=lower_approximation(R, X_moderado_indices)
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        upper_approximation(R,  X_moderado_indices)
        U=upper_approximation(R,  X_moderado_indices)
        U-L

        X_severo_indices = [set(nuevo_dataframe[nuevo_dataframe['nivel_riesgo'] == 'Riesgo severo'].index.tolist())]
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        L=lower_approximation(R, X_severo_indices)
        R=indiscernibility(mejor_lista, nuevo_dataframe)
        upper_approximation(R,  X_severo_indices)
        U=upper_approximation(R,  X_severo_indices)
        U-L


#######################

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


    # Título de la aplicación
    st.subheader("Creación de un modelo de árbol de decisión")

    st.markdown("""
    En esta sección se obtiene un modelo de **arbol de decisión** que permite determinar las reglas de clasificación, a partir de las preguntas que conforman el reducto, y que produce la misma partición que la lista completa. El modelo de árbol que se obtiene permite clasificar incluso a aquellos participantes que no hayan contestado a los cinco cuestionamientos que se necesitan para asignar un nivel de riesgo de padecer sarcopenia. Además, permite establecer una gearquía sobre la importancia relativa que cada pregunta tiene para determinar un nivel de riesgo.
""")    

    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Definir las columnas de atributos
    columnas_atributos = ['C37_21', 'H11_21', 'H5_21', 'H6_21']

    # Separar los datos en atributos (X) y etiquetas (y)
    X = nuevo_dataframe[columnas_atributos]
    y = nuevo_dataframe['nivel_riesgo']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    # Crear el clasificador de árbol de decisión
    clf = DecisionTreeClassifier()

    # Entrenar el clasificador
    clf.fit(X_train, y_train)

    # Predecir las etiquetas para los datos de prueba
    y_pred = clf.predict(X_test)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)

    # Visualizar el árbol de decisión en Streamlit
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Para evitar advertencias de Streamlit
    plt.figure(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=columnas_atributos, class_names=clf.classes_)
    st.pyplot()


    with st.expander("**Métricas de evaluación de la precisión del modelo**"):
        # Mostrar la precisión del modelo en Streamlit
        st.write(f'Precisión del modelo: {accuracy:.2f}')

        # Mostrar el reporte de clasificación en Streamlit
        st.subheader("Reporte de Clasificación:")
        st.text(classification_report(y_test, y_pred))

        # Mostrar la matriz de confusión en Streamlit
        st.subheader("Matriz de Confusión:")
        st.write(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=clf.classes_, index=clf.classes_))

##################


#    # Definir las columnas de atributos
#    columnas_atributos = ['C37_21', 'H11_21', 'H5_21', 'H6_21']

#    # Separar los datos en atributos (X) y etiquetas (y)
#    X = nuevo_dataframe[columnas_atributos]
#    y = nuevo_dataframe['nivel_riesgo']

#    # Dividir los datos en conjuntos de entrenamiento y prueba
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#    # Crear el clasificador de árbol de decisión
#    clf = DecisionTreeClassifier(random_state=42)

    # Entrenar el clasificador
#    clf.fit(X_train, y_train)


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
    import seaborn as sns
    import streamlit as st
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    # Definir las columnas de atributos
    #columnas_atributos = ['C37_21', 'H11_21', 'H5_21', 'H6_21']

    # Suponiendo que 'datos_limpios' ya está definido y contiene los datos necesarios

    # Función para asignar nivel de riesgo a una fila
    def asignar_nivel_riesgo(fila, modelo, columnas_atributos):
        X = fila[columnas_atributos].values.reshape(1, -1)
        y_pred = clf.predict(X)
        return y_pred[0]

    # Aplicar la función asignar_nivel_riesgo al DataFrame datos_limpios
    #datos_limpios['Diagnóstico_árbol'] = datos_limpios.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    nuevo_dataframe['Diagnóstico_árbol'] = nuevo_dataframe.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)

    # Mostrar los resultados en Streamlit
    #st.write("Resultados de asignación de nivel de riesgo:")
    #st.dataframe(nuevo_dataframe[['C37_21', 'H11_21', 'H6_21', 'H5_21', 'Diagnóstico_árbol']], use_container_width=True)

    # Calcular el número de coincidencias y no coincidencias
    coincidencias = (nuevo_dataframe['nivel_riesgo'] == nuevo_dataframe['Diagnóstico_árbol']).sum()
    total_filas = len(nuevo_dataframe)
    no_coincidencias = total_filas - coincidencias

    # Mostrar los resultados en Streamlit
    st.write(f"Número de filas en las que coinciden los valores: {coincidencias}")
    st.write(f"Número de filas en las que no coinciden los valores: {no_coincidencias}")
 
    datos_limpios['Diagnóstico_árbol'] = datos_limpios.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    # Aplicar la función asignar_nivel_riesgo al dataframe datos_limpios
    #datos_limpios[['C37_21','H11_21', 'H6_21','H5_21','Diagnóstico_árbol']]

    #datos_filtrados.drop('H15A_21', axis=1, inplace=True) 
    datos_filtrados = datos_limpios_filtrados.dropna()
 
    datos_filtrados['Diagnóstico_árbol'] = datos_filtrados.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    datos_filtrados = datos_filtrados[['H11_21', 'H5_21', 'H6_21','C37_21','Diagnóstico_árbol']].dropna()
    #datos_filtrados[['H11_21', 'H5_21', 'H6_21','C37_21','Diagnóstico_árbol']]

    nuevo_dataframe_filtrado['Diagnóstico_árbol'] = nuevo_dataframe_filtrado.apply(asignar_nivel_riesgo, args=(clf, columnas_atributos), axis=1)
    #df_filtrado = df.dropna()
    st.dataframe(nuevo_dataframe_filtrado, use_container_width=True)

    st.write(f'La base seleccionada contiene **{datos_filtrados.shape[0]}** filas y **{datos_filtrados.shape[1]}** columnas.')


    # Botón para descargar el dataframe reducido en formato csv
    csv_data = convert_df_to_csv(nuevo_dataframe_filtrado)
    st.download_button(
        label="Descargar Dataframe en formato CSV",
        data=csv_data,
        file_name="dataframe_reducido.csv",
        mime="text/csv"
        )

    xlsx_data = convert_df_to_xlsx(nuevo_dataframe_filtrado)
    st.download_button(
        label="Descargar Dataframe en formato XLSX",
        data=xlsx_data,
        file_name="dataframe_reducido.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown("""
    El modelo de árbol, obtenido a partir del reducto, permite estimar el nivel de riesgo aún en el caso de pacientes que no hayan contestado la encuesta completa. En la Figura debajo se comparan los tamaños de los conjuntos de pacientes en cada nivel de riesgo de acuerdo a la lista completa (izquierda) con la del reducto (derecha).
""")

    # Suponiendo que 'datos_filtrados' ya está definido y contiene los datos necesarios

    # Agrupar por 'Diagnóstico_árbol' y contar el número de ocurrencias para nuevo_dataframe
    grupo_diagnostico_nuevo = nuevo_dataframe.groupby('Diagnóstico_árbol').size()

    # Agrupar por 'Diagnóstico_árbol' y contar el número de ocurrencias para datos_filtrados
    grupo_diagnostico_filtrados = nuevo_dataframe_filtrado.groupby('Diagnóstico_árbol').size()

    # Crear el panel con dos subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico para nuevo_dataframe
    sns.barplot(x=grupo_diagnostico_nuevo.values, y=grupo_diagnostico_nuevo.index, palette='Dark2', ax=axes[0])
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].set_xlabel('Número de filas')
    axes[0].set_ylabel('Diagnóstico')
    axes[0].set_title('Conteo de diagnósticos (Lista completa)')

    # Gráfico para datos_filtrados
    sns.barplot(x=grupo_diagnostico_filtrados.values, y=grupo_diagnostico_filtrados.index, palette='Dark2', ax=axes[1])
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].set_xlabel('Número de filas')
    axes[1].set_ylabel('Diagnóstico')
    axes[1].set_title('Conteo de diagnósticos (Reducto)')

    # Mostrar el panel con subplots en Streamlit
    st.pyplot(fig)

# Crear una nueva columna "Diagnóstico_árbol" en el dataframe "nuevo_dataframe"
#nuevo_dataframe['Diagnóstico_árbol'] = nuevo_dataframe.apply(asignar_nivel_riesgo, axis=1)

    #nuevo_dataframe.shape

    #datos_filtrados.shape

    #df_filtrado.shape



    # Configurar la interfaz de usuario de Streamlit
    st.title("Comparación por comorbilidad")

    st.markdown("""
    En esta sección puede compararse la distribución de niveles de riesgo para pacientes con diversas comorbilidades. En su verisón actual (01-06-2024) se comparan 4 grupos: pacientes sanos, pacientes con diabetes, pacientes con hipertensión y pacientes con diabetes e hipertensión. Para visualizar los resultados es necesario que carge los archivos de participantes de la encuesta con cada tipo de comorbilidad o sanos en los que ya se halla estimado un nivel de riesgo por sarcopenia (estos archivos se obtienen al correr los pasos previos a esta sección, usando el archivo **conjunto_de_datos_sect_a_c_d_f_e_pc_h_i_enasem_2021**.)
""")
    # Cargar hasta 4 archivos
    archivos = st.file_uploader("Cargar archivos (máximo 4 archivos, CSV o Excel)", 
                                type=["csv", "xlsx"], 
                                accept_multiple_files=True, 
                               key="archivos_uploader")

    # Verificar que no se carguen más de 4 archivos
    if len(archivos) > 4:
        st.error("Por favor, carga un máximo de 4 archivos.")
    else:
        # Inicializar una lista para almacenar los dataframes
        dfs = []

        # Procesar cada archivo
        for i, archivo in enumerate(archivos):
            if archivo.name.endswith('.csv'):
                df = pd.read_csv(archivo)
                #st.write(f"Archivo {archivo.name} cargado correctamente como df_{i+1}:")
                st.dataframe(df)
            elif archivo.name.endswith('.xlsx'):
                xls = pd.ExcelFile(archivo)
                sheet_name = xls.sheet_names[0]  # Obtener el nombre de la primera hoja
                df = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl')
                #st.write(f"Archivo {archivo.name} cargado correctamente como df_{i+1}:")
                #st.dataframe(df)
        
            # Asignar el dataframe a una variable dinámica
            globals()[f'df_{i+1}'] = df
            dfs.append(df)

        # Mostrar los nombres de los dataframes creados
        #for i in range(len(dfs)):
        #    st.write(f"df_{i+1} creado.")
        # Concatenar los dataframes verticalmente
    
    # Crear la columna 'comorbilidad' con el valor 'negativo'
    #df_1['Comorbilidad'] = 'Ninguna'
    #df_2['Comorbilidad'] = 'Diabetes'
    #df_3['Comorbilidad'] = 'Hipertensión'
    #df_4['Comorbilidad'] = 'Hipertensión y Diabetes'

    datos_concatenados = pd.concat([df_1, df_2, df_3, df_4], ignore_index=True)

    # Mostrar el dataframe resultante
    st.dataframe(datos_concatenados, use_container_width=True)

    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st


    # Contar el número de ocurrencias de cada valor en la columna "Diagnóstico_árbol"
    diagnosticos_counts = datos_concatenados['Diagnóstico_árbol'].value_counts()

    # Crear el diagrama de pastel
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(diagnosticos_counts, labels=diagnosticos_counts.index, autopct='%1.1f%%', startangle=140)
    ax.set_title('Porcentaje por evaluación de riesgo de Sarcopenia')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Agregar leyenda de texto
    plt.text(-1.3, -1.1, "Muestra de la ENASEM de 2021 de voluntarios mayores de 60 años", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)



#    # Crear figura y ejes para los subplots
#    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

#    # Iterar sobre cada nivel de riesgo y crear un gráfico de pastel
#    for i, nivel in enumerate(nivel_riesgo):
#        # Filtrar el dataframe para obtener solo las filas con el nivel de riesgo actual
#        filtro_riesgo = datos_concatenados[datos_concatenados['Diagnóstico_árbol'] == nivel]
    
#        # Contar el número de ocurrencias de cada valor en la columna "Comorbilidad"
#        comorbilidad_counts = filtro_riesgo['Comorbilidad'].value_counts()
    
#        # Crear el gráfico de pastel en el subplot correspondiente
#        ax = axs[i // 2, i % 2]
#        ax.pie(comorbilidad_counts, labels=comorbilidad_counts.index, autopct='%1.1f%%', startangle=140)
#        ax.set_title(f'Porcentaje de comorbilidades - {nivel}')
#        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

#    # Agregar leyenda de texto en el subplot inferior izquierdo
#    fig.text(0.5, 0.02, "Muestra de la ENASEM de 2021 de voluntarios mayores de 60 años", ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

#    # Ajustar diseño de subplots
#    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

#    # Mostrar el gráfico en Streamlit
#    st.pyplot(fig)





    st.title("Análisis de Comorbilidades y Diagnóstico de Sarcopenia")

    # Agrupar por "Diagnóstico árbol" y "Comorbilidad" y contar el número de filas en cada grupo
    grupo_diagnostico_comorbilidad = datos_concatenados.groupby(['Diagnóstico_árbol', 'Comorbilidad']).size().unstack(fill_value=0)

    # Calcular el porcentaje para cada subgrupo
    grupo_diagnostico_comorbilidad_porcentaje = grupo_diagnostico_comorbilidad.div(grupo_diagnostico_comorbilidad.sum(axis=1), axis=0) * 100

    # Ordenar las barras según las categorías "Sin Riesgo", "Riesgo leve", "Riesgo moderado" y "Riesgo considerable"
    orden_categorias = ["Sin Riesgo", "Riesgo leve", "Riesgo moderado", "Riesgo severo"]
    grupo_diagnostico_comorbilidad_porcentaje = grupo_diagnostico_comorbilidad_porcentaje.reindex(orden_categorias)

    # Crear el diagrama de barras con colores específicos para cada tipo de comorbilidad
    fig, ax = plt.subplots(figsize=(10, 6))
    grupo_diagnostico_comorbilidad_porcentaje.plot(kind='bar', stacked=True, ax=ax, color=['green', 'orange', 'red', 'blue'])

    # Configurar el gráfico 
    plt.title('Muestra de la ENASEM de 2021 de voluntarios mayores de 60 años')
    plt.xlabel('Diagnóstico árbol')
    plt.ylabel('Porcentaje')
    plt.xticks(rotation=45)
    plt.legend(title='Comorbilidad', bbox_to_anchor=(1.01, 0.89), loc='center left')

    # Mostrar los porcentajes numéricos dentro de las barras
    for i, (index, row) in enumerate(grupo_diagnostico_comorbilidad_porcentaje.iterrows()):
        acumulado = 0
        for j, value in enumerate(row):
            if value != 0:  # Mostrar el valor solo si no es cero
                altura_fraccion = acumulado + value / 2
                ax.text(i, altura_fraccion, f'{value:.1f}%', ha='center', va='center', color='white')
                acumulado += value

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)


##################


##################

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

