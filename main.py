import streamlit as st




import pandas as pd
import gdown
import io
import re
import numpy as np    
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from math import log1p
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px  # ✅ Faltaba este import






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
option = st.sidebar.selectbox("Seleccione una pestaña", ["Introducción", "Buscador de variables", "Relaciones de Indiscernibilidad", "Análisis por subconjunto", "Equipo de trabajo"])

if option == "Introducción":
    #
    #st.title("Analizador ENASEM-RS")
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

        st.markdown("""
<div style="text-align: justify;">

Los adultos con diabetes tienen un riesgo dos o tres veces mayor de sufrir ataques cardíacos y accidentes cerebrovasculares (<a href="https://www.niddk.nih.gov/health-information/informacion-de-la-salud/diabetes/informacion-general/prevenir-problemas/diabetes-enfermedades-cardiacas-accidentes-cerebrovasculares" target="_blank"><strong>NIDDK</strong></a>). Combinado con un flujo sanguíneo reducido, la neuropatía (daño a los nervios) en los pies aumenta la posibilidad de úlceras, infecciones y la eventual necesidad de amputación de una extremidad (<a href="https://www.imss.gob.mx/sites/all/statics/profesionalesSalud/investigacionSalud/historico/programas/16-pai-retinopatia-diabetica.pdf" target="_blank"><strong>IMSS</strong></a>). 

La retinopatía diabética es una causa importante de ceguera y se produce como resultado del daño acumulado a largo plazo en los pequeños vasos sanguíneos de la retina, afectando a cerca de 1 millón de personas en todo el mundo (<a href="https://www.imss.gob.mx/sites/all/statics/profesionalesSalud/investigacionSalud/historico/programas/16-pai-retinopatia-diabetica.pdf" target="_blank"><strong>IMSS</strong></a>). Además, la diabetes es una de las principales causas de insuficiencia renal crónica (<a href="https://www.baxter.mx/es/es/noticias-baxter/la-diabetes-entre-las-principales-causas-de-la-enfermedad-renal-cronica" target="_blank"><strong>Baxter</strong></a>). Para más información general sobre el tema, consulta el panorama de la <a href="https://www.paho.org/es/temas/diabetes" target="_blank"><strong>OPS</strong></a>.

</div>
""", unsafe_allow_html=True)

        st.subheader("Asociación entre Sarcopenia y Diabetes")

        st.markdown("""
<div style="text-align: justify;">

Se ha propuesto que la sarcopenia y la diabetes se pueden relacionar mediante múltiples mecanismos fisiopatológicos, como la resistencia a la insulina (<a href="https://scielo.isciii.es/scielo.php?script=sci_arttext&pid=S1699-695X2017000200086" target="_blank"><strong>Silva et al., 2017</strong></a>). La resistencia a la insulina se asocia con una disminución de la capacidad del cuerpo para la síntesis de proteína, favoreciendo la pérdida progresiva de masa y fuerza muscular relacionada con la sarcopenia. La diabetes tipo 2 incrementa significativamente el riesgo de desarrollar sarcopenia, con un aumento de entre dos y tres veces respecto a las personas que no padecen diabetes. Asimismo, la sarcopenia puede dificultar el control metabólico de la diabetes, debido al desequilibrio hormonal asociado con la pérdida de tejido músculo esquelético, generando un <a href="https://www.revistadiabetes.org/wp-content/uploads/9-Debes-saber-El-ciruclo-vicioso-de-diabets-y-sarcopenia-en-las-personas-de-edad-avanzada.pdf" target="_blank"><strong>círculo vicioso</strong></a> entre ambas condiciones. Factores como el <a href="https://revistasad.com/index.php/diabetes/article/view/360" target="_blank"><strong>sedentarismo</strong></a>, el control glucémico deficiente, la inflamación crónica y algunos tratamientos antidiabéticos (por ejemplo, sulfonilureas) también contribuyen a la aparición y progresión de la sarcopenia en pacientes diabéticos.

</div>
""", unsafe_allow_html=True)


               
                    
    with tab2:
            st.header("Hipertensión arterial")
        
            st.markdown("""
<div style="text-align: justify;">

La <a href="https://doi.org/10.1016/j.jacc.2017.11.006" target="_blank"><strong>hipertensión arterial</strong></a>, definida como presión arterial sistólica igual o superior a 140 mmHg o presión arterial diastólica igual o superior a 90 mmHg, es uno de los factores de riesgo más importantes para las <a href="https://www.elsevier.es/es-revista-medicina-integral-63-articulo-hipertension-arterial-riesgo-cardiovascular-10022761" target="_blank"><strong>enfermedades cardiovasculares</strong></a> y la <a href="https://doi.org/10.1001/jama.2016.19043" target="_blank"><strong>enfermedad renal crónica</strong></a>. La presión arterial es un rasgo multifacético, afectado por la <a href="https://doi.org/10.1161/CIR.0b013e31820d0793" target="_blank"><strong>nutrición</strong></a>, el medio ambiente y el comportamiento a lo largo del curso de la vida, incluida la nutrición y el crecimiento fetal y la infancia, la adiposidad, los componentes específicos de la dieta —especialmente la ingesta de sodio y potasio (<a href="https://doi.org/10.1161/CIR.0b013e31820d0793" target="_blank"><strong>Appel et al., 2011</strong></a>)—, el <a href="https://www.revespcardiol.org/es-consumo-alcohol-riesgo-hipertension-tiene-articulo-13137594" target="_blank"><strong>consumo de alcohol</strong></a> y el tabaquismo, la <a href="https://doi.org/10.1161/CIR.0b013e3181dbece1" target="_blank"><strong>contaminación del aire</strong></a>, el <a href="https://docta.ucm.es/entities/publication/7ae210b1-25b1-4420-b211-f9f76f78edd6" target="_blank"><strong>plomo</strong></a>, el <a href="https://archivosdeprevencion.eu/view_document.php?tpd=2&i=850" target="_blank"><strong>ruido</strong></a>, el <a href="https://doi.org/10.1007/s11906-009-0087-4" target="_blank"><strong>estrés psicosocial</strong></a> y el uso de medicamentos para bajar la presión arterial.

</div>
""", unsafe_allow_html=True)


            
            st.subheader("Impacto en la salud")
        
            st.markdown(""" <div style="text-align: justify;">
            La hipertensión es un trastorno médico grave que puede incrementar el riesgo de enfermedades cardiovasculares, cerebrales, renales y otras. Esta importante causa de defunción prematura en todo el mundo afecta a más de uno de cada cuatro hombres y una de cada cinco mujeres, o sea, más de 1000 millones de personas. La carga de morbilidad por hipertensión es desproporcionadamente alta en los países de ingresos bajos y medianos, en los que se registran dos terceras partes de los casos, debido en gran medida al aumento de los factores de riesgo entre esas poblaciones en los últimos decenios. 
            https://www.paho.org/es/enlace/hipertension
            """,  unsafe_allow_html=True)

            st.subheader("Hipertensión y su Asociación con la Sarcopenia")

            st.markdown("""
<div style="text-align: justify;">

La hipertensión arterial se ha asociado con la sarcopenia en adultos mayores a través de diversos mecanismos fisiopatológicos y epidemiológicos. Estudios recientes indican que la hipertensión puede contribuir a la pérdida de masa y función muscular debido a factores como la inflamación crónica, el daño vascular y la reducción del flujo sanguíneo muscular, que afectan negativamente la nutrición y el metabolismo muscular (<a href="https://www.elsevier.es/es-revista-archivos-cardiologia-mexico-293-avance-fisiopatologia-hipertension-arterial-secundaria-obesidad-S1405994017300101" target="_blank"><strong>Arch Cardiol Mex</strong></a>).

Además, ciertos tratamientos antihipertensivos, como los inhibidores de la enzima convertidora de angiotensina (IECA) y los bloqueadores de los receptores de angiotensina II (ARA II), han mostrado efectos beneficiosos en la prevención o reducción de la sarcopenia, posiblemente por mejorar la perfusión muscular y reducir la inflamación (<a href="https://iydt.wordpress.com/wp-content/uploads/2025/02/2_57_asociacion-de-terapia-antihipertensiva-y-sarcopenia-en-pacientes-adultos-mayores-de-la-umf-no.3.pdf" target="_blank"><strong>Asociación de terapia antihipertensiva y sarcopenia</strong></a>).

Un estudio observacional en adultos mayores encontró que quienes usaban IECA o ARA II tenían menor prevalencia de sarcopenia comparados con otros antihipertensivos, sugiriendo un efecto protector de estos fármacos. Asimismo, la hipertensión está frecuentemente presente como comorbilidad en pacientes con sarcopenia (<a href="https://cienciauanl.uanl.mx/?p=13231" target="_blank"><strong>Comorbilidades y riesgo de sarcopenia</strong></a>) y el sedentarismo asociado a la hipertensión también contribuye a la pérdida muscular y al aumento del riesgo de mortalidad (<a href="https://revistafac.org.ar/ojs/index.php/revistafac/article/view/361" target="_blank"><strong>Relación entre hipertensión, sedentarismo y sarcopenia</strong></a>).

</div>
""", unsafe_allow_html=True)


#elif option == "Filtrar datos":
#    st.header("Extracción de datos a partir de la ENASEM")
#    st.markdown(""" En esta sección puede cargar algunos de los conjuntos de datos de la ENASEM (ya sea de las ediciones de 2018 o de 2021). En el menú desplegable puede seleccionar el archivo a cargar. </div> """,  unsafe_allow_html=True)
#    st.write("")  # Esto agrega un espacio en blanco

#    # Menú desplegable para elegir el archivo
#    selected_file = st.selectbox("**Selecciona un archivo CSV**", list(file_urls.keys()))

#    if selected_file:
#        # Cargar el archivo seleccionado
#        data = load_csv_from_drive(file_urls[selected_file])
        
#        st.write(f"**Archivo seleccionado:** {selected_file}")
#        st.write(data)
        
#        # Lista de verificación para seleccionar columnas
#        st.markdown(""" <div style="text-align: justify;"> A continuación puede generar una base de datos a partir de las columnas que seleccione del menú desplegable. Una vez seleccionadas podrá visualizar la base de datos y descargarla en formato .csv o .xlsx al presionar cualquiera de los botones de descarga. </div> """,  unsafe_allow_html=True)
#        st.write("")  # Esto agrega un espacio en blanco
#        selected_columns = st.multiselect("**Selecciona las columnas para mostrar**", data.columns.tolist())
        
#        if selected_columns:
#            # Crear dataframe reducido
#            reduced_data = data[selected_columns]
#            st.write("")  # Esto agrega un espacio en blanco
#            st.write("**Base de datos con las columnas seleccionadas:**")
#            st.dataframe(reduced_data, use_container_width=True)

#            with st.expander("**Información adicional**"):
#                # Mostrar información del dataframe reducido
#                num_rows, num_cols = reduced_data.shape
#                st.write(f"**Número de filas**: {num_rows}")
#                st.write(f"**Número de columnas**: {num_cols}")
            
#                # Contar valores NaN por columna
#                nan_counts = reduced_data.isna().sum().reset_index()
#                nan_counts.columns = ["Clave", "Conteo"]
            
#                st.write("**Conteo de valores NaN por columna:**")
#                st.write(nan_counts)

#            # Botón para descargar el dataframe reducido en formato csv
#            csv_data = convert_df_to_csv(reduced_data)
#            st.download_button(
#                label="**Descargar Dataframe en formato CSV**",
#                data=csv_data,
#                file_name="dataframe_reducido.csv",
#                mime="text/csv"
#            )

#            xlsx_data = convert_df_to_xlsx(reduced_data)
#            st.download_button(
#                label="**Descargar Dataframe en formato XLSX**",
#                data=xlsx_data,
#                file_name="dataframe_reducido.xlsx",
#                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#            )


#    st.subheader("Unir dataframes")

#    st.markdown("""<div style="text-align: justify;"> En esta sección puede unir dos archivos .csv para formar una base de datos mas grande (recuerde seleccionar archivos que correspondan al mismo año). La base de datos se mostrará abajo, así como información sobre el conteo de filas con columnas vacías </div> """,  unsafe_allow_html=True)
#    st.write("")  # Esto agrega un espacio en blanco
#    # Seleccionar dos archivos CSV para unir
#    selected_files = st.multiselect("**Selecciona dos archivos CSV para unir**", list(file_urls.keys()), default=None, max_selections=2)

#    if len(selected_files) == 2:
#        # Cargar los dos archivos seleccionados
#        df1 = load_csv_from_drive(file_urls[selected_files[0]])
 #       df2 = load_csv_from_drive(file_urls[selected_files[1]])
        
#        if df1 is not None and df2 is not None:
#            # Unir los dataframes usando la columna 'CUNICAH'
#            merged_data = pd.merge(df1, df2, on='CUNICAH', how='inner')
            
#            st.write("**Base de datos unida**:")
#            st.dataframe(merged_data, use_container_width=True)

#            with st.expander("**Información adicional**"):
#                # Mostrar información del dataframe reducido
#                num_rows, num_cols = merged_data.shape
#                st.write(f"**Número de filas**: {num_rows}")
#                st.write(f"**Número de columnas**: {num_cols}")
            
#                # Contar valores NaN por columna
#                nan_counts = merged_data.isna().sum().reset_index()
#                nan_counts.columns = ["Clave", "Conteo"]
#            
#                st.write("**Conteo de valores NaN por columna:**")
#                st.write(nan_counts)
            

#            # Botón para descargar el dataframe reducido en formato csv
#            csv_data = convert_df_to_csv(merged_data)
#            st.download_button(
#                label="**Descargar Dataframe en formato CSV**",
#                data=csv_data,
#                file_name="dataframe_unificado.csv",
#                mime="text/csv"
#            )
            
#            # Botón para descargar el dataframe unido en formato CSV
#            csv_data = convert_df_to_csv(merged_data)
#            st.download_button(
#                label="**Descargar Dataframe unido en formato CSV**",
#                data=csv_data,
#                file_name="dataframe_unido.csv",
#                mime="text/csv"
#            )

#        st.subheader("Selección de columnas")
#        st.markdown("""<div style="text-align: justify;"> A continuación puede generar una base de datos a partir de las columnas que seleccione del menú desplegable. Una vez seleccionadas podrá visualizar la base de datos y descargarla en formato .csv o .xlsx al presionar cualquiera de los botones de descarga. </div> """,  unsafe_allow_html=True)
#    # Seleccionar dos archivos CSV para unir
#        # Lista de verificación para seleccionar columnas
#        st.write("")  # Esto agrega un espacio en blanco
#        selected_columns = st.multiselect("**Selecciona las columnas para mostrar**", merged_data.columns.tolist())
        
#        if selected_columns:
#            # Crear dataframe reducido
#            reduced_merged_data = merged_data[selected_columns]
            
#            st.write("**Base de datos:**")
#            st.dataframe(reduced_merged_data, use_container_width=True)

#            with st.expander("**Información adicional**"):
#                # Mostrar información del dataframe reducido
#                num_rows, num_cols = reduced_merged_data.shape
#                st.write(f"**Número de filas**: {num_rows}")
#                st.write(f"**Número de columnas**: {num_cols}")
            
#                # Contar valores NaN por columna
#                nan_counts = reduced_merged_data.isna().sum().reset_index()
 #               nan_counts.columns = ["Clave", "Conteo"]
            
#                st.write("**Conteo de valores NaN por columna:**")
#                st.write(nan_counts)

            
 #           # Botón para descargar el dataframe reducido en formato csv
 #           csv_data = convert_df_to_csv(reduced_merged_data)
 #           st.download_button(
 #               label="**Descargar Dataframe en formato CSV**",
 #               data=csv_data,
 #               file_name="dataframe_unificado_reducido.csv",
 #               mime="text/csv"
 #           )

 #           xlsx_data = convert_df_to_xlsx(reduced_merged_data)
 #           st.download_button(
 #               label="**Descargar Dataframe en formato XLSX**",
 #               data=xlsx_data,
 #               file_name="dataframe_unificado_reducido.xlsx",
 #               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
 #           )
elif option == "Buscador de variables":



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

elif option == "Relaciones de Indiscernibilidad":

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
    st.title("Predictor de riesgo de sarcopenia")
    st.markdown("""
    <style>
    /* Justificar todo el texto de párrafos, listas y tablas */
    p, li, td { text-align: justify; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    En esta sección se usan datos de la **Encuesta Nacional sobre Envejecimiento en México**.

    1. **Cargue el archivo** del año que desee analizar desde el botón en la barra lateral  
       (trabajar con el archivo csv que contiene las secciones: `conjunto_de_datos_sect_a_c_d_f_e_pc_h_i_enasem_20XX.csv`).  
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




    import re
    import pandas as pd
    import streamlit as st

    with st.sidebar:
        st.header("Cargar datos")
        archivo = st.file_uploader("Sube un CSV o Excel (ENASEM 2018/2021)", type=["csv", "xlsx"])

    if archivo is None:
        st.info("Sube un archivo en la barra lateral para comenzar.")
        st.stop()

    # --- Leer archivo (CSV o Excel) ---
    try:
        if archivo.name.lower().endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()

    # --- Snapshot crudo ---
    st.session_state["df_raw"] = df.copy()

    # --- Normalizar nombres ---
    # 1) limpiar espacios
    cols = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    # 2) quitar sufijos _18 o _21 SOLO al final (p. ej., C49_1_18 -> C49_1)
    cols = [re.sub(r"_(18|21)$", "", c) for c in cols]
    df.columns = cols

    # --- Quitar columnas 'Unnamed: x' ---
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:\s*\d+$")]

    # --- Resolver duplicados tras normalizar ---
    def dedup_columns(columns):
        seen = {}
        out = []
        for c in columns:
            if c not in seen:
                seen[c] = 1
                out.append(c)
            else:
                seen[c] += 1
                out.append(f"{c}__dup{seen[c]}")  # renombra duplicados
        return out

    if df.columns.duplicated().any():
        df.columns = dedup_columns(list(df.columns))
        st.info("Se detectaron columnas duplicadas tras normalizar; se renombraron con sufijo __dupN.")

    # --- Columna derivada: C67 = C67_1 + C67_2/100 (NO se borran las originales) ---
    if {"C67_1", "C67_2"}.issubset(df.columns):
        c1 = pd.to_numeric(df["C67_1"], errors="coerce")
        c2 = pd.to_numeric(df["C67_2"], errors="coerce")
        df["C67"] = c1 + (c2 / 100.0)

    # --- Agregar 'Indice' al inicio ---
    df.insert(0, "Indice", df.index)

    # --- Guardar la versión completa normalizada para todo el flujo ---
    st.session_state["df_original_norm"] = df.copy()
    st.session_state["df_original_cols"] = list(df.columns)

    # --- Construir versión reducida: datos_seleccionados ---
    columnas_deseadas_base = [
        "AGE","SEX","C4","C6","C12","C19","C22A","C26","C32","C37",
        "C49_1","C49_2","C49_8","C64","C66","C67_1","C67_2","C68E","C68G","C68H",
        "C69A","C69B","C71A","C76","H1","H4","H5","H6","H8","H9","H10","H11","H12",
        "H13","H15A","H15B","H15D","H16A","H16D","H17A","H17D","H18A","H18D","H19A","H19D"
    ]

    # Siempre incluir 'Indice' al frente
    cols_objetivo = ["Indice"] + columnas_deseadas_base

    presentes = [c for c in cols_objetivo if c in df.columns]
    faltantes = [c for c in cols_objetivo if c not in df.columns]

    datos_seleccionados = df[presentes].copy()

    if faltantes:
        st.warning("Columnas no encontradas en el archivo (se omiten en el reducido): " + ", ".join(faltantes))

#    # --- Mostrar ambos (completo y reducido) en tabs ---
#    tab1, tab2 = st.tabs(["📄 Datos completos (normalizados)", "🔎 datos_seleccionados (reducido)"])
#    with tab1:
#        st.dataframe(st.session_state["df_original_norm"], use_container_width=True)
#    with tab2:
#        st.dataframe(datos_seleccionados, use_container_width=True)

#    # (Opcional) verificar unicidad de 'Indice'
#    if df["Indice"].duplicated().any():
#        st.warning("⚠️ 'Indice' no es único en el archivo cargado. Considera crear un ID único.")

    
    
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
    #if st.session_state["df_sexo"] is not None:
    #    st.subheader("Filtrado por sexo")
    #    c1, c2 = st.columns(2)
    #    c1.metric("Filas totales", len(datos_seleccionados))
    #    c2.metric("Filas después de filtrar por sexo", len(st.session_state["df_sexo"]))
    #    #st.dataframe(st.session_state["df_sexo"].head(30), use_container_width=True)


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
        st.subheader("Seleccione el rango de edad (puede teclear los valores dentro de los recuadros).")
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
    #if st.session_state["df_filtrado"] is not None:
    #    st.subheader("Filtrado por sexo + edad")
    #    c1, c2, c3, c4 = st.columns(4)
    #    c1.metric("Filas base", len(base_df))
    #    c2.metric("Edad mínima", st.session_state["age_min"] if st.session_state["age_min"] is not None else "-")
    #    c3.metric("Edad máxima", st.session_state["age_max"] if st.session_state["age_max"] is not None else "-")
    #    #c4.metric("Filas después de filtrado", len(base_df))
    #    c4.metric("Filas después de filtrado",         len(st.session_state["df_filtrado"]))


    #st.dataframe(st.session_state["df_filtrado"].head(30), use_container_width=True)
    #st.success(f"Filtrado final: {len(st.session_state['df_filtrado']):,} filas")

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


#with st.sidebar:
#    st.subheader("Seleccione comorbilidades del grupo de pacientes a estudiar")

#    opciones_visibles = [lbl for lbl, col in comorb_map.items() if col in df_base_comorb.columns]
#
#    if not opciones_visibles:
#        st.warning("No se encontraron columnas de comorbilidades esperadas (C4, C6, C12, C19, C22A, C26, C32).")
#        df_out = df_base_comorb.copy()
#        # Agregamos columna comorbilidad (todo 'Desconocido' si no hay columnas)
#        df_out["comorbilidad"] = "Desconocido"
#        st.session_state["df_comorb"] = df_out
#        st.caption(f"Filas tras filtro de comorbilidades: {len(df_out):,}")
#    else:
#        # NUEVO: Modo de estudio
#        modo_estudio = st.radio(
#            "Modo de estudio",
#            ["Filtrar (según selección)", "Comparar (Sin vs Con)", "Todos (un solo grupo)"],
#            index=0, horizontal=False
#        )

#        # Configuración del filtro (solo relevante para 'Filtrar' y 'Comparar')
#        modo = st.radio("Lógica entre las seleccionadas", ["Todas (AND)", "Cualquiera (OR)"],
#                        index=0, horizontal=True)
#        exigir_no = st.checkbox(
#            "Exigir que las NO seleccionadas estén en 0/2",
#            value=True,
#            help="Si está activado, las comorbilidades no seleccionadas deben ser 0 o 2."
#        )
#
#        opciones_visibles_con_none = ["Sin comorbilidades"] + opciones_visibles
#        seleccion = st.multiselect(
#            "Comorbilidades (1 = Sí, 2/0 = No).",
#            options=opciones_visibles_con_none,
#            default=[],
#            help=("• ‘Sin comorbilidades’: conserva filas con TODAS las comorbilidades en 2/0.\n"
#                  "• Si seleccionas comorbilidades: combina con lógica AND/OR y decide si las NO seleccionadas deben estar en 0/2.")
#        )
#        st.session_state["comorb_selection"] = seleccion

#        # Preparar DF y asegurar numérico 0/1/2
#        df_work = df_base_comorb.copy()
#        comorb_cols_presentes = [comorb_map[lbl] for lbl in opciones_visibles]
#        for c in comorb_cols_presentes:
#            df_work[c] = pd.to_numeric(df_work[c], errors="coerce").fillna(0)

#        NO_SET = {0, 2}
#        YES_VAL = 1

#        # Máscaras base
#        mask_none = df_work[comorb_cols_presentes].isin(NO_SET).all(axis=1)   # Sin comorbilidades
#        mask_any  = (df_work[comorb_cols_presentes] == YES_VAL).any(axis=1)   # Al menos una

#        # Auxiliar para "con comorbilidades" según selección
#        def build_with_group():
#            if not seleccion or (len(seleccion) == 1 and "Sin comorbilidades" in seleccion):
#                return df_work[mask_any].copy()

#            cols_sel = [comorb_map[lbl] for lbl in seleccion if lbl in comorb_map and comorb_map[lbl] in df_work.columns]
#            cols_rest = [c for c in comorb_cols_presentes if c not in cols_sel]

#            if not cols_sel:
#                return df_work[mask_any].copy()

#            if modo.startswith("Todas"):
#                mask_sel = (df_work[cols_sel] == YES_VAL).all(axis=1)  # AND
#            else:
#                mask_sel = (df_work[cols_sel] == YES_VAL).any(axis=1)  # OR

#            if exigir_no and cols_rest:
#                mask_rest = df_work[cols_rest].isin(NO_SET).all(axis=1)
#                mask = mask_sel & mask_rest
#            else:
#                mask = mask_sel

#            return df_work[mask].copy()

#        # ----- LÓGICA POR MODO -----
#        if modo_estudio == "Todos (un solo grupo)":
#            # No filtramos filas: estudiamos todo el universo como un solo grupo
#            df_out = df_work.copy()
#            # Etiquetamos por conveniencia (no obliga a agrupar)
#            df_out["comorbilidad"] = np.where(mask_none, "Sin comorbilidades", "Con comorbilidades")

#            st.session_state["df_comorb"] = df_out
#            st.caption(f"Filas (todos los pacientes): {len(df_out):,}")

#        elif modo_estudio == "Comparar (Sin vs Con)":
#            df_none = df_work[mask_none].copy()
#            df_with = build_with_group()

#            df_none["comorbilidad"] = "Sin comorbilidades"
#            df_with["comorbilidad"] = "Con comorbilidades"

#            df_both = pd.concat([df_none, df_with], axis=0, ignore_index=True)

#            st.session_state["df_comorb"] = df_both
#            st.caption(
#                f"Sin comorbilidades: {len(df_none):,} | Con comorbilidades: {len(df_with):,} | Total: {len(df_both):,}"
#            )

#        else:  # "Filtrar (según selección)"
#            if not seleccion:
#                df_out = df_work.copy()
                # Etiqueta basada en presencia/ausencia
#                df_out["comorbilidad"] = np.where(
#                    mask_none, "Sin comorbilidades", "Con comorbilidades"
#                )

#            elif "Sin comorbilidades" in seleccion:
#                if len(seleccion) > 1:
#                    st.info("Se seleccionó 'Sin comorbilidades'. Se ignorarán otras selecciones para este filtro.")
#                df_out = df_work[mask_none].copy()
#                df_out["comorbilidad"] = "Sin comorbilidades"

#            else:
#                cols_sel = [comorb_map[lbl] for lbl in seleccion if comorb_map[lbl] in df_work.columns]
#                cols_rest = [c for c in comorb_cols_presentes if c not in cols_sel]

#                if not cols_sel:
#                    df_out = df_work.copy()
#                    df_out["comorbilidad"] = np.where(mask_none, "Sin comorbilidades", "Con comorbilidades")
#                else:
#                    if modo.startswith("Todas"):
#                        mask_sel = (df_work[cols_sel] == YES_VAL).all(axis=1)
#                    else:
#                        mask_sel = (df_work[cols_sel] == YES_VAL).any(axis=1)

#                    if exigir_no and cols_rest:
#                        mask_rest = df_work[cols_rest].isin(NO_SET).all(axis=1)
#                        mask = mask_sel & mask_rest
#                    else:
#                        mask = mask_sel

#                    df_out = df_work[mask].copy()
#                    df_out["comorbilidad"] = "Con comorbilidades"

#            st.session_state["df_comorb"] = df_out
#            st.caption(f"Filas tras filtro de comorbilidades: {len(df_out):,}")

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
            # Modo de estudio (SIN la opción "Comparar (Sin vs Con)")
            modo_estudio = st.radio(
                "Modo de estudio",
                ["Filtrar (según selección)", "Todos (un solo grupo)"],
                index=0, horizontal=False
            )

            # Configuración del filtro
            modo = st.radio("Lógica entre las seleccionadas", ["Todas (las comorbilidades seleccionadas deben aparecer simultaneamente)", "Cualquiera (los participantes tendrán al menos una de las comorbilidades seleccionadas)."],
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

            # ----- LÓGICA POR MODO -----
            if modo_estudio == "Todos (un solo grupo)":
                # No filtramos filas: estudiamos todo el universo como un solo grupo
                df_out = df_work.copy()
                # Etiquetamos por conveniencia (no obliga a agrupar)
                df_out["comorbilidad"] = np.where(mask_none, "Sin comorbilidades", "Con comorbilidades")

                st.session_state["df_comorb"] = df_out
                st.caption(f"Filas (todos los pacientes): {len(df_out):,}")

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
                            mask_sel = (df_work[cols_sel] == YES_VAL).all(axis=1)   # AND
                        else:
                            mask_sel = (df_work[cols_sel] == YES_VAL).any(axis=1)   # OR

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
        #st.subheader("Tras filtrado por sexo + edad + comorbilidades")
        # Seleccionar base segura para longitud
        base_df_for_len = st.session_state.get("df_filtrado")
        if not isinstance(base_df_for_len, pd.DataFrame) or base_df_for_len.empty:
            base_df_for_len = st.session_state.get("df_sexo")
        if not isinstance(base_df_for_len, pd.DataFrame) or base_df_for_len.empty:
            base_df_for_len = datos_seleccionados

        base_len = len(base_df_for_len)

#    c1, c2 = st.columns(2)
#    c1.metric("Filas base para filtrar.", base_len)
#    c2.metric("Filas después del filtrado", len(st.session_state["df_comorb"]))
#    st.markdown("""**A continuación se muestra la base de datos que se utilizará en el análisis.**""")
#    st.dataframe(st.session_state["df_comorb"].head(30), use_container_width=True)

#    # Resumen rápido (cuenta de 1 en cada comorbilidad seleccionada)
#    if st.session_state["comorb_selection"] and "Sin comorbilidades" not in st.session_state["comorb_selection"]:
#        with st.expander("Resumen de comorbilidades seleccionadas (conteos de 1)"):
#            df_show = st.session_state["df_comorb"]
#            for lbl in st.session_state["comorb_selection"]:
#                col = comorb_map[lbl]
#                if col in df_show.columns:
#                    cnt = int((pd.to_numeric(df_show[col], errors="coerce") == 1).sum())
#                    st.write(f"- **{lbl}**: {cnt:,} casos con valor 1")


#cols_H = [col for col in st.session_state["df_comorb"].columns if col.startswith("H")]
#st.session_state["df_comorb"][cols_H] = st.session_state["df_comorb"][cols_H].replace({6: 1, 7: 1})

    if st.session_state["df_comorb"] is not None:
    # 🔹 Convertir respuestas 6 o 7 en columnas H a 1 en la base final
    #cols_H = [col for col in st.session_state["df_comorb"].columns if col.startswith("H")]
    #if cols_H:
    #    st.session_state["df_comorb"][cols_H] = (
    #        st.session_state["df_comorb"][cols_H].replace({6: 1, 7: 1})
    #    )

        cols_H = [col for col in st.session_state["df_comorb"].columns if col.startswith("H")]
        if cols_H:
            st.session_state["df_comorb"][cols_H] = (
                st.session_state["df_comorb"][cols_H]
                .replace({6: 1, 7: 1, 8: np.nan})
            )

        #st.subheader("Tras filtrado por sexo + edad + comorbilidades")
        # Seleccionar base segura para longitud
        base_df_for_len = st.session_state.get("df_filtrado")
        if not isinstance(base_df_for_len, pd.DataFrame) or base_df_for_len.empty:
            base_df_for_len = st.session_state.get("df_sexo")
        if not isinstance(base_df_for_len, pd.DataFrame) or base_df_for_len.empty:
            base_df_for_len = datos_seleccionados

        base_len = len(base_df_for_len)

   #     c1, c2 = st.columns(2)
   #     c1.metric("Filas base para filtrar.", base_len)
   #     c2.metric("Filas después del filtrado", len(st.session_state["df_comorb"]))
   #     st.markdown("""**A continuación se muestra la base de datos que se utilizará en el análisis.**""")
   #     st.dataframe(st.session_state["df_comorb"].head(30), use_container_width=True)

   #     # Resumen rápido (cuenta de 1 en cada comorbilidad seleccionada)
   #     if st.session_state["comorb_selection"] and "Sin comorbilidades" not in st.session_state["comorb_selection"]:
   #         with st.expander("Resumen de comorbilidades seleccionadas (conteos de 1)"):
   #             df_show = st.session_state["df_comorb"]
   #             for lbl in st.session_state["comorb_selection"]:
   #                 col = comorb_map[lbl]
   #                 if col in df_show.columns:
   #                     cnt = int((pd.to_numeric(df_show[col], errors="coerce") == 1).sum())
   #                     st.write(f"- **{lbl}**: {cnt:,} casos con valor 1")


    # === Resumen compacto de filtros y base actual (en un solo expander) ===
    ss = st.session_state
    df_sexo     = ss.get("df_sexo")
    df_filtrado = ss.get("df_filtrado")
    df_comorb   = ss.get("df_comorb")
    age_min     = ss.get("age_min")
    age_max     = ss.get("age_max")
    sel_comorb  = ss.get("comorb_selection", [])

    # Helper: devuelve el primer DataFrame no vacío de la lista
    def pick_first_df(*objs):
        for o in objs:
            if isinstance(o, pd.DataFrame):
                return o
        return None

    # intenta usar base_df si existe; si no, cae a df_sexo, df_filtrado, df_comorb
    base_df_ref = pick_first_df(locals().get("base_df"), df_sexo, df_filtrado, df_comorb)
    base_len = len(base_df_ref) if isinstance(base_df_ref, pd.DataFrame) else 0

    # referencia para "Filas totales" en la sección de sexo
    datos_sel_df = locals().get("datos_seleccionados")
    if isinstance(datos_sel_df, pd.DataFrame):
        total_sexo_ref = len(datos_sel_df)
    elif isinstance(df_sexo, pd.DataFrame):
        total_sexo_ref = len(df_sexo)
    else:
        total_sexo_ref = 0

    with st.expander("📊 Resumen de filtros aplicados y muestra activa", expanded=False):

        # --- Filtrado por sexo ---
        if isinstance(df_sexo, pd.DataFrame):
            st.subheader("Filtrado por sexo")
            c1, c2 = st.columns(2)
            c1.metric("Filas totales", f"{total_sexo_ref:,}")
            c2.metric("Filas después de filtrar por sexo", f"{len(df_sexo):,}")

        # --- Filtrado por sexo + edad ---
        if isinstance(df_filtrado, pd.DataFrame):
            st.subheader("Filtrado por sexo + edad")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Filas base", f"{(len(base_df_ref) if isinstance(base_df_ref, pd.DataFrame) else 0):,}")
            c2.metric("Edad mínima", age_min if age_min is not None else "-")
            c3.metric("Edad máxima", age_max if age_max is not None else "-")
            c4.metric("Filas después de filtrado", f"{len(df_filtrado):,}")

        # --- Muestra que se usará en el análisis (tras comorbilidades) ---
        if isinstance(df_comorb, pd.DataFrame):
            st.subheader("Base final para el análisis")
            c1, c2 = st.columns(2)
            c1.metric("Filas base para filtrar", f"{base_len:,}")
            c2.metric("Filas después del filtrado", f"{len(df_comorb):,}")
            st.markdown("**A continuación se muestra la base de datos que se utilizará en el análisis.**")
            st.dataframe(df_comorb.head(30), use_container_width=True)

            # Resumen rápido de comorbilidades seleccionadas (conteos de 1)
            if sel_comorb and "Sin comorbilidades" not in sel_comorb:
                with st.expander("Resumen de comorbilidades seleccionadas (conteos de 1)", expanded=False):
                    df_show = df_comorb
                    for lbl in sel_comorb:
                        col = comorb_map.get(lbl)
                        if col in df_show.columns:
                            cnt = int((pd.to_numeric(df_show[col], errors="coerce") == 1).sum())
                            st.write(f"- **{lbl}**: {cnt:,} casos con valor 1")

    

    st.markdown("""
    ### Análisis por teoría de Rough Sets para la búsqueda de similitud entre los pacientes

    Usaremos la **teoría de conjuntos rugosos** para encontrar **grupos de pacientes** que respondieron exactamente igual 
    a un conjunto específico de preguntas.

    A estos grupos se les llama **relaciones de indiscernibilidad**, y nos permiten:

    - Comparar cómo responden distintos grupos de pacientes.
    - Detectar patrones comunes en sus respuestas.
    - Identificar el **nivel de dificultad en actividades de la vida diaria** dentro de la muestra.

    En pocas palabras: **agrupamos respuestas similares para entender mejor los perfiles y retos que enfrentan los participantes.**
    """)


# HAsta aqui el filtrado
# =========================
# Indiscernibilidad + resumen + pastel + radar (con exclusión de NaN)
# =========================


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
#df_base_ind = st.session_state.get("df_comorb")
#if not isinstance(df_base_ind, pd.DataFrame) or df_base_ind.empty:
#    df_base_ind = st.session_state.get("df_filtrado")
#if not isinstance(df_base_ind, pd.DataFrame) or df_base_ind.empty:
#    df_base_ind = st.session_state.get("df_sexo")
#if not isinstance(df_base_ind, pd.DataFrame) or df_base_ind.empty:
#    df_base_ind = datos_seleccionados.copy()

    df_base_ind = st.session_state.get("df_comorb")
    if not isinstance(df_base_ind, pd.DataFrame) or df_base_ind.empty:
        df_base_ind = st.session_state.get("df_filtrado")
    if not isinstance(df_base_ind, pd.DataFrame) or df_base_ind.empty:
        df_base_ind = st.session_state.get("df_sexo")
    if not isinstance(df_base_ind, pd.DataFrame) or df_base_ind.empty:
        df_base_ind = st.session_state.get("datos_seleccionados")  # <- en sesión
    if not isinstance(df_base_ind, pd.DataFrame) or df_base_ind.empty:
        st.warning("No hay DataFrame base disponible aún.")
        st.stop()  # o return si envuelves esta sección en una función



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
        st.subheader("Relaciones de Indiscernibilidad")
        adl_opts = st.session_state.get("ind_adl_cols", [])
        sugeridas = [c for c in ["C37","H11","H15A","H5","H6"] if c in adl_opts]
        cols_attrs = st.multiselect(
            "Atributos para agrupar (solo actividades de la vida diaria).",
            options=adl_opts,
            default=sugeridas or adl_opts[:5],
            help="Se forman clases con la combinación exacta de estas actividades."
        )
        min_size_for_pie = st.number_input(
            "Tamaño mínimo integrantes para que el subconjunto sea incluido en el gráfico de pastel",
            min_value=1, max_value=100000, value=30, step=1
        )
        top_n_radar = st.number_input(
            "Número máximo de conjuntos para mostrar",
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

            # --- Crear columna 'Subconjunto' ---
            subconjunto_map = {}
            for idx, indices in enumerate(clases):
                for fila_idx in indices:
                    subconjunto_map[fila_idx] = nombres[idx]

            df_ind["Subconjunto"] = df_ind.index.map(subconjunto_map).fillna("Fuera de subconjunto")

            
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
            st.markdown("Filas que se incluyen dentro de las **relaciones de indicernibilidad** (puede ajustarlo al definir el tamaño mínimo de participantes que debe tener una clase para incluirla y el número de conjuntos a considerar). **Solo se muestran las respuestas a las preguntas de las actividades de la vida diaría**. La última columna corresponde al nivel de riesgo de sarcopenia.")
            st.dataframe(df_eval_riesgo.reset_index(), use_container_width=True)
            st.download_button(
                "Descargar filas del pastel con nivel_riesgo (CSV)",
                data=df_eval_riesgo.reset_index().to_csv(index=False).encode("utf-8"),
                file_name="filas_pastel_con_nivel_riesgo.csv",
                mime="text/csv",
                key="dl_df_eval_riesgo"
            )
            ss["df_eval_riesgo"] = df_eval_riesgo.copy()

        with st.expander("**ℹ️¿Cómo se define el nivel de riesgo?**", expanded=False):
            st.markdown(
                """
        <div style="text-align: justify">

        Usamos **solo las Actividades de la vida diaria (AVD) seleccionadas** para la indiscernibilidad (las que se elijen en la barra lateral).  
        Antes de calcular el riesgo **excluimos** las filas que tengan valores faltantes (**NaN**) en cualquiera de esas AVD.

        **Interpretación de valores por AVD**
        - **2** → sin dificultad (estado óptimo).
        - **1** → con dificultad.
    
        ### Regla de clasificación
        Contamos cuántas AVD valen **1** (“dificultad”) y verificamos si **todas** valen **2** (“sin dificultad”):

        | Condición en las AVD seleccionadas | Nivel de riesgo |
        |---|---|
        | **Todas** valen **2** | **Riesgo nulo** |
        | **1 o 2** valen **1** | **Riesgo leve** |
        | **Exactamente 3** valen **1** | **Riesgo moderado** |
        | **4 o más** valen **1** | **Riesgo severo** |

        </div>
                """,
                unsafe_allow_html=True
            )


        # --- Radar de los N conjuntos más grandes (sobre df_eval) ---
        def determinar_color(valores):
            cnt = sum(1 for v in valores if pd.to_numeric(v, errors="coerce") == 1)
            if cnt == 0: return 'blue'
            if 1 <= cnt < 3: return 'green'
            if cnt == 3: return 'yellow'
            if 4 <= cnt < 5: return 'orange'
            return 'red'

#    st.subheader("Radar de los conjuntos más numerosos")
#    top_idxs = [i for i, _ in longitudes_orden[:int(top_n_radar)]]
#    top_sets = [(nombres[i], clases[i]) for i in top_idxs]

#    total_pacientes = len(df_eval)
#    n = int(top_n_radar)
#    cols_grid = 5
#    rows_grid = int(np.ceil(n / cols_grid))
#    fig, axs = plt.subplots(rows_grid, cols_grid, figsize=(cols_grid*6, rows_grid*5), subplot_kw=dict(polar=True))
#    axs = np.atleast_2d(axs); fig.subplots_adjust(hspace=0.8, wspace=0.6)

#    k = len(cols_attrs)
#    angulos = np.linspace(0, 2 * np.pi, k, endpoint=False).tolist()
#    angulos_cerrado = angulos + angulos[:1]

#    for idx_plot in range(rows_grid * cols_grid):
#        r = idx_plot // cols_grid; c = idx_plot % cols_grid
#        ax = axs[r, c]
#        if idx_plot >= n:
#            ax.axis('off'); continue
#        nombre, conjunto_idx = top_sets[idx_plot]
#        indices = sorted(list(conjunto_idx))
#        df_conj = df_eval.loc[indices, cols_attrs]
#        if df_conj.empty:
#            valores = [0]*k; num_filas_df = 0
#        else:
#            valores = df_conj.iloc[0].tolist(); num_filas_df = len(df_conj)
#        valores_cerrados = list(valores) + [valores[0]]
#        color = determinar_color(valores)
#        ax.plot(angulos_cerrado, valores_cerrados, color=color)
#        ax.fill(angulos_cerrado, valores_cerrados, color=color, alpha=0.25)
#        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
#        ax.set_xticks(angulos); ax.set_xticklabels(cols_attrs, fontsize=10)
#        ax.yaxis.grid(True); ax.set_ylim(0, 2)
#        ax.set_yticks([0, 1, 2]); ax.set_yticklabels([0, 1, 2], fontsize=9)
#        pct = (num_filas_df / total_pacientes * 100) if total_pacientes else 0.0
#        ax.set_title(nombre, fontsize=12)
#        ax.text(0.5, -0.2, f"Filas: {num_filas_df} ({pct:.2f}%)",
#                transform=ax.transAxes, ha="center", va="center", fontsize=10)
#    st.pyplot(fig)

#    # --- Gráfico compuesto (pastel + radares incrustados) ---
#    candidatas_idx_nom_tam = [(i, nombres[i], tam) for i, tam in longitudes_orden if tam >= min_size_for_pie]
#    if candidatas_idx_nom_tam:
#        nombres_dataframes = [nom for _, nom, _ in candidatas_idx_nom_tam]
#        tamanios = [tam for _, _, tam in candidatas_idx_nom_tam]
#        total_incluido = sum(tamanios)
#        porcentajes = [(nom, (tam/total_incluido*100.0) if total_incluido else 0.0)
#                       for _, nom, tam in candidatas_idx_nom_tam]

#        valores_dataframes, colores_dataframes = [], []
#        for idx, _, _ in candidatas_idx_nom_tam:
#            indices = sorted(list(clases[idx]))
#            sub = df_eval.loc[indices, cols_attrs]
#            vals = sub.iloc[0].tolist() if not sub.empty else [0]*len(cols_attrs)
#            valores_dataframes.append(vals)
#            colores_dataframes.append(determinar_color(vals))

#        min_radio = 1.0; max_radio = 2.40
#        radar_size_min = 0.10; radar_size_max = 0.19
#        etiquetas_radar = [et.replace('_21','').replace('_18','') for et in cols_attrs]

#        fig_comp = plt.figure(figsize=(16, 16))
#        main_ax = plt.subplot(111); main_ax.set_position([0.1, 0.1, 0.8, 0.8])

#        if porcentajes:
#            _, valores_porcentajes = zip(*porcentajes)
#            valores_porcentajes = [float(p) for p in valores_porcentajes]
#        else:
#            valores_porcentajes = []

#        colores_ajustados = colores_dataframes[:len(valores_porcentajes)]
#        wedges, texts, autotexts = main_ax.pie(
#            valores_porcentajes, colors=colores_ajustados,
#            autopct='%1.1f%%', startangle=90,
#            textprops={'fontsize': 17}, labeldistance=1.1
#        )

#        if wedges:
#            angulos_pastel = [(w.theta1 + w.theta2)/2 for w in wedges]
#            anchos = [abs(w.theta2 - w.theta1) for w in wedges]
#            max_ancho = max(anchos) if anchos else 1
#            angulos_rad = [np.deg2rad(a) for a in angulos_pastel]

#            radios_personalizados = [
#                min_radio + (1 - (log1p(a)/log1p(max_ancho))) * (max_radio - min_radio)
#                for a in anchos
#            ]
#            tamaños_radar = [
#                radar_size_min + (a/max_ancho) * (radar_size_max - radar_size_min)
#                for a in anchos
#            ]
#
#            angulos_rad_separados = angulos_rad.copy()
#            min_sep = np.deg2rad(7)
#            for i in range(1, len(angulos_rad_separados)):
#                while abs(angulos_rad_separados[i] - angulos_rad_separados[i-1]) < min_sep:
#                    angulos_rad_separados[i] += min_sep/2

#            for i, (nombre, vals, color, ang_rad, r_inset, tam_radar) in enumerate(
#                zip(nombres_dataframes, valores_dataframes, colores_dataframes,
#                    angulos_rad_separados, radios_personalizados, tamaños_radar)
#            ):
#                factor_alejamiento = 2.3
#                x = 0.5 + r_inset*np.cos(ang_rad)/factor_alejamiento
#                y = 0.5 + r_inset*np.sin(ang_rad)/factor_alejamiento
#                radar_ax = fig_comp.add_axes([x - tam_radar/2, y - tam_radar/2, tam_radar, tam_radar], polar=True)

#                vals = list(vals)[:len(cols_attrs)] or [0]*len(cols_attrs)
#                vals_c = vals + [vals[0]]
#                angs = np.linspace(0, 2*np.pi, len(cols_attrs), endpoint=False).tolist()
#                angs_c = angs + [angs[0]]

#                radar_ax.set_theta_offset(np.pi/2); radar_ax.set_theta_direction(-1)
#                radar_ax.plot(angs_c, vals_c, color=color)
#                radar_ax.fill(angs_c, vals_c, color=color, alpha=0.3)
#                radar_ax.set_xticks(angs); radar_ax.set_xticklabels(etiquetas_radar, fontsize=13)
#                radar_ax.set_yticks([0,1,2]); radar_ax.set_yticklabels(['0','1','2'], fontsize=11)
#                radar_ax.set_ylim(0,2); radar_ax.yaxis.grid(True, linestyle='dotted', linewidth=0.5)
#
#                x0 = 0.5 + 0.3*np.cos(ang_rad); y0 = 0.5 + 0.3*np.sin(ang_rad)
#                con = ConnectionPatch(
#                    xyA=(x0, y0), coordsA=fig_comp.transFigure,
#                    xyB=(x, y), coordsB=fig_comp.transFigure,
#                    color='gray', lw=0.8, linestyle='--'
#                )
#                fig_comp.add_artist(con)

#        st.pyplot(fig_comp)

        # --- Gráfico compuesto DOBLE: (A) Top-K + Otros  y  (B) Desglose de "Otros" ---
        K_MAIN  = st.sidebar.number_input("Rebanadas en pastel principal (subconjuntos mas numerosos)", 3, 20, value=12, step=1)
        K_OTROS = st.sidebar.number_input("Rebanadas máximas en pastel 'Otros' (subconjuntos minoritarios)", 5, 30, value=16, step=1)
        min_pct = st.sidebar.slider(
            "Umbral mínimo (%) para aparecer en el pastel principal",
            0.0, 10.0, value=1.0, step=0.1
        )

        # Clases candidatas para pastel (≥ umbral de tamaño)
        candidatas_idx_nom_tam = [(i, nombres[i], tam) for i, tam in longitudes_orden if tam >= min_size_for_pie]
        if not candidatas_idx_nom_tam:
            st.info(f"No hay clases con tamaño ≥ {min_size_for_pie} para el pastel.")
        else:
            total_incluido = sum(tam for _, _, tam in candidatas_idx_nom_tam) or 1

            # Armar estructura: nombre, tamaño, %, vals radar y color
            candidatas = []
            for idx, nom, tam in sorted(candidatas_idx_nom_tam, key=lambda x: x[2], reverse=True):
                indices = sorted(list(clases[idx]))
                sub = df_eval.loc[indices, cols_attrs]
                vals = sub.iloc[0].tolist() if not sub.empty else [0]*len(cols_attrs)
                col  = determinar_color(vals)
                pct  = 100.0 * tam / total_incluido
                candidatas.append({"nombre": nom, "tam": tam, "pct": pct, "vals": vals, "color": col})

            # Selección para el pastel principal: respeta umbral % y Top-K
            principales = [s for s in candidatas if s["pct"] >= min_pct][:int(K_MAIN)]
            # Si el umbral deja vacío, forzar al menos la clase más grande
            if not principales and candidatas:
                principales = candidatas[:1]

            # Resto → "Otros"
            nombres_principales = {s["nombre"] for s in principales}
            resto = [s for s in candidatas if s["nombre"] not in nombres_principales]
            tam_otros = sum(s["tam"] for s in resto)

            # ========= helper: pastel con radares alrededor =========
            def pie_con_radares(slices, titulo, agregar_otros_total=0):
                """
                slices: lista de dicts {nombre, tam, pct, vals, color}
                agregar_otros_total: si >0, añade rebanada 'Otros' sin radar con ese tamaño
                """
                etiquetas = [s["nombre"] for s in slices]
                valores   = [s["tam"]   for s in slices]
                colores   = [s["color"] for s in slices]
                if agregar_otros_total > 0:
                    etiquetas += ["Otros"]
                    valores   += [agregar_otros_total]
                    colores   += ["lightgray"]

                fig = plt.figure(figsize=(16, 16))
                main_ax = plt.subplot(111)
                main_ax.set_position([0.1, 0.1, 0.8, 0.8])

                if sum(valores) == 0:
                    main_ax.axis("off")
                    st.pyplot(fig)
                    return

                wedges, _, _ = main_ax.pie(
                    valores, labels=etiquetas, colors=colores,
                    autopct='%1.1f%%', startangle=90,
                    textprops={'fontsize': 17}, labeldistance=1.1
                )
                main_ax.set_title(titulo)

                # Radares solo para 'slices' (no para 'Otros')
                if not slices:
                    st.pyplot(fig)
                    return

                # Geometría y tamaños relativos
                angulos_pastel = [(w.theta1 + w.theta2)/2 for w in wedges[:len(slices)]]
                anchos = [abs(w.theta2 - w.theta1) for w in wedges[:len(slices)]]
                max_ancho = max(anchos) if anchos else 1
                angulos_rad = [np.deg2rad(a) for a in angulos_pastel]

                min_radio, max_radio = 1.0, 2.40
                radar_size_min, radar_size_max = 0.10, 0.19
                etiquetas_radar = [et.replace('_21','').replace('_18','') for et in cols_attrs]

                radios_personalizados = [
                    min_radio + (1 - (log1p(a)/log1p(max_ancho))) * (max_radio - min_radio)
                    for a in anchos
                ]
                tamaños_radar = [
                    radar_size_min + (a/max_ancho) * (radar_size_max - radar_size_min)
                    for a in anchos
                ]

                # Separación angular para evitar solapes
                ang_sep = angulos_rad.copy()
                min_sep = np.deg2rad(7)
                for i in range(1, len(ang_sep)):
                    while abs(ang_sep[i] - ang_sep[i-1]) < min_sep:
                        ang_sep[i] += min_sep/2

                # Dibujar radares
                k = len(cols_attrs)
                for s, ang_rad, r_inset, tam_radar in zip(slices, ang_sep, radios_personalizados, tamaños_radar):
                    x = 0.5 + r_inset*np.cos(ang_rad)/2.3
                    y = 0.5 + r_inset*np.sin(ang_rad)/2.3
                    radar_ax = fig.add_axes([x - tam_radar/2, y - tam_radar/2, tam_radar, tam_radar], polar=True)

                    vals = (s["vals"][:k] if len(s["vals"]) >= k else s["vals"] + [0]*(k-len(s["vals"])))
                    angs = np.linspace(0, 2*np.pi, k, endpoint=False).tolist()
                    vals_c = vals + [vals[0]]
                    angs_c = angs + [angs[0]]

                    radar_ax.set_theta_offset(np.pi/2); radar_ax.set_theta_direction(-1)
                    radar_ax.plot(angs_c, vals_c, color=s["color"])
                    radar_ax.fill(angs_c, vals_c, color=s["color"], alpha=0.3)
                    radar_ax.set_xticks(angs); radar_ax.set_xticklabels(etiquetas_radar, fontsize=13)
                    radar_ax.set_yticks([0,1,2]); radar_ax.set_yticklabels(['0','1','2'], fontsize=11)
                    radar_ax.set_ylim(0,2); radar_ax.yaxis.grid(True, linestyle='dotted', linewidth=0.5)

                    # Conexión pastel ↔ radar
                    x0 = 0.5 + 0.3*np.cos(ang_rad)
                    y0 = 0.5 + 0.3*np.sin(ang_rad)
                    fig.add_artist(ConnectionPatch(
                        xyA=(x0, y0), coordsA=fig.transFigure,
                        xyB=(x, y),  coordsB=fig.transFigure,
                        color='gray', lw=0.8, linestyle='--'
                    ))

                st.pyplot(fig)

            # (A) Pastel principal: Top-K (≥ umbral %) + rebanada "Otros"
            pie_con_radares(
                principales,
                "Participación por clase — Subconjuntos principales + 'Otros'",
                agregar_otros_total=tam_otros
            )

            # (B) Pastel secundario: desglose de "Otros" (limitado por K_OTROS)
            if tam_otros > 0:
                resto_view = sorted(resto, key=lambda s: s["tam"], reverse=True)[:int(K_OTROS)]
                if len(resto) > len(resto_view):
                    st.caption(
                        f"Mostrando {len(resto_view)} de {len(resto)} clases en 'Otros'. "
                        f"Ajusta K o el umbral % en la barra lateral."
                    )
                pie_con_radares(resto_view, "Desglose de 'Otros' (subconjuntos minoritarios)", agregar_otros_total=0)



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
                "**Elige un subconjunto del pastel para visualizar y correlacionar**",
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
                "Descargar subconjunto AVD (CSV)",
                data=df_sub_disp.reset_index().to_csv(index=False).encode("utf-8"),
                file_name=f"{nombres[sel_i]}_AVD.csv",
                mime="text/csv",
                key=f"dl_{sel_i}_adl"
            )


            # ---- Matriz de correlación (todas las ADL del subconjunto), sin NaN en el gráfico ----
            st.subheader("Matriz de correlación (todas las AVD del subconjunto)")

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

            with st.expander("**ℹ️ ¿Qué estoy viendo en la matriz de correlación?**"):
                st.markdown("""
        Esta matriz muestra cómo **se relacionan entre sí** las respuestas de las distintas 
        **Actividades de la Vida Diaria (AVD)** en el subconjunto seleccionado.

        **Cómo leerla:**
        - Cada fila y columna representa una AVD.
        - El valor dentro de la celda indica el **coeficiente de correlación de Pearson** entre las dos ADL correspondientes.
            - **Cercano a +1**: cuando una actividad es difícil para una persona, la otra también tiende a serlo.
            - **Cercano a -1**: cuando una es difícil, la otra suele ser fácil (relación inversa).
            - **Cercano a 0**: no hay una relación lineal clara.
        - El color de la celda refleja la fuerza y dirección de la correlación:
            - **Rojo/azul intenso** → correlación fuerte positiva/negativa.
            - **Tonos claros o gris** → correlación débil o sin datos suficientes.
        - Las celdas grises indican que **no había datos suficientes** para calcular la correlación.

        **Importante:**  
        Antes de construir la matriz se eliminan columnas con poca variación o con demasiados valores faltantes para asegurar que los coeficientes sean confiables.
        """)



# =========================
# Reductos de 4 y 3 variables (evaluación vs. partición original en el subconjunto del pastel)
# =========================
    ss = st.session_state
    need = ("ind_df_eval", "ind_cols", "ind_classes", "ind_lengths", "ind_min_size")
    if not all(k in ss for k in need) or not isinstance(ss["ind_df_eval"], pd.DataFrame):
        st.info(" ")
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

                        st.subheader("Reductos: como predecir el nivel de riesgo con menos datos de los necesarios")
                        #st.markdown("""Buscamos una lista reducida de AVD que clasifique el nivel de riesgo igual que la lista completa; para hallarla aplicamos pruebas quita-1 y quita-2 (eliminamos una o dos AVD y verificamos si las agrupaciones de pacientes se mantienen idénticas: si no cambian, se preservan las relaciones de indiscernibilidad y esa lista reducida es válida). La app usa una jerarquía de AVD (de mayor a menor utilidad) para estimar el riesgo con datos incompletos y, si la decisión queda indeterminada, sugiere qué AVD medir a continuación. Ventajas: menos tiempo y costo, tolerancia a faltantes y guía clara de recolección; límites: depende de la población de datos (conviene recalibrar) y es un apoyo clínico, no reemplaza el juicio profesional.""")
                        st.markdown("""
                        - **Objetivo:** buscar combinaciones reducidas de **4** y **3** AVD (**reductos**) que repliquen lo mejor posible la **partición original** formada con todas las AVD elegidas.
                        - **Dónde se evalúa:** solo en el **subconjunto del pastel** (clases con tamaño ≥ umbral) y **sobre filas sin NaN** en esas AVD/ADL.

                        - **Cómo se construyen y comparan:**
                          - **Estrategia quita-1 y quita-2:** eliminamos 1 o 2 AVD de la lista completa y verificamos si las **agrupaciones de pacientes** (relaciones de                 indiscernibilidad) se preservan.
                          - Cada reducto genera su partición y se compara contra la original con:
                            - **ARI** (Adjusted Rand Index) → **1.0** = particiones idénticas; mayor es mejor.
                            - **NMI** (Normalized Mutual Information) → **1.0** = información equivalente; mayor es mejor.
                            - **Preservación de pares**: porcentaje de pares de filas que el reducto **mantiene juntos / separados** igual que la partición original.

                        - **Qué se muestra:**
                          - **Tabla** ordenada por desempeño (ARI y preservaciones).
                          - Los **mejores reductos** de **4** y **3** variables.
                          - (Opcional) **Boxplot** de tamaños de bloque y **heatmap** de correspondencia.

                        - **Uso posterior (datos incompletos):**
                          - La app aplica una **jerarquía de AVD (de mayor a menor utilidad)** para **estimar el riesgo** cuando faltan datos.
                          - Si la decisión queda **indeterminada**, sugiere **qué AVD medir a continuación**.

                        - **Ventajas:** menos tiempo y costo, **tolerancia a faltantes** y **guía** clara de recolección.
                        - **Límites:** depende de la **población de datos** (conviene **recalibrar**) y es **apoyo clínico**, **no** reemplaza el juicio profesional.
                        """)


                        
                        st.caption(f"Filas en evaluación: {len(universo_sel):,} | Variables originales: {m}")
                        st.dataframe(df_closeness, use_container_width=True)

                        #st.download_button(
                        #    "Descargar métricas de reductos (CSV)",
                        #    data=df_closeness.to_csv(index=False).encode("utf-8"),
                        #    file_name="reductos_4y3_metricas.csv",
                        #    mime="text/csv",
                        #    key="dl_reductos_4y3"
                        #)

                        # ---------- mejores reductos de 4 y 3 ----------
                        best4 = df_closeness[df_closeness["#vars"] == 4].head(1)
                        best3 = df_closeness[df_closeness["#vars"] == 3].head(1)

#                    if not best4.empty:
#                        r = best4.iloc[0]
#                        st.success(
#                            f"🟩 Mejor reducto de 4 variables: **{r['Reducto']}** — "
#                            f"ARI={r['ARI']}, NMI={r['NMI']}, "
#                            f"Pres. iguales={r['Preservación iguales (%)']}%, "
#                            f"Pres. distintos={r['Preservación distintos (%)']}%"
#                        )
#                    if not best3.empty:
#                       r = best3.iloc[0]
#                        st.success(
#                            f"🟨 Mejor reducto de 3 variables: **{r['Reducto']}** — "
#                            f"ARI={r['ARI']}, NMI={r['NMI']}, "
#                            f"Pres. iguales={r['Preservación iguales (%)']}%, "
#                            f"Pres. distintos={r['Preservación distintos (%)']}%"
#                        )



                        # =========================
                        #  Heatmaps ARI: quitar 1 variable y quitar 2 variables
                        # =========================

                        def labels_from_cols(cols):
                            bloques = indiscernibility(cols, df_eval_sub)
                            return blocks_to_labels(bloques, universo_sel)

                        def ari_matrix(partitions):
                            """partitions: dict nombre -> lista_de_columnas"""
                            nombres = list(partitions.keys())
                            Y = [labels_from_cols(partitions[n]) for n in nombres]
                            k = len(nombres)
                            M = np.eye(k, dtype=float)
                            for i in range(k):
                                for j in range(i+1, k):
                                    C = contingency_from_labels(Y[i], Y[j])
                                    M[i, j] = M[j, i] = ari_from_contingency(C)
                            return nombres, M

                        def plot_heatmap(nombres, M, titulo):
                            fig, ax = plt.subplots(figsize=(min(16, 2+0.9*len(nombres)), min(16, 2+0.9*len(nombres))))
                            im = ax.imshow(M, vmin=0, vmax=1, cmap="YlGnBu")
                            for i in range(M.shape[0]):
                                for j in range(M.shape[1]):
                                    ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=8)
                            ax.set_xticks(range(len(nombres))); ax.set_xticklabels(nombres, rotation=90)
                            ax.set_yticks(range(len(nombres))); ax.set_yticklabels(nombres)
                            ax.set_title(titulo)
                            cbar = fig.colorbar(im, ax=ax)
                            cbar.set_label("ARI")
                            fig.tight_layout()
                            st.pyplot(fig)

                        # ---- Partición original + quitar 1 variable
                        parts_1 = {"Original": cols_all}
                        for c in cols_all:
                            parts_1[f"Sin {c}"] = [x for x in cols_all if x != c]

                        nombres1, M1 = ari_matrix(parts_1)
                    #plot_heatmap(nombres1, M1, "Similitud entre particiones (ARI) — quitar 1 variable")

                    # ---- Partición original + quitar 2 variables
                        parts_2 = {"Original": cols_all}
                        for a, b in combinations(cols_all, 2):
                            parts_2[f"Sin {a} y {b}"] = [x for x in cols_all if x not in (a, b)]

                        nombres2, M2 = ari_matrix(parts_2)
                        #plot_heatmap(nombres2, M2, "Similitud entre particiones (ARI) — quitar 2 variables")
                        with st.expander("ℹ️**Similitud entre particiones creadas por los reductos**"):
                            
                            plot_heatmap(nombres1, M1, "Similitud entre particiones (ARI) — quitar 1 variable")
                            plot_heatmap(nombres2, M2, "Similitud entre particiones (ARI) — quitar 2 variables")
                            st.markdown("""
                            ### Interpretación de la matriz de coincidencias (heatmap)

                            Esta matriz muestra el **índice de similitud** (por ejemplo, *Adjusted Rand Index*) entre las agrupaciones generadas:

                            1. **Diagonal principal** (valor = 1.00):  
                               Coincidencia perfecta de cada partición consigo misma. Es la línea base.

                            2. **Cruces con la columna/fila "Original"**:  
                               Comparan cada agrupación reducida (sin una o más variables) contra la agrupación generada con **todas las variables**.  
                               - Valores **altos** (cercanos a 1) → la variable eliminada no cambia mucho las agrupaciones.  
                               - Valores **bajos** → esa variable aportaba información importante para diferenciar a los participantes.

                            3. **Cruces entre reductos** (no involucrando el "Original"):  
                               Comparan dos particiones reducidas entre sí.  
                               - Valores **altos** → quitar esas variables produce agrupaciones muy similares, indicando posible **redundancia**.  
                               - Valores **medios o bajos** → las variables influyen de forma diferente y generan cambios notables en la estructura de los grupos.

                            Sobre los tonos de color:  
                            - **Más oscuro** → más similitud entre agrupaciones.  
                            - **Más claro** → menos similitud y mayor impacto de la(s) variable(s) eliminada(s) en la formación de los subconjuntos.
                            """)



                    # (Opcional) botón de descarga para cada matriz
                    #st.download_button(
                    #    "⬇️ Descargar ARI (quitar 1 var) CSV",
                    #    data=pd.DataFrame(M1, index=nombres1, columns=nombres1).to_csv().encode("utf-8"),
                    #    file_name="ari_quitar_1_variable.csv",
                    #    mime="text/csv",
                    #    key="dl_ari_1"
                    #)
                    #st.download_button(
                    #    "⬇️ Descargar ARI (quitar 2 var) CSV",
                    #    data=pd.DataFrame(M2, index=nombres2, columns=nombres2).to_csv().encode("utf-8"),
                    #    file_name="ari_quitar_2_variables.csv",
                    #    mime="text/csv",
                    #    key="dl_ari_2"
                    #)


                    
                        # ---------- Expander con gráficos opcionales ----------
                        #with st.expander("Gráficos: Boxplot de tamaños y Heatmap del mejor reducto", expanded=False):
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
                            st.markdown("""
    **Boxplot de tamaños de bloque**
    - **Qué muestra:** la distribución del **tamaño de los bloques** (nº de filas por bloque) para la partición **Original** y para cada **reducto** seleccionado.
    - **Cómo leerlo:**
      - **Línea central** = mediana; **caja** = Q1–Q3 (50% central); **bigotes** = rango hasta 1.5×IQR; **puntos** = atípicos.
      - **Mediana más baja** → bloques más pequeños (más **fragmentación**).
      - **Mediana más alta** → bloques más grandes (más **fusión** de clases).
      - **Caja ancha** → mucha **heterogeneidad** en tamaños de bloque.
    - **Interpretación práctica:** si las cajas del reducto se parecen a la original (mediana y rango), el reducto **conserva bien la granularidad** de la partición.""")

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
                            st.markdown("""
    **Heatmap de correspondencia**
    - **Qué muestra:** filas = bloques **Originales**, columnas = bloques del **Reducto**; cada celda = **cuántas filas** comparten ambos bloques.
    - **Señales clave:**
      - **Manchas intensas en diagonal** → mapeo 1-a-1 (el reducto replica bien esos bloques).
      - **Una fila original repartida en varias columnas** → el reducto **divide** ese bloque (fragmentación).
      - **Una columna del reducto que recibe de muchas filas** → el reducto **fusiona** varios bloques originales.
    - **Tips de lectura:**
      - Etiquetas: `Orig_i` (original) vs `Red_j` (reducto); están ordenados por tamaño de bloque.
      - El heatmap usa **conteos** (no proporciones); conviene mirar también el **ARI/NMI** arriba para tener una métrica global.

    **Checklist para elegir reducto**
    - **ARI** y **NMI** altos ✅.
    - Boxplot con **mediana y rango** similares a la original ✅.
    - Heatmap con **poca dispersión** fuera de la diagonal (pocas fusiones/divisiones) ✅.

    > Si tu prioridad es **no perder resolución**, prefiere reductos que **no fusionen** (evitar columnas con contribuciones de muchas filas originales).  
    > Si tu prioridad es **simplificar**, tolera algo de fusión pero evita **excesiva fragmentación** (muchas celdas pequeñas en una misma fila).
        """)


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
                    
        #                with st.expander("ℹ️ ¿Qué hace esta sección? (Resumen rápido)", expanded=False):
        #                    st.markdown("""
        #                - **Objetivo:** buscar combinaciones de **4** y **3** ADL (reductos) que repliquen lo mejor posible la **partición original** hecha con todas las ADL elegidas.
        #                - **Dónde se evalúa:** solo en el **subconjunto del pastel** (clases con tamaño ≥ umbral) y **sin NaN** en esas ADL.
        #                - **Cómo se compara:** cada reducto genera su partición y se compara contra la original con estas métricas:
        #                  - **ARI** (Adjusted Rand Index): 1.0 = particiones idénticas; mayor es mejor.
        #                  - **NMI** (Normalized Mutual Information): 1.0 = información equivalente; mayor es mejor.
        #                  - **Pres. iguales / distintos**: porcentaje de pares de filas que el reducto mantiene juntos / separados igual que la partición original.
        #                    - **Qué se muestra:**
        #                  - Una **tabla** ordenada por desempeño (ARI, preservaciones).
        #                  - Los **mejores** reductos de **4** y **3** variables.
        #                  - (Opcional) **Boxplot** de tamaños de bloque y **heatmap** de correspondencia.
        #                - **Notas:**
        #                  - Solo se consideran filas **sin NaN** en las ADL evaluadas.
        #                  - Si hay demasiadas combinaciones, se limita el número para evitar tiempos largos.
        #                  - Puedes usar las columnas del mejor reducto para entrenar modelos posteriores.
        #                    """)

    # =========================
    # Reductos + RF (rápido) + Predicción en todo el pastel + barras comparativas
    # =========================
    ss = st.session_state
    needed = ("ind_cols","ind_df","ind_classes","ind_lengths","ind_min_size")
    if not all(k in ss for k in needed):
        st.info(" ")
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
            st.session_state["df_pastel_eval"] = df_pastel_eval.copy()

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
            st.session_state["df_pastel_eval"] = df_pastel_eval.copy()  # <- después de añadir la columna

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

            #st.success("Modelos RF entrenados (best4 y, si procede, best3).")

    ####FALLBACK
    # === Fallback que acepta NaN en predicción: HistGradientBoosting ===

    ss = st.session_state

    # 1) Elegir DF preferido: df_eval_riesgo -> df_pastel_eval
    df_fb = ss.get("df_eval_riesgo")
    if not isinstance(df_fb, pd.DataFrame) or df_fb.empty:
        df_fb = ss.get("df_pastel_eval")

    if not isinstance(df_fb, pd.DataFrame) or df_fb.empty:
        st.info(" ")
    else:
        ind_cols = ss.get("ind_cols") or ss.get("ind_adl_cols") or []
        if not ind_cols:
            st.info("No encuentro 'ind_cols' para el fallback.")
        else:
            # 2) Si falta 'nivel_riesgo', lo calculamos al vuelo con la misma regla
            if "nivel_riesgo" not in df_fb.columns:
                sub = df_fb[ind_cols].apply(pd.to_numeric, errors="coerce")
                mask_ok = sub.notna().all(axis=1)
                if not mask_ok.any():
                    st.info("No puedo derivar 'nivel_riesgo' (faltan valores en todas las ADL seleccionadas).")
                else:
                    vals = sub.loc[mask_ok]
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
                    df_fb = df_fb.copy()
                    df_fb.loc[mask_ok, "nivel_riesgo"] = nivel
                    # opcional: persistir ya corregido
                    ss["df_pastel_eval"] = df_fb

            # 3) Entrenar HGB si ya tenemos 'nivel_riesgo'
            if "nivel_riesgo" in df_fb.columns:
                from sklearn.ensemble import HistGradientBoostingClassifier
                from sklearn.preprocessing import LabelEncoder

                # Verifica columnas presentes
                faltan = [c for c in ind_cols if c not in df_fb.columns]
                if faltan:
                    st.info(f"Faltan columnas en el DF para el fallback: {faltan}")
                else:
                    Xfb = df_fb[ind_cols].apply(pd.to_numeric, errors="coerce")
                    yfb_raw = df_fb["nivel_riesgo"].astype(str)

                    le_fb = LabelEncoder()
                    yfb = le_fb.fit_transform(yfb_raw.values)

                    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=42)
                    hgb.fit(Xfb, yfb)

                    ss["fb_hgb_model"] = hgb
                    ss["fb_hgb_cols"]  = ind_cols
                    ss["fb_hgb_le"]    = le_fb
                    #st.info("Modelo fallback (HGB) entrenado.")

#with st.expander("🛟 Fallback HGB: ¿cuándo se usa y por qué?", expanded=False):
#    st.markdown("""
#- Si RF no puede predecir (faltan demasiadas ADL), usamos **HistGradientBoosting**.
#- Requiere tener **`nivel_riesgo`** en el DF de entrenamiento (lo tomamos de `df_eval_riesgo` o lo **reconstruimos** con la regla).
#- Ventaja: tolera **NaN** durante **predicción** a través del pipeline (cuando imputamos/aceptamos menos observadas).
#""")

    st.subheader("Predicción de riesgo en la base completa")

    st.markdown("""
    **¿Qué hace esta parte?**  
    Calculamos el **nivel de riesgo** para **todas** las personas de la base (no solo las del gráfico de pastel). Usamos las AVD (actividades de la vida diaria) que ya seleccionaste.

    **¿Cómo se calcula?** *(en cascada, de más a menos datos disponibles)*  
    1) Usamos dos modelos ya entrenados:
       - Uno con **4 AVD** (más completo).
       - Si a alguien le faltan datos, probamos con otro de **3 AVD**.
    2) Para evitar depender solo de “rellenos”, pedimos un **mínimo de respuestas reales**:
       - Modelo de 4 AVD: necesita **≥ 3** respuestas contestadas.
       - Modelo de 3 AVD: necesita **≥ 2** respuestas contestadas.
    3) Si aun así **no alcanza la información**, marcamos ese caso como **“Sin datos”**.

    **¿Qué muestran los gráficos?**  
    - **Barras:** comparan cuántas personas quedan en **Riesgo nulo / leve / moderado / severo**:
      - con la **regla fija** (solo en quienes **no** tienen faltantes), y  
      - con la **predicción del modelo** en **toda** la base.  
    - **Pasteles:** muestran las proporciones y su nivel de riesgo de acuerdo a los registros completos (primer gráfico) y las predicciones de Random Forest (segundo gráfico).
    
    """)



    # =========================
    # Predicción en TODO el DF indiscernible (no solo el pastel)
    # =========================
    ss = st.session_state

    have4 = all(k in ss for k in ("rf_best4","rf_best4_cols","rf_best4_imp","rf_best4_le"))
    have3 = all(k in ss for k in ("rf_best3","rf_best3_cols","rf_best3_imp","rf_best3_le"))

    if "ind_df" not in ss:
        st.info(" ")
    else:

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
        #st.dataframe(df_pred_all.reset_index().head(50), use_container_width=True)
        #st.download_button(
        #    "Descargar predicciones RF (todo ind_df) CSV",
        #    data=df_pred_all.reset_index().to_csv(index=False).encode("utf-8"),
        #    file_name="predicciones_rf_todo_ind_df.csv",
        #    mime="text/csv",
        #    key="dl_pred_all_rf"
        #)

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
        .    reindex(orden + (["Sin datos"] if "Sin datos" in df_pred_all["nivel_riesgo_pred"].values else []), fill_value=0)
        )
        dist_rf_bar = dist_rf_all.reindex(orden, fill_value=0)

        x = np.arange(len(orden)); width = 0.38
        ymax = max(dist_regla.max(), dist_rf_bar.max(), 1)

        fig_b, ax_b = plt.subplots(figsize=(9, 4.8))
        b1 = ax_b.bar(x - width/2, dist_regla.values, width, label="Sin datos faltantes")
        b2 = ax_b.bar(x + width/2, dist_rf_bar.values,  width, label="Predicción de Random Forest")
        ax_b.set_xticks(x); ax_b.set_xticklabels(orden)
        ax_b.set_ylabel("Participantes")
        ax_b.set_title("Niveles de riesgo (comparación entre filas sin datos faltantes y predicción usando RF)")
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
            ax1.axis('equal'); ax1.set_title("Proporción de participantes y su nivel de riesgo (usando solo filas sin datos faltantes)")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(6.5, 6.5))
        v2 = dist_rf_all.values
        if v2.sum() == 0:
            ax2.text(0.5, 0.5, "No hay predicciones RF disponibles",
                     ha="center", va="center", fontsize=12)
            ax2.axis("off")
        else:
            ax2.pie(v2, labels=dist_rf_all.index, autopct=lambda p: f"{p:.1f}%", startangle=120)
            ax2.axis('equal'); ax2.set_title("Proporción de participantes y su nivel de riesgo (con datos imputados usando Random Forest)")
        st.pyplot(fig2)




    
    # =========================
    # Enriquecer predicciones con columnas del DF original
    # =========================
    st.subheader("Predicciones + columnas originales (enriquecido)")
    st.markdown("""A continuación se muestra la base de datos con todas las columnas (además de dos columnas adicionales: una que muestra si pertenecieron a un subconjunto de las relaciones de indiscernibilidad y otra con el nivel de riesgo predicho por los modelos de Random Forest.""")
    ss = st.session_state

    # Asegurar que df_pred_all exista (viene del bloque anterior)
    if "df_pred_all_rf" in ss:
        df_pred_all = ss["df_pred_all_rf"].copy()
    else:
        # Si no está en sesión, tomamos el df_pred_all de la variable local (si existe)
        try:
            df_pred_all = df_pred_all.copy()
        except NameError:
            st.warning("No encuentro el DataFrame de predicciones en memoria.")
            df_pred_all = None

    # DF original normalizado (con 'Indice'), guardado al cargar datos
    df_orig = ss.get("df_original_norm")

    if isinstance(df_pred_all, pd.DataFrame) and isinstance(df_orig, pd.DataFrame):
        # Trabajar con 'Indice' como columna para el merge
        df_pred_all_reset = df_pred_all.reset_index()  # trae 'Indice' si era índice
        if "Indice" not in df_pred_all_reset.columns:
            # En caso raro de que el índice no se llame 'Indice', forzamos el nombre
            idx_name = df_pred_all.index.name or "Indice"
            df_pred_all_reset = df_pred_all.reset_index().rename(columns={idx_name: "Indice"})

        if "Indice" not in df_orig.columns:
            st.warning("El DF original no tiene la columna 'Indice'; no se puede enriquecer.")
            df_enriquecido = df_pred_all_reset.copy()
        else:
            # Solo columnas que NO estén ya en df_pred_all_reset
            base_cols = list(df_pred_all_reset.columns)
            cols_extra = [c for c in df_orig.columns if c not in base_cols]

            if cols_extra:
                df_enriquecido = df_pred_all_reset.merge(
                    df_orig[["Indice"] + cols_extra],
                    on="Indice", how="left"
                )
            else:
                df_enriquecido = df_pred_all_reset.copy()

            # Opcional: ordenar para que 'Indice' y 'nivel_riesgo_pred' queden al inicio
            first = [c for c in ["Indice", "nivel_riesgo_pred"] if c in df_enriquecido.columns]
            rest  = [c for c in df_enriquecido.columns if c not in first]
            df_enriquecido = df_enriquecido[first + rest]

        st.dataframe(df_enriquecido, use_container_width=True)
        st.download_button(
            "Descargar predicciones enriquecidas (CSV)",
            data=df_enriquecido.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_rf_enriquecidas.csv",
            mime="text/csv",
            key="dl_pred_all_rf_enriched"
        )
    else:
        st.info("Aún no hay predicciones o no se ha guardado el DF original normalizado (‘df_original_norm’).")


    
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

elif option == "Análisis por subconjunto":
    import unicodedata
    import re  # ✅ Se usa en norm()
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import io
    import streamlit as st

    # -------------------------------
    # Utilidades
    # -------------------------------
    def norm(s: str) -> str:
        """Normaliza nombres de columnas: minúsculas, sin acentos, sin espacios extras."""
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"\s+", "_", s.strip().lower())
        return s

    def find_candidate_columns(df: pd.DataFrame, keyword_groups):
        """
        Devuelve columnas candidatas cuyas versiones normalizadas contienen
        TODOS los keywords en algún grupo. keyword_groups: lista de listas de palabras, p.ej.
        [['nivel','riesgo'], ['diagnostico','arbol']]
        """
        cols = []
        norm_map = {c: norm(c) for c in df.columns}
        for c, nc in norm_map.items():
            for group in keyword_groups:
                if all(k in nc for k in group):
                    cols.append(c)
                    break
        # mantener orden original y quitar duplicados
        seen = set(); out = []
        for c in cols:
            if c not in seen:
                seen.add(c); out.append(c)
        return out

    @st.cache_data(show_spinner=False)
    def read_any_csv(uploaded_file) -> pd.DataFrame:
        return pd.read_csv(uploaded_file, sep=None, engine="python")

    # -------------------------------
    # Sección: Cargar archivo
    # -------------------------------
    st.header("Cargar y filtrar por Nivel de riesgo o Subconjunto")

    # === Debajo de fig_bar ===
    st.markdown("### Diccionario de variables mostradas")

    # 1) Menú para seleccionar año
    anio_dicc = st.selectbox("Selecciona el año del diccionario", [2018, 2021], index=0)

    
    uploaded = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if not uploaded:
        st.info("Carga un .csv para comenzar.")
        st.stop()

    # Leer CSV
    try:
        df = read_any_csv(uploaded)
    except Exception as e:
        st.error(f"No se pudo leer el CSV: {e}")
        st.stop()

    # Limpiar columnas tipo Unnamed:
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:\s*\d+")].copy()

    st.success(f"Archivo cargado: {uploaded.name}")
    st.caption(f"{df.shape[0]} filas × {df.shape[1]} columnas")

    # -------------------------------
    # Detectar columnas candidatas
    # -------------------------------
    # Riesgo: columnas con "nivel"+"riesgo" o "diagnostico"+"arbol"
    riesgo_candidates = find_candidate_columns(
        df,
        keyword_groups=[
            ["nivel","riesgo"],
            ["diagnostico","arbol"],   # p.ej. "Diagnóstico_árbol"
            ["riesgo"]                 # fallback laxa
        ]
    )

    # Subconjunto: "subconjunto", "conjunto", "cluster", "bloque", "equivalencia"
    subconj_candidates = find_candidate_columns(
        df,
        keyword_groups=[
            ["subconjunto"],
            ["conjunto"],
            ["cluster"],
            ["bloque"],
            ["equivalencia"]
        ]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Columnas candidatas a “Nivel de riesgo”**")
        if riesgo_candidates:
            st.code(", ".join(riesgo_candidates), language="text")
        else:
            st.warning("No se detectaron columnas de riesgo automáticamente.")

    with col2:
        st.write("**Columnas candidatas a “Subconjunto”**")
        if subconj_candidates:
            st.code(", ".join(subconj_candidates), language="text")
        else:
            st.warning("No se detectaron columnas de subconjunto automáticamente.")

    # -------------------------------
    # Controles de filtrado
    # -------------------------------
    modo = st.radio(
        "¿Cómo deseas filtrar?",
        options=["Por nivel de riesgo", "Por subconjunto"],
        index=0 if riesgo_candidates else 1 if subconj_candidates else 0,
        horizontal=True
    )

    if modo == "Por nivel de riesgo":
        if not riesgo_candidates:
            st.error("No hay columnas candidatas de riesgo. Cambia a 'Por subconjunto' o revisa nombres de columnas.")
            st.stop()
        col_riesgo = st.selectbox("Elige la columna de riesgo:", riesgo_candidates)
        valores = sorted(df[col_riesgo].dropna().unique().tolist())
        sel = st.multiselect("Selecciona uno o más niveles de riesgo:", valores, default=valores[:1] if valores else [])
        if not sel:
            st.info("Selecciona al menos un valor para visualizar el filtrado.")
            st.stop()
        df_filtrado = df[df[col_riesgo].isin(sel)].copy()

    else:  # Por subconjunto
        if not subconj_candidates:
            st.error("No hay columnas candidatas de subconjunto. Cambia a 'Por nivel de riesgo' o revisa nombres de columnas.")
            st.stop()
        col_sub = st.selectbox("Elige la columna de subconjunto:", subconj_candidates)
        valores = sorted(df[col_sub].dropna().unique().tolist())
        sel = st.multiselect("Selecciona uno o más subconjuntos:", valores, default=valores[:1] if valores else [])
        if not sel:
            st.info("Selecciona al menos un valor para visualizar el filtrado.")
            st.stop()
        df_filtrado = df[df[col_sub].isin(sel)].copy()

    # ✅ Guarda el filtrado para la vista por secciones
    st.session_state["df_filtrado"] = df_filtrado

    # -------------------------------
    # Resultados
    # -------------------------------
    st.markdown("### Vista de filas filtradas")
    st.caption(f"Filas seleccionadas: {df_filtrado.shape[0]} de {df.shape[0]}")
    st.dataframe(df_filtrado, use_container_width=True)

    # Conteo por categoría elegida (útil para verificar)
    with st.expander("Ver conteo por categoría seleccionada"):
        if modo == "Por nivel de riesgo":
            st.write(df_filtrado[col_riesgo].value_counts(dropna=False))
        else:
            st.write(df_filtrado[col_sub].value_counts(dropna=False))

    # Descargar CSV filtrado
    csv_bytes = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar filas filtradas (CSV)",
        data=csv_bytes,
        file_name="filtrado.csv",
        mime="text/csv"
    )

    # =======================
    # Vista por secciones (después del filtrado)
    # =======================

    # 1) Usar directamente df_filtrado desde session_state
    df_base = st.session_state.get("df_filtrado")

    if not isinstance(df_base, pd.DataFrame) or df_base.empty:
        st.info("No hay datos disponibles para mostrar por secciones aún.")
    else:
        st.subheader("Vista por secciones de columnas")

        # 2) Definir secciones -> listas de columnas
        SECCIONES = {
            "1. Datos demográficos básicos": [
                "AGE", "SEX"
            ],
            "2. Diagnóstico y antecedentes médicos generales": [
                "C4", "C6", "C12", "C19", "C22A", "C26", "C32", "C37"
            ],
            "3. Bienestar psicológico y salud mental": [
                "C49_1", "C49_2", "C49_8", "C76"
            ],
            "4. Autopercepción de salud y comparación temporal": [
                "C64"
            ],
            "5. Datos antropométricos y condición física": [
                "C66", "C67_1", "C67_2", "C67",
                "C68E", "C68G", "C68H", "C69A", "C69B", "C71A"
            ],
            "6. Actividades de la vida diaria (AVD)": [
                "H1","H4","H5","H6","H8","H9","H10","H11","H12","H13",
                "H15A","H15B","H15D","H16A","H16D","H17A","H17D","H18A","H18D","H19A","H19D"
            ],
        }

        # 3) Selector de sección (la “barra para seleccionar categoría”)
        seccion = st.selectbox("Selecciona una sección para mostrar:", list(SECCIONES.keys()))

        # 4) Columnas base a mostrar (intersecta con columnas presentes)
        cols_obj = SECCIONES[seccion]
        cols_presentes = [c for c in cols_obj if c in df_base.columns]

        # Anteponer Indice y nivel_riesgo si existen
        encabezado = [c for c in ["Indice", "nivel_riesgo", "nivel_riesgo_pred"] if c in df_base.columns]
        cols_finales = encabezado + [c for c in cols_presentes if c not in encabezado]

        # 5) (Opcional) permitir agregar columnas extra
        with st.expander("➕ Agregar columnas adicionales a la vista", expanded=False):
            otras = st.multiselect(
                "Columnas extra:",
                options=[c for c in df_base.columns if c not in cols_finales],
                default=[]
            )
            cols_finales = cols_finales + otras

        # 6) Mostrar tabla
        if not cols_finales:
            st.warning("Ninguna de las columnas de la sección está presente en el DataFrame.")
        else:
            st.dataframe(df_base[cols_finales], use_container_width=True, height=520)

            # 7) Botón de descarga de la vista
            st.download_button(
                f"Descargar vista: {seccion} (CSV)",
                data=df_base[cols_finales].to_csv(index=False).encode("utf-8"),
                file_name=f"vista_{seccion.split('.')[0]}_{seccion.split('. ',1)[-1].replace(' ','_')}.csv",
                mime="text/csv",
                key=f"dl_vista_{seccion}"
            )

    # ========= 1) Barras apiladas: proporciones por pregunta (solo preguntas discretas) =========
    if not cols_presentes:
        st.warning("No hay columnas de la sección presentes para generar resúmenes.")
    else:
        st.subheader("Proporción de respuestas por pregunta (solo variables discretas)")

        # 1) Normalización básica a numérico y mapeo Sí/No
        df_prop = (
            df_base[cols_presentes]
            .replace({'Sí': 2, 'Si': 2, 'NO': 1, 'No': 1})
            .apply(pd.to_numeric, errors='coerce')
        )

        # 2) Detectar columnas discretas (excluir continuas como edad/estatura)
        DISCRETE_MAX_UNIQUE = 10  # ajusta si quieres ser más/menos estricto
        cols_discretas = []
        for c in df_prop.columns:
            serie = df_prop[c].dropna()
            if serie.empty:
                continue
            nun = serie.nunique(dropna=True)
            # Discreta si pocos valores distintos o si solo hay 1/2 (o 0/1/2)
            es_12 = serie.isin([1, 2]).all()
            es_012 = serie.isin([0, 1, 2]).all()
            if nun <= DISCRETE_MAX_UNIQUE or es_12 or es_012:
                cols_discretas.append(c)

        if not cols_discretas:
            st.info("No se detectaron variables discretas en la selección (p. ej., solo continuas como edad/estatura).")
        else:
            # 3) Proporciones por pregunta usando SOLO discretas (evita error de reset_index)
            long = df_prop[cols_discretas].melt(var_name="Pregunta", value_name="Respuesta")
            long["Respuesta_cat"] = pd.Categorical(
                np.where(long["Respuesta"].isna(), "NaN", long["Respuesta"].astype("Int64").astype(str)),
                categories=["1", "2", "NaN"],
                ordered=True
            )
            prop = (
                long.groupby("Pregunta")["Respuesta_cat"]
                    .value_counts(normalize=True)
                    .rename("Porcentaje")
                    .mul(100)
                    .reset_index()
            )

            fig_bar = px.bar(
                prop,
                x="Pregunta",
                y="Porcentaje",
                color="Respuesta_cat",
                barmode="stack",
                text=prop["Porcentaje"].round(1).astype(str) + "%",
                category_orders={"Respuesta_cat": ["1", "2", "NaN"]}
            )
            fig_bar.update_layout(yaxis_title="Porcentaje", xaxis_title=None, legend_title="Respuesta", bargap=0.25)
            st.plotly_chart(fig_bar, use_container_width=True)


            # --- Tabla de código y descripción para las variables del gráfico ---

            import re, unicodedata
            import pandas as pd
            import streamlit as st

            # Variables presentes en el gráfico (eje X)
            vars_en_grafico = prop["Pregunta"].unique().tolist()

            #st.markdown("#### Diccionario de variables")
            #anio_dic = st.selectbox(
            #    "Selecciona el año del diccionario:",
            #    [2018, 2021],
            #    index=0
            #)

            # URLs RAW en GitHub
            urls_dic = {
                2018: "https://raw.githubusercontent.com/SArcD/ENASEM/main/diccionario_datos_sect_a_c_d_f_e_pc_h_i_enasem_2018.csv",
                2021: "https://raw.githubusercontent.com/SArcD/ENASEM/main/diccionario_datos_sect_a_c_d_e_pc_f_h_i_2021_enasem_2021.csv",
            }

            # Helpers de normalización
            def norm_text(s: str) -> str:
                s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
                s = s.strip()
                s = re.sub(r"\s+", " ", s)
                return s

            def norm_colname(s: str) -> str:
                # para nombres de columnas del diccionario
                s = norm_text(s).lower()
                s = s.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
                s = s.replace("descripcion","descripcion")  # por si viniera con acento
                s = s.replace("código","codigo")
                s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
                return s

            def norm_codigo_para_join(s: str) -> str:
                # normaliza códigos de variables (diccionario y tus columnas)
                s = norm_text(s)
                s = s.replace(".", "_")           # C49.1 -> C49_1
                s = re.sub(r"_(18|21)$", "", s)   # quita sufijos de año
                s = re.sub(r"__+", "_", s)
                return s.lower()

            # Lee diccionario
            try:
                dic = pd.read_csv(urls_dic[anio_dic])
            except Exception as e:
                st.error(f"No se pudo leer el diccionario del {anio_dic}: {e}")
                dic = pd.DataFrame()

            tabla_dic = pd.DataFrame({"Variable": vars_en_grafico})
            tabla_dic["var_norm_join"] = tabla_dic["Variable"].apply(norm_codigo_para_join)

            if not dic.empty:
                # Normaliza encabezados del diccionario
                dic.columns = [norm_colname(c) for c in dic.columns]

                # Detecta columna de código y de descripción de manera flexible
                candidatos_codigo = [c for c in dic.columns if any(
                    key in c for key in ["codigo","variable","var","clave"]        
                )]
                candidatos_desc = [c for c in dic.columns if any(
                    key in c for key in ["descripcion","descripcion_de_la_variable","label","pregunta","texto","definicion","enunciado"]
                )]

                cod_col = candidatos_codigo[0] if candidatos_codigo else None
                desc_col = candidatos_desc[0] if candidatos_desc else None

                if not cod_col:
                    st.warning("No se encontró columna de **código** en el diccionario (busqué: código/variable/var/clave).")
                if not desc_col:
                    st.warning("No se encontró columna de **descripción** en el diccionario (busqué: descripción/label/pregunta/texto/definición).")

                if cod_col and desc_col:
                    dic["_join_code"] = dic[cod_col].astype(str).apply(norm_codigo_para_join)
                    dic_mini = dic[["_join_code", desc_col]].rename(columns={desc_col: "Descripción"})
                    # merge
                    tabla_dic = tabla_dic.merge(dic_mini, left_on="var_norm_join", right_on="_join_code", how="left")
                    tabla_dic = tabla_dic.drop(columns=["var_norm_join","_join_code"])
                else:
                    # sin columnas detectadas, muestra solo Variable
                    tabla_dic = tabla_dic.drop(columns=["var_norm_join"])

            # Muestra tabla
            st.dataframe(tabla_dic.rename(columns={"Variable":"Código"}), use_container_width=True)

            # Descarga
            st.download_button(
                "⬇️ Descargar tabla de códigos y descripciones (CSV)",
                data=tabla_dic.rename(columns={"Variable":"Codigo"}).to_csv(index=False).encode("utf-8"),
                file_name=f"codigos_descripciones_{anio_dic}.csv",
                mime="text/csv"
            )



            
            # ========= 2) Patrones idénticos (bloques de indiscernibilidad) =========
            st.subheader("Patrones de respuesta idéntica (bloques)")
            top_n = st.slider("Mostrar los TOP patrones (por frecuencia)", 5, 50, 15, 5)

            df_pat = df_prop[cols_discretas].copy()
            # Construir clave de patrón como tupla de valores por columna discreta (incluye NaN)
            pat_keys = df_pat.apply(lambda r: tuple(r[c] for c in cols_discretas), axis=1)
            counts = pat_keys.value_counts(dropna=False)
            total = counts.sum()

            if total == 0:
                st.info("No hay filas válidas para construir patrones con las variables discretas seleccionadas.")
            else:
                top_counts = counts.head(top_n).reset_index()
                top_counts.columns = ["Patrón", "Frecuencia"]
                top_counts["Porcentaje"] = 100 * top_counts["Frecuencia"] / total

                def pattern_to_str(p):
                    def fmt(v):
                        if pd.isna(v): return "NaN"
                        try: return str(int(v))
                        except: return str(v)
                    return " | ".join(f"{c}:{fmt(v)}" for c, v in zip(cols_discretas, p))

                top_counts["Patrón (pregunta:valor)"] = top_counts["Patrón"].apply(pattern_to_str)
                st.caption(f"Patrones únicos en total (solo discretas): {counts.shape[0]:,}")
                st.dataframe(
                    top_counts[["Patrón (pregunta:valor)","Frecuencia","Porcentaje"]]
                        .assign(Porcentaje=lambda d: d["Porcentaje"].round(2)),
                    use_container_width=True, height=400
                )

                # Mini KPI: cobertura del TOP-N
                cobertura = top_counts["Frecuencia"].sum() / total * 100
                st.metric("Cobertura del TOP mostrado", f"{cobertura:.1f}%")

                # ========= 3) Treemap de patrones (tamaño = frecuencia; color = % de '2') =========
                st.subheader("Treemap de patrones (tamaño = frecuencia, color = % de '2')")
                def pct_twos(p):
                    vals = [x for x in p if not pd.isna(x)]
                    return 100 * (np.sum(np.array(vals) == 2) / len(vals)) if vals else 0.0

                treemap_df = top_counts.copy()
                treemap_df["%2_en_patron"] = treemap_df["Patrón"].apply(pct_twos)
                treemap_df["Patrón (corto)"] = treemap_df["Patrón"].apply(
                    lambda p: " / ".join(str(int(x)) if pd.notna(x) else "NaN" for x in p)
                )

                fig_tree = px.treemap(
                    treemap_df,
                    path=["Patrón (corto)"],
                    values="Frecuencia",
                    color="%2_en_patron",
                    color_continuous_scale="Greens",
                    hover_data={"Porcentaje":":.2f","%2_en_patron":":.1f"}
                )
                fig_tree.update_layout(margin=dict(t=30,l=0,r=0,b=0), coloraxis_colorbar=dict(title="% de '2'"))
                st.plotly_chart(fig_tree, use_container_width=True)

                # ========= Descargas =========
                st.download_button(
                    "⬇️ Descargar TOP patrones (CSV)",
                    data=top_counts[["Patrón (pregunta:valor)","Frecuencia","Porcentaje"]]
                        .to_csv(index=False).encode("utf-8"),
                    file_name="top_patrones_identicos.csv",
                    mime="text/csv"
                )





else:
       st.subheader("Equipo de Trabajo")

       # Información del equipo
       equipo = [{
               "nombre": "Dr. Santiago Arceo Díaz",
               "foto": "ArceoS.jpg",
               "reseña": "Licenciado en Física, Maestro en Física y Doctor en Ciencias (Astrofísica). Posdoctorante de la Universidad de Colima y profesor del Tecnológico Nacional de México Campus Colima. Cuenta con el perfil deseable, pertenece al núcleo académico y es colaborador del cuerpo académico Tecnologías Emergentes y Desarrollo Web de la Maestría Sistemas Computacionales. Ha dirigido tesis de la Maestría en Sistemas Computacionales y en la Maestría en Arquitectura Sostenible y Gestión Urbana.",
               "CV": "https://scholar.google.com.mx/citations?user=3xPPTLoAAAAJ&hl=es", "contacto": "santiagoarceodiaz@gmail.com"},
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
               "nombre": "Dra. Elena Elsa Bricio Barrios",
               "foto": "BricioE.jpg",
               "reseña": "Química Metalúrgica, Maestra en Ciencias en Ingeniería Química y doctorante en Ingeniería Química. Actualmente es profesora del Tecnológico Nacional de México Campus Colima. Cuenta con el perfil deseable, es miembro del cuerpo académico Tecnologías Emergentes y Desarrollo Web y ha codirigido tesis de la Maestría en Sistemas Computacionales.",
               "CV": "https://scholar.google.com.mx/citations?hl=es&user=TGZGewEAAAAJ", "contacto": "elena.bricio@colima.tecnm.mx"},
               {
               "nombre": "Dra. Mónica Ríos Silva",
               "foto": "rios.jpg",
               "reseña": "Médica cirujana y partera con especialidad en Medicina Interna y Doctorado en Ciencias Médicas por la Universidad de Colima, médica especialista del Hospital Materno Infantil de Colima y PTC de la Facultad de Medicina de la Universidad de Colima. Es profesora de los posgrados en Ciencias Médicas, Ciencias Fisiológicas, Nutrición clínica y Ciencia ambiental global.",
               "CV": "https://scholar.google.com.mx/scholar?hl=en&as_sdt=0%2C5&q=Monica+Rios+silva&btnG=", "contacto": "mrios@ucol.mx"},
               {
               "nombre": "Dra. Rosa Yolitzy Cárdenas María",  
               "foto": "cardenas.jpg",
               "reseña": "Ha realizado los estudios de Química Farmacéutica Bióloga, Maestría en Ciencias Médicas y Doctorado en Ciencias Médicas, todos otorgados por la Universidad de Colima. Actualmente, se desempeña como Técnica Académica Titular C en el Centro Universitario de Investigaciones Biomédicas de la Universidad de Colima, enfocándose en la investigación básica y clínica de enfermedades crónico-degenerativas no transmisibles en investigación. También es profesora en la Maestría y Doctorado en Ciencias Médicas, así como en la Maestría en Nutrición Clínica de la misma universidad. Es miembro del Sistema Nacional de Investigadores nivel I y miembro fundador activo de la asociación civil DAYIN (https://www.dayinac.org/)",
               "CV": "https://scholar.google.com.mx/scholar?hl=en&as_sdt=0%2C5&q=rosa+yolitzy+c%C3%A1rdenas-mar%C3%ADa&btnG=&oq=rosa+yoli", "contacto": "rosa_cardenas@ucol.mx"}
               ]

       # Establecer la altura deseada para las imágenes
       altura_imagen = 150  # Cambia este valor según tus preferencias

       # Mostrar información de cada miembro del equipo
       for miembro in equipo:
           st.subheader(miembro["nombre"])
           img = st.image(miembro["foto"], caption=f"Foto de {miembro['nombre']}", use_container_width=False, width=altura_imagen)
           st.write(f"Correo electrónico: {miembro['contacto']}")
           st.write(f"Reseña profesional: {miembro['reseña']}")
           st.write(f"CV: {miembro['CV']}")

       # Información de contacto
       st.subheader("Información de Contacto")
       st.write("Si deseas ponerte en contacto con nuestro equipo, puedes enviar un correo a santiagoarceodiaz@gmail.com")
