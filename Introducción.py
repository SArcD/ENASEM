import streamlit as st
import pandas as pd
import gdown

# Enlaces a los archivos CSV en Google Drive
file_urls = {
    "Archivo 1": "https://drive.google.com/uc?id=1OXrglgbqvwA1Oa2aMB5iLh9bMLJNo-uu",
    "Archivo 2": "https://drive.google.com/uc?id=1pn8-1nCeVb8piMgad-7foAI9z1nmfqsO"
}

# Función para descargar y cargar un archivo CSV desde Google Drive
def load_csv_from_drive(url):
    output = "temp.csv"
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# Función para convertir el dataframe a xlsx y crear un enlace de descarga
def convert_df_to_xlsx(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


# Interfaz de Streamlit
#st.title("ENASEM_")

# Crear una barra lateral para la selección de pestañas
st.sidebar.title("Navegación")
option = st.sidebar.selectbox("Seleccione una pestaña", ["Introducción", "Filtrar datos", "Equipo de trabajo"])

if option == "Introducción":
    #
    st.subheader("Sobre el envejecimiento en México")



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

            # Botón para descargar el dataframe reducido en formato xlsx
            xlsx_data = convert_df_to_xlsx(reduced_data)
            st.download_button(
                label="Descargar Dataframe en formato XLSX",
                data=xlsx_data,
                file_name="dataframe_reducido.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

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

