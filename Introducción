import streamlit as st
import pickle

#************************************* Función para cargar o crear el contador de visitas

# Función para cargar el contador de visitas
def load_counter():
    try:
        with open("visit_counter.pkl", "rb") as f:
            counter = pickle.load(f)
        return counter
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        raise Exception("El archivo del contador está vacío o corrupto.")

# Función para guardar el contador de visitas
def save_counter(counter):
    with open("visit_counter.pkl", "wb") as f:
        pickle.dump(counter, f)

# Obtener el contador actual de visitas
try:
    counter = load_counter()
except Exception as e:
    print(e)
    counter = 0

# Incrementar el contador y guardar los cambios
counter += 1
save_counter(counter)

# Mostrar el contador en la aplicación de Streamlit
#st.title("Contador de Visitas")
st.markdown(f"**Esta página ha sido visitada {counter} veces.**")



# Título de la página
st.title("Acerca de Sarc-open-IA")

st.subheader("Objetivo")
st.markdown("""
El objetivo de esta aplicación es asistir al personal médico en la captura, almacenamiento y análisis de datos antropométricos de adultos mayores para la determinación de dependencia funcional y sarcopenia. Es el resultado de una estancia de investigación posdoctoral, resultado de la colaboración entre el **Consejo Nacional de Humanidades Ciencia y Tecnología (CONAHCYT) y la Universidad de Colima (UCOL)** y desarrollada entre **octubre de 2022 y septiembre 2023**, en la que se utilizó una base de datos antropométricos de adultos mayores para crear modelos predictivos de dependencia funcional y sarcopenia. Estos modelos representan la primera fase de una estrategia diseñada para facilitar la identificación temprana de síntomas de condiciones debilitantes en adultos mayores, utilizando técnicas de inteligencia artificial y aprendizaje automático.
"""
           )

st.subheader("Ventajas y características")

st.markdown("""

- **Objetivo de Facilitar su Uso:** Queríamos que nuestra herramienta fuera fácil de usar para el personal médico, incluso si no estaban familiarizados con la inteligencia artificial o la programación. Para lograrlo, elegimos el lenguaje de programación Python y las plataformas Streamlit y GitHub. Estas opciones permiten una fácil visualización y manipulación de la aplicación, además de almacenar los algoritmos en la nube.

- **Interfaz Amigable:** El resultado es una interfaz gráfica que permite a los médicos ingresar los datos antropométricos de los pacientes y ver gráficas útiles para el análisis estadístico. También ofrece un diagnóstico en tiempo real de la sarcopenia, y todo esto se hace utilizando cajas de texto y deslizadores para ingresar y manipular los datos.

- **Accesibilidad Total:** El personal médico puede descargar de forma segura las gráficas y los archivos generados por la aplicación. Además, pueden acceder a ella desde cualquier dispositivo con conexión a internet, ya sea un teléfono celular, una computadora, tablet o laptop.
""")

st.subheader("Método")
st.markdown("""
A partir de los datos registrados en el año 2021 en una muestra de adultos mayores que residen en la Zona Metropolitana, Colima, Villa de Álvarez, México, se procedió al desarrollo de modelos predictivos mediante el algoritmo Random Forest. Mediante la aplicación de técnicas de reducción de dimensionalidad, análisis de componentes principales, clustering jerárquico y teoría de conjuntos rugosos, los modelos presentados aquí generan una lista de condiciones que, en caso de cumplirse, indicarían un posible diagnóstico de sarcopenia. Estas condiciones de diagnóstico fueron propuestas con el objetivo de minimizar la cantidad de parámetros antropométricos y establecer puntos de corte que puedan ser validados por personal médico capacitado. Este enfoque se asemeja a lo que se conoce en inteligencia artificial como un sistema experto, ya que los modelos resultantes requieren validación por parte de especialistas.
"""
           )




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

# Establecer la altura deseada para las imágenes
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
st.write("Si deseas ponerte en contacto con nuestro equipo, puedes enviar un correo a pv_sarceo@ucol.mx")
