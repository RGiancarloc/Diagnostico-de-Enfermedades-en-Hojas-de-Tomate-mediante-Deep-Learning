import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
st.set_page_config(
    page_title="Diagnóstico de Enfermedades en Tomate v3",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título
st.title("🍅 Detector de Enfermedades en Hojas de Tomate v3")
st.markdown("""
Sistema de inteligencia artificial para identificar **9 enfermedades comunes** en cultivos de tomate 
y recomendar **tratamientos específicos**.
""")

# Cargar modelo
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/best_tomato_model.h5')


model = load_model()

# Información de enfermedades y tratamientos
DISEASE_INFO = {
    0: {
        'name': 'Desconocido',
        'treatment': 'Clasificación no definida. Verifique que la imagen sea una hoja de tomate clara y centrada.',
        'symptoms': 'N/A',
        'color': 'gray'
    },
    1: {
        'name': 'Tizón Temprano',
        'treatment': '1. Aplicar fungicidas a base de clorotalonil\n2. Eliminar hojas infectadas\n3. Mejorar circulación de aire',
        'symptoms': 'Manchas concéntricas oscuras con halos amarillos',
        'color': '#8B4513'
    },
    2: {
        'name': 'Tizón Tardío',
        'treatment': '1. Fungicidas sistémicos (ej. Fosetil-Al)\n2. Reducir humedad\n3. Destruir plantas gravemente afectadas',
        'symptoms': 'Lesiones acuosas que se vuelven marrones',
        'color': '#A52A2A'
    },
    3: {
        'name': 'Moho Foliar',
        'treatment': '1. Fungicidas preventivos (cobre)\n2. Evitar riego por aspersión\n3. Podar para mejorar aireación',
        'symptoms': 'Manchas amarillas en haz, moho púrpura en envés',
        'color': '#9370DB'
    },
    4: {
        'name': 'Mancha Septoria',
        'treatment': '1. Aplicar fungicidas (azoxystrobin)\n2. Rotación de cultivos\n3. Eliminar residuos infectados',
        'symptoms': 'Pequeñas manchas circulares con centros grises',
        'color': '#708090'
    },
    5: {
        'name': 'Ácaros',
        'treatment': '1. Jabones insecticidas\n2. Aceite de neem\n3. Introducir depredadores naturales',
        'symptoms': 'Punteado amarillo, telarañas finas en envés',
        'color': '#FFA500'
    },
    6: {
        'name': 'Mancha Objetivo',
        'treatment': '1. Fungicidas (mancozeb)\n2. Reducir estrés hídrico\n3. Solarización del suelo',
        'symptoms': 'Manchas con anillos concéntricos',
        'color': '#CD5C5C'
    },
    7: {
        'name': 'Virus Enrollamiento',
        'treatment': '1. Controlar mosca blanca (vector)\n2. Usar variedades resistentes\n3. Eliminar plantas infectadas',
        'symptoms': 'Hojas amarillas enrolladas hacia arriba',
        'color': '#FFD700'
    },
    8: {
        'name': 'Virus Mosaico',
        'treatment': '1. Eliminar plantas infectadas\n2. Control de áfidos\n3. Desinfectar herramientas',
        'symptoms': 'Patrón de mosaico verde claro/oscuro',
        'color': '#9ACD32'
    },
        9: {

        'name': 'Sano',
        'treatment': 'No se requiere tratamiento. Continúe con el monitoreo regular.',
        'symptoms': 'Hojas verdes sin manchas ni decoloraciones anormales',
        'color': 'green'
    }
}

# Preprocesamiento de imágenes
def preprocess_image(image):
    # Convertir imagen a RGB
    img = image.convert("RGB")
    
    # Redimensionar a 300x300 (tamaño que el modelo espera)
    img = img.resize((300, 300))
    
    # Convertir a array numpy y normalizar
    img = np.array(img) / 255.0
    
    # Expandir dimensiones para batch (1, 300, 300, 3)
    return np.expand_dims(img, axis=0)


# Sidebar
st.sidebar.title("Opciones")
app_mode = st.sidebar.selectbox(
    "Modo de Operación",
    ["Diagnóstico", "Guía de Enfermedades", "Reportes Técnicos"]
)

# Módulo de diagnóstico
if app_mode == "Diagnóstico":
    st.header("🔍 Diagnóstico por Imagen")
    uploaded_file = st.file_uploader(
        "Suba una foto de hoja de tomate", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Imagen subida", use_column_width=True)
            
        with col2:
            if st.button("Analizar", type="primary"):
                with st.spinner("Procesando imagen..."):
                    processed_img = preprocess_image(image)
                    prediction = model.predict(processed_img)
                    pred_class = int(np.argmax(prediction[0]))
                    confidence = float(np.max(prediction[0]) * 100)
                    disease = DISEASE_INFO.get(pred_class, {
                        'name': 'Desconocido',
                        'treatment': 'N/A',
                        'symptoms': 'N/A',
                        'color': 'gray'
                    })

                    # Resultado
                    st.markdown(f"""
                    <div style='border-left: 5px solid {disease['color']}; padding: 10px;'>
                        <h3 style='color:{disease['color']}'>{disease['name']}</h3>
                        <p><b>Confianza:</b> {confidence:.1f}%</p>
                        <p><b>Síntomas típicos:</b> {disease['symptoms']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tratamiento
                    st.subheader("📋 Tratamiento Recomendado")
                    st.markdown(f"```\n{disease['treatment']}\n```")

                    # Gráfico de probabilidades
                    st.subheader("📊 Distribución de Probabilidades")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    probs = prediction[0] * 100
                    colors = [DISEASE_INFO[i]['color'] if i in DISEASE_INFO else 'gray' for i in range(len(probs))]
                    bars = ax.bar([DISEASE_INFO[i]['name'] if i in DISEASE_INFO else f"Clase {i}" for i in range(len(probs))], 
                                  probs, color=colors)
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel("Probabilidad (%)")
                    plt.ylim(0, 100)

                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.1f}%',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                    st.pyplot(fig)

                    # 📥 Generar y descargar reporte PDF
                    import io
                    from reportlab.lib.pagesizes import letter
                    from reportlab.pdfgen import canvas
                    from datetime import datetime
                    from reportlab.lib.utils import ImageReader

                    # 1. Guardar el gráfico matplotlib como imagen en memoria
                    fig_buffer = io.BytesIO()
                    fig.savefig(fig_buffer, format="png", bbox_inches="tight")
                    fig_buffer.seek(0)

                    # 2. Crear el PDF en memoria
                    pdf_buffer = io.BytesIO()
                    c = canvas.Canvas(pdf_buffer, pagesize=letter)

                    # 3. Texto del PDF
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(50, 750, "Reporte de Diagnóstico - Hoja de Tomate")
                    c.setFont("Helvetica", 10)
                    c.drawString(50, 735, f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    c.drawString(50, 710, f"🔍 Enfermedad detectada: {disease['name']}")
                    c.drawString(50, 695, f"📊 Confianza: {confidence:.2f}%")

                    c.drawString(50, 670, "Síntomas típicos:")
                    text = c.beginText(60, 655)
                    text.textLines(disease['symptoms'])
                    c.drawText(text)

                    c.drawString(50, 620, "Tratamiento recomendado:")
                    text2 = c.beginText(60, 605)
                    text2.textLines(disease['treatment'])
                    c.drawText(text2)

                    # 4. Insertar gráfico en el PDF (como imagen)
                    chart_image = ImageReader(fig_buffer)
                    c.drawImage(chart_image, 50, 350, width=500, height=200)  # Ajusta tamaño y posición

                    c.showPage()
                    c.save()
                    pdf_buffer.seek(0)

                    # 5. Botón de descarga del PDF completo
                    st.download_button(
                        label="📄 Descargar Reporte PDF con gráfico",
                        data=pdf_buffer,
                        file_name="reporte_tomate.pdf",
                        mime="application/pdf"
                    )


# Módulo educativo
elif app_mode == "Guía de Enfermedades":
    st.header("📚 Guía Visual de Enfermedades")
    
    tabs = st.tabs([DISEASE_INFO[i]['name'] for i in range(len(DISEASE_INFO))])
    
    for i, tab in enumerate(tabs):
        with tab:
            disease = DISEASE_INFO[i]
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {disease['name']}")
                st.image(f"data/examples/{i}.jpg", 
                        caption=f"Ejemplo de {disease['name']}")
            
            with col2:
                st.markdown("**Síntomas característicos:**")
                st.write(disease['symptoms'])
                
                st.markdown("**Tratamiento recomendado:**")
                st.markdown(f"```\n{disease['treatment']}\n```")
                
                if i != 0:  # No mostrar para plantas sanas
                    st.markdown("**Medidas preventivas:**")
                    st.write("- Rotación de cultivos (3-4 años)")
                    st.write("- Uso de semillas certificadas")
                    st.write("- Monitoreo semanal de cultivos")

# Módulo técnico
elif app_mode == "Reportes Técnicos":
    st.header("📈 Rendimiento del Modelo")
    
    st.subheader("Comparación de Arquitecturas")
    st.image("reports/model_comparison.png")
    
    st.subheader("Matriz de Confusión")
    st.image("reports/confusion_matrix.png")
    
    st.subheader("Curvas de Aprendizaje")
    st.image("reports/learning_curves.png")
    
    st.markdown("""
    ### Métricas Clave:
    | Métrica               | Valor   |
    |-----------------------|---------|
    | Precisión Global      | 97.2%   |
    | Sensibilidad Promedio | 96.8%   |
    | F1-Score Promedio     | 96.9%   |
    | Tiempo Inferencia     | 120ms   |
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Dataset:** PlantVillage (18,000+ imágenes)  
**Modelo:** CNN Optimizado  
**Precisión:** 97.2% (test)  
**Actualizado:** Enero 2024
""")
