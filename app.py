import os
import platform
import traceback
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# 1. Configuración de página
st.set_page_config(
    page_title="Asistente RAG PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado para detalles estéticos
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Barra lateral (Configuración y Configuración de API)
with st.sidebar:
    # Imagen de encabezado cambiada a Uximg.png
    try:
        image = Image.open('Uximg.png')
        st.image(image, use_container_width=True)
    except Exception:
        pass

    st.title("⚙️ Configuración")
    st.caption("Este agente analiza el contenido de tu PDF mediante modelos del lenguaje.")

    st.markdown("---")

    # Entrada de API Key
    ke = st.text_input('Clave API de OpenAI', type="password", help="Tu API Key no se almacena en ningún servidor.")
    if ke:
        os.environ['OPENAI_API_KEY'] = ke
        st.success("API Key configurada correctamente", icon="✅")
    else:
        st.warning("Ingresa tu API Key para habilitar la app", icon="🔑")

    st.markdown("---")
    st.caption(f"🐍 Python v{platform.python_version()}")

# 3. Panel Principal
st.title("Analizador de Documentos PDF 💬")
st.write("Carga tu archivo, procesa el texto y realiza preguntas sobre su contenido en tiempo real.")

st.markdown("---")

# Sección de Carga
col_upload, col_info = st.columns([1, 1], gap="medium")

with col_upload:
    st.subheader("1. Cargar Documento")
    pdf = st.file_uploader("Selecciona un archivo PDF", type="pdf", label_visibility="collapsed")

# Procesamiento del PDF
if pdf is not None:
    if not ke:
        with col_info:
            st.warning("Por favor ingresa tu API Key en la barra lateral para continuar.")
    else:
        try:
            with st.spinner("Procesando documento..."):
                # Extracción de texto
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                
                # Splitter
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=500,
                    chunk_overlap=20,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                
                # Vectorstore
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)

            with col_info:
                st.subheader("2. Estado del Documento")
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Caracteres", f"{len(text):,}")
                m_col2.metric("Fragmentos", len(chunks))
                st.success("¡Documento procesado y listo!", icon="🎉")

            st.markdown("---")

            # 4. Sección de Preguntas y Respuestas
            st.subheader("3. Consulta sobre el documento")
            user_question = st.text_input("¿Qué deseas saber?", placeholder="Ej. ¿Cuál es el tema principal del documento?")

            if user_question:
                with st.spinner("Buscando respuestas..."):
                    docs = knowledge_base.similarity_search(user_question)
                    llm = OpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=user_question)

                # Visualización con chat nativo
                with st.chat_message("user"):
                    st.write(user_question)

                with st.chat_message("assistant"):
                    st.write(response)

        except Exception as e:
            st.error("Ocurrió un error al procesar el archivo")
            with st.expander("Ver detalle del error"):
                st.code(traceback.format_exc())
else:
    with col_info:
        st.subheader("2. Estado del Documento")
        st.info("Esperando que cargues un archivo PDF para comenzar.")
