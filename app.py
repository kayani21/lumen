import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

# Page setup
st.set_page_config(page_title="Lumen", page_icon="✨", layout="centered")

# ---- ATMOSPHERIC STARFIELD ----
st.markdown("""
<style>
    /* Deep space base with subtle nebula glow */
    .stApp {
        background:
            radial-gradient(ellipse at 20% 30%, rgba(139, 92, 246, 0.18) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 70%, rgba(251, 191, 36, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 100%, rgba(236, 72, 153, 0.10) 0%, transparent 60%),
            #0A0A14;
    }

    /* Star layer 1 (small, far away) */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        background-image:
            radial-gradient(1px 1px at 25px 50px, #FFFFFF, transparent),
            radial-gradient(1px 1px at 75px 120px, #E5E7EB, transparent),
            radial-gradient(1px 1px at 125px 80px, #FBBF24, transparent),
            radial-gradient(1px 1px at 175px 200px, #FFFFFF, transparent),
            radial-gradient(1px 1px at 225px 150px, #E5E7EB, transparent),
            radial-gradient(1px 1px at 275px 250px, #FFFFFF, transparent),
            radial-gradient(1px 1px at 325px 100px, #FBBF24, transparent),
            radial-gradient(1px 1px at 375px 50px, #FFFFFF, transparent),
            radial-gradient(1px 1px at 425px 200px, #E5E7EB, transparent),
            radial-gradient(1px 1px at 475px 130px, #FFFFFF, transparent),
            radial-gradient(1px 1px at 525px 280px, #FBBF24, transparent),
            radial-gradient(1px 1px at 575px 75px, #FFFFFF, transparent),
            radial-gradient(1px 1px at 625px 225px, #E5E7EB, transparent),
            radial-gradient(1px 1px at 675px 165px, #FFFFFF, transparent),
            radial-gradient(1px 1px at 725px 30px, #FBBF24, transparent),
            radial-gradient(1px 1px at 775px 195px, #FFFFFF, transparent),
            radial-gradient(1px 1px at 825px 115px, #E5E7EB, transparent),
            radial-gradient(1px 1px at 875px 260px, #FFFFFF, transparent),
            radial-gradient(1px 1px at 925px 90px, #FBBF24, transparent),
            radial-gradient(1px 1px at 975px 175px, #FFFFFF, transparent),
            radial-gradient(1px 1px at 1025px 240px, #E5E7EB, transparent),
            radial-gradient(1px 1px at 1075px 60px, #FFFFFF, transparent),
            radial-gradient(1px 1px at 1125px 145px, #FBBF24, transparent),
            radial-gradient(1px 1px at 1175px 210px, #FFFFFF, transparent);
        background-repeat: repeat;
        background-size: 1200px 300px;
        animation: drift-slow 300s linear infinite;
        opacity: 1;
    }

    /* Star layer 2 (medium, with glow) */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
        background-image:
            radial-gradient(2px 2px at 100px 80px, #FBBF24, transparent),
            radial-gradient(2px 2px at 300px 200px, #FFFFFF, transparent),
            radial-gradient(2px 2px at 500px 50px, #FBBF24, transparent),
            radial-gradient(2px 2px at 700px 250px, #FFFFFF, transparent),
            radial-gradient(2px 2px at 900px 150px, #F59E0B, transparent),
            radial-gradient(2px 2px at 1100px 100px, #FFFFFF, transparent),
            radial-gradient(2px 2px at 200px 350px, #FBBF24, transparent),
            radial-gradient(2px 2px at 600px 400px, #FFFFFF, transparent),
            radial-gradient(2px 2px at 1000px 380px, #F59E0B, transparent);
        background-repeat: repeat;
        background-size: 1200px 500px;
        animation: drift-medium 180s linear infinite;
        opacity: 1;
    }

    @keyframes drift-slow {
        from { transform: translateY(0); }
        to { transform: translateY(-3000px); }
    }
    @keyframes drift-medium {
        from { transform: translateY(0); }
        to { transform: translateY(-3000px); }
    }

    /* Make all main content sit ABOVE the star layers */
    .main .block-container {
        position: relative;
        z-index: 10;
    }

    /* Glowing header card */
    .lumen-header {
        background: linear-gradient(135deg, rgba(20, 20, 35, 0.85) 0%, rgba(10, 10, 20, 0.85) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        border: 1px solid rgba(251, 191, 36, 0.2);
        box-shadow: 0 0 80px rgba(251, 191, 36, 0.1);
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        z-index: 10;
    }
    .lumen-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 50%, #FFFFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.02em;
        text-shadow: 0 0 60px rgba(251, 191, 36, 0.3);
    }
    .lumen-tagline {
        font-size: 1.1rem;
        color: #FBBF24;
        margin-top: 0.5rem;
        font-weight: 500;
        letter-spacing: 0.1em;
    }
    .lumen-caption {
        font-size: 0.875rem;
        color: #9CA3AF;
        margin-top: 0.75rem;
        font-weight: 400;
    }

    /* Frosted glass effect */
    [data-testid="stFileUploader"] {
        background: rgba(20, 20, 35, 0.7);
        backdrop-filter: blur(10px);
        border: 1px dashed rgba(251, 191, 36, 0.3);
        border-radius: 12px;
        padding: 1rem;
    }
    [data-testid="stAlert"] {
        background: rgba(20, 20, 35, 0.7) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(251, 191, 36, 0.2);
    }
    [data-testid="stTextInput"] input {
        background: rgba(20, 20, 35, 0.7) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(251, 191, 36, 0.15) !important;
    }

    /* Hide ONLY Streamlit's own branding (not all headers) */
    #MainMenu {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}

    /* Custom footer */
    .lumen-footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        color: #6B7280;
        font-size: 0.85rem;
        border-top: 1px solid rgba(251, 191, 36, 0.1);
        position: relative;
        z-index: 10;
    }
    .lumen-footer a {
        color: #FBBF24;
        text-decoration: none;
    }
    .lumen-footer a:hover {
        color: #F59E0B;
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("""
<div class="lumen-header">
    <h1 class="lumen-title">✨ Lumen</h1>
    <p class="lumen-tagline">CLARITY, ON DEMAND</p>
    <p class="lumen-caption">Upload any PDF and ask questions about it · Powered by LangChain, ChromaDB, and Claude</p>
</div>
""", unsafe_allow_html=True)

# ---- API KEY HANDLING ----
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

if not api_key:
    api_key = st.text_input(
        "Enter your Anthropic API key:",
        type="password",
        help="Get one free at console.anthropic.com"
    )

if api_key:
    os.environ["ANTHROPIC_API_KEY"] = api_key

# ---- SESSION STATE ----
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

MAX_FREE_QUESTIONS = 5

# ---- FILE UPLOADER ----
uploaded_file = st.file_uploader("📄 Upload a PDF to get started", type="pdf")

if uploaded_file and api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    @st.cache_resource
    def build_chain(pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

        prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context. "
            "If the answer isn't in the context, say so honestly.\n\n"
            "Context:\n{context}\n\nQuestion: {question}"
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain, len(chunks)

    with st.spinner("✨ Reading your document..."):
        rag_chain, num_chunks = build_chain(pdf_path)

    st.success(f"✨ Lumen has read your document — {num_chunks} chunks indexed and ready.")

    questions_left = MAX_FREE_QUESTIONS - st.session_state.question_count

    if questions_left > 0:
        st.caption(f"💫 You have {questions_left} free question(s) left this session.")
        question = st.text_input("💬 Ask a question about your document:")

        if question:
            with st.spinner("✨ Thinking..."):
                try:
                    answer = rag_chain.invoke(question)
                    st.session_state.question_count += 1
                    st.markdown("### Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")
    else:
        st.warning(
            "You've reached the free question limit for this session. "
            "Refresh the page to start over, or enter your own Anthropic API key for unlimited use."
        )

# ---- FOOTER ----
st.markdown("""
<div class="lumen-footer">
    Built by <strong>Kayla Williams</strong> ·
    <a href="https://kaylaxtechportfolio.my.canva.site" target="_blank">Portfolio</a> ·
    Powered by Claude
</div>
""", unsafe_allow_html=True)