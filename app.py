
import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma


# =====================================================
# 1. PAGE SETTINGS
# =====================================================
load_dotenv()

st.set_page_config(
    page_title="Manufacturing SOP Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# 2. PROFESSIONAL DESIGN
# =====================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #10243A;
        --navy-light: #183B56;
        --teal: #11A89A;
        --teal-dark: #087F78;
        --canvas: #F3F6FA;
        --line: #E1E8EF;
        --muted: #718096;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--canvas);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--navy) !important;
    }

    [data-testid="stSidebar"] {
        background: var(--navy);
    }

    [data-testid="stSidebar"] * {
        color: #F3F7FB;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #B8CAD8;
    }

    [data-testid="stSidebar"] hr {
        border-color: #35516A;
    }

    .stButton button {
        background: var(--teal);
        color: white;
        border: 1px solid var(--teal);
        border-radius: 10px;
        min-height: 43px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    .stButton button:hover {
        background: var(--teal-dark);
        border-color: var(--teal-dark);
        color: white;
    }

    .stTextInput input {
        background: white;
        border: 1px solid #D4DFE8;
        border-radius: 10px;
        min-height: 43px;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #B6C9D7;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stExpander"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 12px;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 16px rgba(16, 36, 58, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--navy);
        font-family: 'Manrope', sans-serif;
    }

    .footer-note {
        color: #8291A0;
        font-size: 12px;
        text-align: center;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 3. SESSION STATE
# =====================================================
defaults = {
    "logged_in": False,
    "user_email": "",
    "vectorstore": None,
    "uploaded_file_name": "",
    "chunk_count": 0,
    "last_question": "",
    "last_answer": "",
    "last_sources": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# 4. GOOGLE API KEY
# =====================================================
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        api_key = None


# =====================================================
# 5. AI MODELS
# =====================================================
@st.cache_resource
def get_embeddings(key):
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=key,
    )


@st.cache_resource
def get_llm(key):
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
        google_api_key=key,
    )


# =====================================================
# 6. LOGIN PAGE
# =====================================================
if not st.session_state.logged_in:

    st.write("")

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.write("")
        st.write("")

        st.markdown("## 🏭 Manufacturing")
        st.markdown("# SOP Assistant")
        st.caption("GENERATIVE AI  ·  SOP KNOWLEDGE")

        st.write("")

        with st.container(border=True):
            st.caption("SMARTER SOP SEARCH")
            st.markdown("## The right procedure.")
            st.markdown("## The right answer.")

            st.write(
                "Explore manufacturing procedures and find answers "
                "from your uploaded Standard Operating Procedure "
                "documents."
            )

        st.write("")
        st.markdown("### Built for manufacturing knowledge")

        st.markdown(
            """
            - 📄  Upload SOP PDF documents
            - 💬  Ask questions in simple language
            - 📚  View the source text used for answers
            """
        )

    with right:
        st.write("")
        st.write("")
        st.write("")

        with st.container(border=True):
            st.caption("WORKSPACE LOGIN")
            st.markdown("## Welcome back")

            st.write(
                "Enter your email address to access the "
                "Manufacturing SOP Assistant."
            )

            email = st.text_input(
                "Email address",
                placeholder="you@company.com",
                key="login_email",
            )

            st.caption(
                "Demo login: email format is checked only. "
                "Email ownership is not verified."
            )

            if st.button(
                "Continue to workspace →",
                use_container_width=True,
            ):
                email = email.strip()

                if not re.fullmatch(
                    r"[^@\s]+@[^@\s]+\.[^@\s]+",
                    email,
                ):
                    st.error("Please enter a valid email address.")
                else:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.rerun()

    st.write("")
    st.markdown(
        '<p class="footer-note">Manufacturing SOP Assistant · '
        'AI-powered document search</p>',
        unsafe_allow_html=True,
    )

    st.stop()


# =====================================================
# 7. CHECK API KEY
# =====================================================
if not api_key:
    st.error(
        "Google API key not found. Please check your .env file."
    )
    st.info(
        "The .env file should contain GOOGLE_API_KEY=your_key. "
        "Keep your API key private."
    )

    if st.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.rerun()

    st.stop()


# =====================================================
# 8. SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 🏭 SOP Assistant")
    st.caption("MANUFACTURING WORKSPACE")

    st.divider()

    st.markdown("### Your account")
    st.caption(st.session_state.user_email)

    st.divider()

    st.markdown("### Knowledge source")

    if st.session_state.uploaded_file_name:
        st.success("SOP loaded")
        st.caption(st.session_state.uploaded_file_name)
        st.caption(
            f"{st.session_state.chunk_count} text sections"
        )
    else:
        st.info("No SOP document loaded yet")

    st.divider()

    if st.button("Log out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.rerun()


# =====================================================
# 9. MAIN DASHBOARD
# =====================================================
st.caption("MANUFACTURING KNOWLEDGE WORKSPACE")
st.title("🏭 Manufacturing SOP Assistant")

st.write(
    "Upload an SOP document, search the procedure, and get "
    "answers based on the document you provide."
)

st.write("")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Knowledge source",
        "Ready" if st.session_state.uploaded_file_name else "Not loaded",
    )

with metric2:
    st.metric(
        "Text sections",
        st.session_state.chunk_count,
    )

with metric3:
    st.metric("Assistant status", "AI enabled")


# =====================================================
# 10. UPLOAD SOP PDF
# =====================================================
st.write("")
st.divider()
st.subheader("01 · SOP Knowledge Source")

st.caption(
    "Upload a manufacturing Standard Operating Procedure PDF."
)

with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Choose SOP PDF",
        type=["pdf"],
        help="Upload a PDF containing readable SOP text.",
        key="sop_pdf",
    )

    if uploaded_file is not None:
        should_process = (
            uploaded_file.name != st.session_state.uploaded_file_name
            or st.session_state.vectorstore is None
        )

        if should_process:
            with st.spinner(
                "Reading and indexing your SOP document..."
            ):
                try:
                    data_dir = Path("data")
                    data_dir.mkdir(parents=True, exist_ok=True)

                    pdf_path = data_dir / "uploaded_sop.pdf"
                    pdf_path.write_bytes(uploaded_file.getvalue())

                    documents = PyPDFLoader(str(pdf_path)).load()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=150,
                    )

                    chunks = splitter.split_documents(documents)

                    if not chunks:
                        st.error(
                            "No readable text was found in this PDF. "
                            "If it is scanned, OCR may be required."
                        )
                    else:
                        new_vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=get_embeddings(api_key),
                        )

                        st.session_state.vectorstore = new_vectorstore
                        st.session_state.uploaded_file_name = (
                            uploaded_file.name
                        )
                        st.session_state.chunk_count = len(chunks)
                        st.session_state.last_question = ""
                        st.session_state.last_answer = ""
                        st.session_state.last_sources = []

                        st.success(
                            f"SOP processed successfully! "
                            f"{len(chunks)} text sections created."
                        )

                except Exception as error:
                    st.error(f"Could not process the PDF: {error}")

    if st.session_state.uploaded_file_name:
        st.caption(
            f"Current document: "
            f"{st.session_state.uploaded_file_name}"
        )


# =====================================================
# 11. ASK THE AI ASSISTANT
# =====================================================
st.write("")
st.divider()
st.subheader("02 · Ask the SOP Assistant")

st.caption(
    "Ask a question about the uploaded procedure or safety precautions."
)

with st.container(border=True):
    question = st.text_input(
        "Your question",
        placeholder="Example: What safety precautions should be followed?",
        key="sop_question",
    )

    if st.button(
        "Search SOP and generate answer →",
        use_container_width=True,
    ):
        if not question.strip():
            st.warning("Please enter a question.")

        elif st.session_state.vectorstore is None:
            st.warning("Please upload an SOP PDF first.")

        else:
            with st.spinner(
                "Searching the SOP and preparing an answer..."
            ):
                try:
                    results = (
                        st.session_state.vectorstore.similarity_search(
                            question.strip(),
                            k=3,
                        )
                    )

                    if not results:
                        answer = (
                            "Information is not available in the provided SOP."
                        )
                        sources = []

                    else:
                        context = "\n\n".join(
                            document.page_content
                            for document in results
                        )

                        prompt = f"""
You are a Manufacturing SOP Assistant.

Answer using ONLY the SOP text provided below.
Do not invent procedures or safety instructions.
If the answer is not supported by the text, say:
"Information is not available in the provided SOP."

SOP TEXT:
{context}

USER QUESTION:
{question.strip()}

Write the answer in simple, clear English.
"""

                        response = get_llm(api_key).invoke(prompt)
                        answer = response.content

                        if isinstance(answer, list):
                            answer = "\n".join(
                                str(item) for item in answer
                            )

                        sources = [
                            document.page_content
                            for document in results
                        ]

                    st.session_state.last_question = question.strip()
                    st.session_state.last_answer = str(answer)
                    st.session_state.last_sources = sources

                except Exception as error:
                    st.error(
                        f"Unable to generate an answer: {error}"
                    )


# =====================================================
# 12. SHOW AI ANSWER
# =====================================================
if st.session_state.last_answer:
    st.write("")
    st.subheader("AI-generated answer")

    with st.container(border=True):
        st.caption("AI RESPONSE")
        st.markdown(st.session_state.last_answer)

    st.caption(
        f"Question: {st.session_state.last_question}"
    )

    with st.expander("View retrieved SOP source text"):
        sources = st.session_state.last_sources

        if sources:
            for index, source_text in enumerate(sources, start=1):
                st.markdown(f"**Source {index}**")
                st.write(source_text)

                if index < len(sources):
                    st.divider()
        else:
            st.info("No source text was retrieved.")


# =====================================================
# 13. FOOTER
# =====================================================
st.write("")
st.divider()

st.markdown(
    '<p class="footer-note">Manufacturing SOP Assistant · '
    'Use approved workplace SOPs as the source of truth.</p>',
    unsafe_allow_html=True,
)
