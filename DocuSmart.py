import streamlit as st
import os
import logging
import pdfplumber
import multiprocessing
import pickle
from ctransformers import AutoModelForCausalLM

from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever, PreProcessor, TransformersReader

from transformers import pipeline
from PIL import Image

logging.getLogger("pdfminer").setLevel(logging.ERROR)

st.set_page_config(page_title="DocuSmart Bot", layout="wide")

# --- Configuration for Indexer Integration ---
INDEX_SAVE_PATH = "indexed_data.pkl" # Must match the path in indexer_script.py
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Must match indexer

# --- Load LLM (Mistral 7B Instruct) ---
@st.cache_resource
def load_llm():
    return AutoModelForCausalLM.from_pretrained(
        ".",
        model_file=os.path.abspath("../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
        model_type="mistral",
        context_length=4096
    )

llm = load_llm()

# --- Load LayoutLM Document QA Model ---
@st.cache_resource
def load_layoutlm_qa_model():
    return pipeline("document-question-answering", model="impira/layoutlm-document-qa")

layoutlm_qa_pipeline = load_layoutlm_qa_model()

# --- Load Extractive QA Reader ---
@st.cache_resource
def load_reader():
    return TransformersReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

reader = load_reader()

# --- THEME TOGGLE ---
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

st.sidebar.selectbox(
    "Choose Theme",
    options=["Light", "Dark"],
    index=0 if st.session_state.theme == "Light" else 1,
    key="theme"
)

# --- SIMPLIFIED inject_theme_css function ---
def inject_theme_css(theme):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    css_file = os.path.join(script_dir, f"theme_{theme.lower()}.css") # This will be theme_light.css

    if os.path.exists(css_file):
        with open(css_file) as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.error(f"Theme file '{css_file}' not found. Please ensure it's in the correct directory.")

inject_theme_css(st.session_state.theme)

# --- LOGO & TITLE ---
st.image("SE_logo.png", width=250)
st.markdown(
    "<h1 style='font-size:2.8rem; font-weight:800; margin-bottom:0.5em;'>"
    "<span class='haaz-purple'>DocuSmart Bot</span>"
    "</h1>",
    unsafe_allow_html=True
)

# --- SIDEBAR ---
st.sidebar.image("SE_logo.png", width=120)
st.sidebar.header("PDF Index Management")

# --- Function to load index ---
def load_indexed_data():
    if os.path.exists(INDEX_SAVE_PATH):
        try:
            with open(INDEX_SAVE_PATH, 'rb') as f:
                saved_data = pickle.load(f)
                document_store = saved_data['document_store']
                retriever = saved_data['retriever']
                # Re-initialize retriever with loaded document store if necessary
                if not isinstance(retriever, EmbeddingRetriever) or retriever.document_store != document_store:
                     retriever = EmbeddingRetriever(
                        document_store=document_store,
                        embedding_model=EMBEDDING_MODEL_NAME
                    )
                     document_store.update_embeddings(retriever) # Ensure embeddings are fresh if retriever was re-created
                return document_store, retriever, True
        except Exception as e:
            st.error(f"Error loading indexed data: {e}. Please ensure the indexer script is running correctly or re-index.")
            return None, None, False
    return None, None, False

# --- SESSION STATE INIT & Initial Load ---
if "document_store" not in st.session_state or st.session_state.document_store is None:
    st.session_state.document_store, st.session_state.retriever, st.session_state.indexed = load_indexed_data()
    if not st.session_state.indexed:
        st.warning("No indexed data found. Please run the background indexer script first or ensure the correct path is configured.")

for key, default in [
    ("docs", []), # This might not be strictly needed if only loading from saved index
    ("chat_history", []),
    ("layoutlm_chat_history", [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- REFRESH INDEX BUTTON ---
st.sidebar.caption("Click to load the latest indexed data from the background process.")
if st.sidebar.button("Reload Indexed PDFs"):
    st.session_state.document_store, st.session_state.retriever, st.session_state.indexed = load_indexed_data()
    if st.session_state.indexed:
        st.success("Successfully reloaded indexed data.")
        st.session_state.chat_history = [] # Clear chat history on reload for clarity
    else:
        st.warning("Could not reload indexed data. Ensure background indexer is running.")

# --- SHOW SAMPLE OF INDEXED DOCS ---
if st.session_state.indexed and st.session_state.document_store:
    try:
        # For InMemoryDocumentStore, you can check number of docs
        num_docs = len(st.session_state.document_store.get_all_documents())
        st.sidebar.write(f"Total documents indexed: {num_docs}")
    except Exception as e:
        st.sidebar.warning(f"Could not retrieve document count: {e}")

# --- Sidebar: LLM context size and top_k controls ---
st.sidebar.markdown("---")
st.sidebar.subheader("Generative Answer Settings")
max_context_chars = st.sidebar.slider(
    "Max context characters for LLM",
    min_value=1000,
    max_value=8000,
    value=3000,
    step=500,
    help="Controls how much context is sent to the LLM for generative answers."
)
top_k_llm = st.sidebar.slider(
    "Number of results to retrieve",
    min_value=1,
    max_value=8,
    value=4,
    step=1,
    help="Controls how many document results are retrieved for the documents."
)
max_new_tokens = st.sidebar.slider(
    "Max tokens in LLM answer",
    min_value=64,
    max_value=1024,
    value=256,
    step=32,
    help="Controls the maximum length of the LLM's answer."
)

# --- Sidebar: Extractive top_k control ---
top_k_extractive = st.sidebar.slider(
    "Number of chunks to retrieve (Extractive)",
    min_value=1,
    max_value=8,
    value=2,
    step=1,
    help="How many chunks to return for extractive answers."
)

# --- Function to answer with LLM ---
def answer_with_llm(question, retriever, top_k=4, max_context_chars=3000, max_new_tokens=256):
    if retriever is None:
        return "Error: Retriever not initialized. Please ensure PDFs are indexed."

    retrieved_docs = retriever.retrieve(query=question, top_k=top_k)
    context = "\n\n".join([doc.content for doc in retrieved_docs])
    if len(context) > max_context_chars:
        st.warning("Context truncated to fit model's maximum context length.")
        context = context[:max_context_chars]
    prompt = f"Given the following context, answer the question in detail.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    try:
        num_cores = max(1, multiprocessing.cpu_count() // 2)
        output = llm(prompt, max_new_tokens=max_new_tokens, threads=num_cores)
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        output = "Sorry, there was an error generating the answer."
    return output

# --- Function to answer with Extractive QA (returns full chunk) ---
def answer_with_extractive_qa(question, retriever, document_store, top_k=2):
    """
    Returns the full content of the top-k retrieved chunks (not just a span).
    """
    if retriever is None or document_store is None:
        return "Error: Retriever or Document Store not initialized. Please ensure PDFs are indexed."

    retrieved_docs = retriever.retrieve(query=question, top_k=top_k)
    if not retrieved_docs:
        return "No relevant chunk found in the documents."
    # Optionally, you can join multiple chunks if top_k > 1
    return "\n\n---\n\n".join([doc.content for doc in retrieved_docs])

# --- MAIN APP LAYOUT WITH TABS ---
st.subheader("Choose your Q&A Mode:")
tab1, tab2 = st.tabs(["Chat with Indexed PDFs", "Upload and ask about Document Image"]) # Original, longer tab titles

with tab1:
    if st.session_state.indexed and st.session_state.document_store and st.session_state.retriever:
        st.subheader("Chat with your Indexed PDFs:")

        # --- Display chat history ---
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['question']}")
            mode_label = "Extractive" if chat.get("mode", "").startswith("Extractive") else "Generative"
            for i, ans in enumerate(chat['answers'], 1):
                st.markdown(
                    f"""
                    <div style='margin-bottom:1em;'>
                    <b>Answer {i} ({mode_label}):</b><br>
                    According to your documents:<br>
                    <blockquote>{ans['content']}</blockquote>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # --- Input for new question in a form ---
        with st.form(key="chat_form", clear_on_submit=True):
            query = st.text_input("Your question:", key="chat_input")
            answer_mode = st.selectbox(
                "Answer Mode",
                ["Extractive (Direct from PDF)", "Generative (LLM Paraphrase)"],
                index=0,
                key="answer_mode_select"
            )
            submitted = st.form_submit_button("Send")
        if submitted and query:
            retriever = st.session_state.retriever
            document_store = st.session_state.document_store
            with st.spinner("Searching..."):
                if answer_mode.startswith("Extractive"):
                    answer = answer_with_extractive_qa(query, retriever, document_store, top_k=top_k_extractive)
                else:
                    answer = answer_with_llm(
                        query,
                        retriever,
                        top_k=top_k_llm,
                        max_context_chars=max_context_chars,
                        max_new_tokens=max_new_tokens
                    )
            st.session_state.chat_history.append({
                "question": query,
                "answers": [{"file": "Multiple", "page": "-", "content": answer, "score": 1.0}],
                "mode": answer_mode # Store mode for display
            })
            st.rerun()

        # --- Optional: Clear chat history button ---
        if st.button("Clear Chat History", key="clear_rag_chat"):
            st.session_state.chat_history = []
            st.rerun()

    else:
        st.markdown(
            "<div class='se-purple-info'>Indexed data not available. Please ensure the background indexer script is running and click 'Reload Indexed PDFs'.</div>",
            unsafe_allow_html=True
        )

with tab2:
    st.subheader("Ask a question about a Document Image:")
    uploaded_file = st.file_uploader("Upload a document image (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Document", use_column_width=True)

        layoutlm_query = st.text_input("Question for the document image:", key="layoutlm_input")
        layoutlm_submitted = st.button("Get Answer from Image", key="submit_layoutlm_qa")

        if layoutlm_submitted and layoutlm_query:
            with st.spinner("Analyzing document image..."):
                try:
                    layoutlm_results = layoutlm_qa_pipeline(image=image, question=layoutlm_query)
                    if layoutlm_results:
                        answer_text = layoutlm_results[0]['answer']
                        st.session_state.layoutlm_chat_history.append({
                            "question": layoutlm_query,
                            "answer": answer_text
                        })
                        st.success("Answer extracted from image:")
                        st.markdown(f"<blockquote>{answer_text}</blockquote>", unsafe_allow_html=True)
                    else:
                        st.warning("No answer found in the document for your question.")
                except Exception as e:
                    st.error(f"Error processing image with LayoutLM: {e}")

        if st.session_state.layoutlm_chat_history:
            st.markdown("---")
            st.subheader("Image Q&A History:")
            for chat in st.session_state.layoutlm_chat_history:
                st.markdown(f"**You:** {chat['question']}")
                st.markdown(f"**Answer:** <blockquote>{chat['answer']}</blockquote>", unsafe_allow_html=True)

        if st.button("Clear Image Q&A History", key="clear_layoutlm_chat"):
            st.session_state.layoutlm_chat_history = []
            st.rerun()