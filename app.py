"""
Streamlit interface for the ITCC508 Maya Bank RAG Chatbot.

This app wraps the exact same pipeline built in
ITCC508_PT_M1_RAG_MayaBank.ipynb (Document Loading -> Chunking ->
all-MiniLM-L6-v2 Embeddings -> ChromaDB -> Groq LLM) in a chat UI.

Run it from the SAME folder that contains your `my_data/` directory
(the one with maya_bank_general_terms_conditions.txt and
maya_bank_savings_terms_conditions.txt), e.g.:

    streamlit run app.py

Requirements (see requirements.txt):
    pip install streamlit "langchain<1.0" "langchain-community<1.0" \
        langchain-groq langchain-huggingface sentence-transformers chromadb pypdf
"""

import os
import streamlit as st

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

DATA_DIR = "my_data"
MODEL_NAME = "openai/gpt-oss-20b"  # same model used in the notebook

# System prompt variants, matching the three configurations tested in
# the notebook's hallucination stress-test (Task 3).
PROMPT_VARIANTS = {
    "Baseline (grounded, temp=0)": {
        "temperature": 0,
        "system_prompt": (
            "You are a specialized AI assistant for the user's uploaded domain.\n"
            "Answer questions strictly using ONLY the provided context below.\n"
            "If the answer cannot be found in the context, reply: "
            "'I cannot answer based on the provided domain data.'\n\n"
            "Context:\n{context}"
        ),
    },
    "Fallback removed (temp=1.0)": {
        "temperature": 1.0,
        "system_prompt": (
            "You are a specialized AI assistant for the user's uploaded domain.\n"
            "Answer questions strictly using ONLY the provided context below.\n\n"
            "Context:\n{context}"
        ),
    },
    "Fully ungrounded (temp=1.0, no constraints)": {
        "temperature": 1.0,
        "system_prompt": (
            "You are a specialized AI assistant for the user's uploaded domain.\n\n"
            "Context:\n{context}"
        ),
    },
}


st.set_page_config(page_title="Maya Bank T&C Chatbot", page_icon="💬")

# --- Maya brand theme (mint green + deep purple, matching the Maya website) ---
st.markdown(
    """
    <style>
    :root {
        --maya-green: #7EE2A8;
        --maya-green-dark: #56D191;
        --maya-purple: #3B1573;
        --maya-black: #0E0E0E;
    }

    /* App background */
    .stApp {
        background-color: #FFFFFF;
    }

    /* Titles and headers in Maya purple */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--maya-purple) !important;
        font-weight: 800 !important;
    }

    /* Caption / body text */
    p, .stMarkdown, .stCaption, label, span {
        color: #1A1A1A;
    }

    /* Sidebar styled like Maya's light panel with green accent border */
    section[data-testid="stSidebar"] {
        background-color: #F4FBF7;
        border-right: 3px solid var(--maya-green);
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: var(--maya-purple) !important;
    }

    /* Radio buttons - green accent */
    div[role="radiogroup"] label span {
        color: #1A1A1A !important;
    }

    /* Buttons styled like Maya's black pill button */
    .stButton > button,
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        background-color: var(--maya-black);
        color: #FFFFFF !important;
        border-radius: 999px;
        border: none;
        font-weight: 600;
    }
    .stButton > button {
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover,
    .stButton > button:hover p,
    .stButton > button:hover span,
    .stButton > button:hover div {
        background-color: var(--maya-purple);
        color: #FFFFFF !important;
    }

    /* Chat input box */
    div[data-testid="stChatInput"] {
        border: 1px solid var(--maya-green-dark) !important;
        border-radius: 999px !important;
    }

    /* User chat bubble - mint green */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #E9FBF1;
        border-radius: 14px;
        padding: 0.5rem;
    }

    /* Assistant chat bubble - light purple tint */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #F5F1FB;
        border-radius: 14px;
        padding: 0.5rem;
    }

    /* Links */
    a { color: var(--maya-purple) !important; }

    /* Alerts (info/warning) tinted to match palette */
    div[data-testid="stAlert"] {
        border-left: 5px solid var(--maya-green);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import base64

with open(os.path.join(os.path.dirname(__file__), "assets", "maya_logo.jpg"), "rb") as f:
    _logo_b64 = base64.b64encode(f.read()).decode()

st.markdown(
    f'<div style="text-align:center; margin-bottom:1rem;">'
    f'<img src="data:image/jpeg;base64,{_logo_b64}" width="160"></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="height:8px;background-color:#7EE2A8;border-radius:4px;margin-bottom:1.5rem;"></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 style="text-align:center;">Maya Bank Terms &amp; Conditions Chatbot</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align:center; color:#555;">'
    "A domain-specific RAG chatbot answering questions strictly from Maya Bank's "
    "General and Savings Terms &amp; Conditions. Built for ITCC508 PT-M1.</p>",
    unsafe_allow_html=True,
)

# --- Sidebar: API key + configuration ---------------------------------
with st.sidebar:
    st.header("Setup")

    # If a key is already configured via Streamlit Cloud "Secrets" or an
    # environment variable, use it silently in the backend — never place a
    # real secret into a text_input, since password fields can be revealed
    # by anyone with the "eye" icon. Only offer an empty override box.
    configured_key = os.environ.get("GROQ_API_KEY", "")
    if not configured_key:
        try:
            configured_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            configured_key = ""

    if configured_key:
        st.success("Groq API key is already configured for this app. ✅")
        override_key = st.text_input(
            "Use a different Groq API Key (optional)",
            type="password",
            value="",
            help="Leave blank to use the app's configured key.",
        )
        api_key = override_key or configured_key
    else:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value="",
            help="Get one at https://console.groq.com. Never shown or logged in plain text.",
        )

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()
    st.subheader("Grounding configuration")
    variant_name = st.radio(
        "Choose a system-prompt / temperature configuration "
        "(mirrors the notebook's Task 3 stress-test):",
        list(PROMPT_VARIANTS.keys()),
        index=0,
    )
    st.caption(
        "Use **Baseline** for normal use. The other two intentionally weaken "
        "grounding, exactly as in the lab's hallucination stress-test, so you "
        "can demo the difference live."
    )

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# --- Build (and cache) the retriever, once per data folder -------------
@st.cache_resource(show_spinner="Loading documents and building the vector index...")
def build_retriever():
    if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
        return None

    raw_documents = []
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname.lower().endswith(".txt"):
                raw_documents.extend(TextLoader(fpath, encoding="utf-8").load())
            elif fname.lower().endswith(".pdf"):
                raw_documents.extend(PyPDFLoader(fpath).load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def build_chain(retriever, variant):
    llm = ChatGroq(model_name=MODEL_NAME, temperature=variant["temperature"])
    prompt = ChatPromptTemplate.from_messages(
        [("system", variant["system_prompt"]), ("human", "{input}")]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


# --- Guard rails: data folder + API key ---------------------------------
if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
    st.warning(
        f"No files found in `./{DATA_DIR}/`. Place your `.txt`/`.pdf` source "
        f"documents there (same as the notebook) and restart the app."
    )
    st.stop()

if not os.environ.get("GROQ_API_KEY"):
    st.info("Enter your Groq API key in the sidebar to start chatting.")
    st.stop()

retriever = build_retriever()
rag_chain = build_chain(retriever, PROMPT_VARIANTS[variant_name])

# --- Chat state -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "maya_avatar.png")

# --- Frequently Asked Questions (horizontally scrollable) ---------------
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

st.markdown(
    '<p style="font-weight:700; color:#3B1573;">Frequently Asked Questions</p>',
    unsafe_allow_html=True,
)
_faq_questions = [
    "What is the PDIC insurance coverage for my account?",
    "Can Maya suspend my account without notice?",
    "How do I close my Maya savings account?",
    "What fees apply to maintaining a Maya Bank account?",
    "Is there a minimum balance requirement?",
    "How is interest calculated on my savings account?",
    "What happens if I violate the terms and conditions?",
    "How do I dispute a transaction on my account?",
]
st.markdown(
    """
    <style>
    .st-key-faq_scroll_container div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        padding-bottom: 0.5rem;
        gap: 0.5rem;
    }
    .st-key-faq_scroll_container div[data-testid="stColumn"] {
        min-width: 240px !important;
        width: 240px !important;
        flex: 0 0 auto !important;
    }
    .st-key-faq_scroll_container .stButton > button {
        background-color: #F4FBF7 !important;
        border: 1.5px solid #7EE2A8 !important;
        border-radius: 12px !important;
        white-space: normal;
        height: 100%;
        min-height: 3rem;
        font-weight: 600;
    }
    .st-key-faq_scroll_container .stButton > button p,
    .st-key-faq_scroll_container .stButton > button span,
    .st-key-faq_scroll_container .stButton > button div {
        background-color: transparent !important;
        color: #3B1573 !important;
        border: none !important;
        outline: none !important;
    }
    .st-key-faq_scroll_container .stButton > button:hover {
        background-color: #7EE2A8 !important;
        border-color: #56D191 !important;
    }
    .st-key-faq_scroll_container .stButton > button:hover p,
    .st-key-faq_scroll_container .stButton > button:hover span,
    .st-key-faq_scroll_container .stButton > button:hover div {
        color: #3B1573 !important;
    }
    .st-key-faq_scroll_container .stButton > button:focus,
    .st-key-faq_scroll_container .stButton > button:focus-visible,
    .st-key-faq_scroll_container .stButton > button:active {
        box-shadow: none !important;
        outline: none !important;
        border: 1.5px solid #7EE2A8 !important;
    }
    .st-key-faq_scroll_container .stButton > button *:focus,
    .st-key-faq_scroll_container .stButton > button *:focus-visible {
        box-shadow: none !important;
        outline: none !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.container(key="faq_scroll_container"):
    _faq_cols = st.columns(len(_faq_questions))
    for _fcol, _q in zip(_faq_cols, _faq_questions):
        if _fcol.button(_q, use_container_width=True, key=f"faq_{_q}"):
            st.session_state.pending_query = _q

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<p style="font-weight:600; margin:0.75rem 0;">{msg["content"]}</p>',
            unsafe_allow_html=True,
        )
    else:
        with st.chat_message("assistant", avatar=LOGO_PATH):
            st.markdown(msg["content"])

# --- Chat input -----------------------------------------------------------
user_query = st.chat_input("Ask about Maya Bank's terms and conditions...")
if not user_query and st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.markdown(
        f'<p style="font-weight:600; margin:0.75rem 0;">{user_query}</p>',
        unsafe_allow_html=True,
    )

    with st.chat_message("assistant", avatar=LOGO_PATH):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
