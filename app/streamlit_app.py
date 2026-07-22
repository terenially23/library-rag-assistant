"""
Chat UI for the library RAG assistant. Run with:

    streamlit run app/streamlit_app.py

Requires the Azure AI Search index to already be populated (see README,
`python ingest.py`). Generation backend is selected via RAG_LLM_BACKEND
(default: groq).

MAX_QUERIES_PER_SESSION caps how many questions a single browser session can
ask -- relevant if this is deployed publicly against a shared, rate-limited
free API key.
"""
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import ask

MAX_QUERIES_PER_SESSION = 15

st.set_page_config(page_title="Library RAG Assistant", layout="wide")

st.title("Library RAG Assistant")
st.caption(
    "Ask questions about a synthetic university library's policies and services. "
    "Answers are grounded in retrieved documents, which are always shown alongside the answer."
)

backend_name = os.environ.get("RAG_LLM_BACKEND", "groq")
with st.sidebar:
    st.subheader("Generation backend")
    st.write(f"Active: `{backend_name}`")
    if backend_name == "groq":
        st.write(f"Model: `{os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant')}`")
        st.caption("Free hosted API (console.groq.com). Requires GROQ_API_KEY.")
    elif backend_name == "azure_openai":
        st.write(f"Deployment: `{os.environ.get('AZURE_OPENAI_DEPLOYMENT', '(not set)')}`")
        st.caption("Azure OpenAI Service. Requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT.")
    elif backend_name == "ollama":
        st.write(f"Model: `{os.environ.get('OLLAMA_MODEL', 'llama3.1')}`")
        st.caption("Free, local. Requires `ollama serve` running.")
    else:
        st.write(f"Model: `{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}`")
        st.caption("Requires OPENAI_API_KEY to be set.")

    st.subheader("Retrieval")
    st.caption(
        f"Azure AI Search index: `{os.environ.get('AZURE_SEARCH_INDEX', 'library-docs')}`\n\n"
        "Embeddings computed locally via sentence-transformers."
    )

    st.subheader("Try asking")
    st.markdown(
        "- How many books can I borrow as a postgraduate student?\n"
        "- What referencing style does the Law school use?\n"
        "- Can I book a study room for one person?\n"
        "- Do I need a VPN to access journal articles from home?\n"
        "- How do I get an alternative format for a reading list text?"
    )

if "history" not in st.session_state:
    st.session_state.history = []

queries_used = sum(1 for kind, *_ in st.session_state.history if kind != "error")
queries_left = MAX_QUERIES_PER_SESSION - queries_used

if queries_left <= 0:
    st.warning(
        f"You've reached the {MAX_QUERIES_PER_SESSION}-question limit for this session. "
        "Refresh the page to reset, or clone the repo to run it locally with your own key."
    )
    question = None
else:
    question = st.chat_input("Ask a question about the library...")
    st.caption(f"{queries_left} question(s) left in this session.")

if question:
    with st.spinner("Retrieving context and generating an answer..."):
        try:
            result = ask(question)
        except Exception as e:  # noqa: BLE001 - surface backend/connection errors in the UI
            result = None
            st.session_state.history.append(("error", question, str(e), None))
    if result is not None:
        st.session_state.history.append(
            ("ok" if result.error is None else "no_results", question, result.answer, result)
        )

for kind, q, answer, result in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        if kind == "error":
            st.error(f"Could not complete the request: {answer}")
            st.info(
                "Check that the Azure AI Search index has been populated (`python ingest.py`) "
                "and that the generation backend's API key is set."
            )
        elif kind == "no_results":
            st.warning("No relevant documents were found for that question.")
        else:
            st.write(result.answer)
            with st.expander(f"Sources ({len(result.sources)})"):
                for src in result.sources:
                    st.markdown(f"- **{src['heading']}** (`{src['source']}`)")
