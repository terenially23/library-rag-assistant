# Library RAG Assistant

Ask plain-English questions about a university library's policies and get
back a grounded answer with the source documents shown alongside it — a
small retrieval-augmented generation (RAG) demo built on Azure AI Search.

## The problem

Library/IT help-desk staff field a lot of repetitive policy questions
("how many books can I borrow?", "do I need a VPN off-campus?") that are
already answered somewhere in the library's own documentation, but finding
the right page takes longer than just asking someone. This project answers
those questions directly from the source documents, with citations, so
answers stay traceable back to the actual policy rather than being an
unverifiable chat response.

## What it does

- Takes a natural-language question about the library.
- Embeds the question locally (Hugging Face `sentence-transformers`, no
  cloud call needed for this step) and retrieves the most relevant
  passages from an **Azure AI Search** index using hybrid (keyword +
  vector) search.
- Sends the retrieved passages to an LLM (Groq by default — free and
  fast — with OpenAI, Azure OpenAI, or local Ollama also supported) and
  asks it to answer **using only that context**.
- Displays the answer with the source document(s) it was grounded in, so
  the user can check the answer traces back to a real policy.

## Why this design

Splitting retrieval and generation onto separate, independently-swappable
backends means the expensive/quota-limited part (LLM compute) and the
cheap/reliable part (search indexing) aren't coupled — you can run the
whole pipeline on entirely free infrastructure (Azure AI Search's F0 free
tier + Groq's free API + a local embedding model), which matters for a
demo project with no production budget. It also means grounding is
enforced structurally: the LLM only ever sees the retrieved passages, not
the whole corpus, so it can't answer from outside knowledge even if asked.

## Project structure

```
library-rag-assistant/
├── data_gen/generate_documents.py   # writes the synthetic library document corpus
├── data/documents/                  # generated Markdown docs (policies, FAQs, subject guides)
├── app/chunking.py                  # splits documents into retrieval-sized chunks
├── app/embeddings.py                # local embedding via Hugging Face sentence-transformers
├── app/search_backend.py            # Azure AI Search index creation, upload, hybrid query
├── app/llm_backends.py              # Groq / OpenAI / Azure OpenAI / Ollama generation backend
├── app/rag.py                       # retrieval -> grounded prompt -> generation pipeline
├── app/streamlit_app.py             # chat UI
├── ingest.py                        # one-off script: chunk, embed, index, upload
├── tests/                           # unit tests (no live Azure/LLM credentials required)
└── requirements.txt
```

## How to run

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# 1. generate the synthetic document corpus
python data_gen/generate_documents.py
```

### 2. Provision Azure AI Search (free tier)

1. In the [Azure Portal](https://portal.azure.com), create an **Azure AI
   Search** resource — pricing tier **Free (F0)** (50MB storage, 3 indexes,
   no time limit, no quota approval needed).
2. From the resource's **Keys** page, copy the **URL** and an **admin key**.
3. Set env vars:

```bash
export AZURE_SEARCH_ENDPOINT=https://<your-service-name>.search.windows.net
export AZURE_SEARCH_API_KEY=...
export AZURE_SEARCH_INDEX=library-docs   # optional, this is the default
```

### 3. Ingest the documents

```bash
python ingest.py
```

This chunks the documents, embeds them locally, creates the index (if it
doesn't already exist), and uploads the chunks.

### 4. Choose a generation backend and run the app

```bash
# free, hosted (Groq) -- the default
export GROQ_API_KEY=gsk_...          # free key from console.groq.com
streamlit run app/streamlit_app.py
```

Other backends (see `app/llm_backends.py` for full env var lists):

```bash
export RAG_LLM_BACKEND=openai        # or azure_openai, or ollama
export OPENAI_API_KEY=sk-...
streamlit run app/streamlit_app.py
```

Run the tests with:

```bash
pytest tests/
```

## Example questions

- "How many books can I borrow as a postgraduate student?"
- "What referencing style does the Law school use?"
- "Can I book a study room for one person?"
- "Do I need a VPN to access journal articles from home?"
- "How do I get an alternative format for a reading list text?"

## Data & scope note

All documents (library policies, FAQs, subject guides) are synthetic and
invented for a fictional "Whitmore University" — not a real institution's
policies. This is a demonstration project, not a production help-desk
system.

## Future work

- Add re-ranking (e.g. a cross-encoder) on top of the initial hybrid
  search results for better precision on ambiguous questions.
- Support conversation history so follow-up questions ("what about for
  staff?") resolve against the previous turn's topic.
- Add a feedback control (thumbs up/down) on answers, logged alongside the
  retrieved sources, to build a dataset for evaluating retrieval quality.
- Chunk by semantic similarity rather than by paragraph for longer,
  less uniformly-structured source documents.
- Add access-controlled indexing (per-audience filtering) for a corpus
  that mixes public and staff-only documents.
