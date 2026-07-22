"""
RAG pipeline: embed the question, retrieve grounding context from Azure AI
Search, and ask the LLM backend to answer using only that context -- with
the retrieved sources always returned alongside the answer so the user can
verify what the answer was actually grounded in, rather than trusting the
model's word for it.
"""
from dataclasses import dataclass

from app.embeddings import embed_query
from app.llm_backends import LLMBackend, get_backend
from app.search_backend import SearchBackend

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a university \
library's services, using ONLY the context below -- do not use any outside knowledge.

Context:
{context}

Rules:
- If the answer isn't contained in the context, say you don't know rather than guessing.
- Keep answers concise (2-4 sentences).
- Do not invent policy details (fees, hours, limits) that aren't stated in the context.
"""


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[dict]
    error: str | None = None


def format_context(hits: list[dict]) -> str:
    """Turn search hits into a numbered context block the prompt can reference."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] ({hit['heading']}) {hit['content']}")
    return "\n\n".join(blocks)


def dedupe_sources(hits: list[dict]) -> list[dict]:
    """Collapse multiple chunk hits from the same document into one source entry."""
    seen = {}
    for hit in hits:
        key = hit["source"]
        if key not in seen:
            seen[key] = {"source": hit["source"], "heading": hit["heading"]}
    return list(seen.values())


def ask(
    question: str,
    search_backend: SearchBackend | None = None,
    llm_backend: LLMBackend | None = None,
    top_k: int = 4,
) -> RAGResult:
    search_backend = search_backend or SearchBackend()
    llm_backend = llm_backend or get_backend()

    query_vector = embed_query(question)
    hits = search_backend.search(question, query_vector, top_k=top_k)

    if not hits:
        return RAGResult(question=question, answer="", sources=[], error="No relevant documents found")

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=format_context(hits))
    answer = llm_backend.generate(system_prompt, question)

    return RAGResult(question=question, answer=answer, sources=dedupe_sources(hits), error=None)
