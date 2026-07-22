"""
Local embedding model via Hugging Face sentence-transformers.

Embedding runs locally and for free -- no Azure OpenAI quota required for
this part of the pipeline, only Azure AI Search (a storage/indexing
service, not a compute one) needs an Azure resource. `all-MiniLM-L6-v2` is
small (~80MB), fast on CPU, and produces 384-dimensional vectors, which is
what the Azure AI Search index schema in search_backend.py is built around.
"""
import os
from functools import lru_cache

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS = 384


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    return model.encode(list(texts), normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
