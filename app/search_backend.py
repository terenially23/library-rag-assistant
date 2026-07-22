"""
Azure AI Search backend: index creation, chunk upload, and hybrid
(keyword + vector) retrieval.

This is deliberately the *only* piece of the pipeline that needs a live
Azure resource -- embeddings are computed locally (app/embeddings.py) and
generation goes through the same pluggable LLMBackend used elsewhere
(app/llm_backends.py), so an Azure AI Search free-tier (F0) resource is
the one thing you need to provision to run this end-to-end.

Required env vars:
    AZURE_SEARCH_ENDPOINT   e.g. https://<service-name>.search.windows.net
    AZURE_SEARCH_API_KEY    admin key, from the resource's "Keys" page
    AZURE_SEARCH_INDEX      default: "library-docs"
"""
import os

from app.chunking import Chunk
from app.embeddings import EMBEDDING_DIMENSIONS

VECTOR_FIELD = "content_vector"
ALGORITHM_NAME = "library-docs-hnsw"
PROFILE_NAME = "library-docs-vector-profile"


class SearchBackend:
    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        index_name: str | None = None,
    ):
        self.endpoint = endpoint or os.environ.get("AZURE_SEARCH_ENDPOINT")
        self.api_key = api_key or os.environ.get("AZURE_SEARCH_API_KEY")
        self.index_name = index_name or os.environ.get("AZURE_SEARCH_INDEX", "library-docs")

        if not self.endpoint:
            raise RuntimeError("AZURE_SEARCH_ENDPOINT is not set")
        if not self.api_key:
            raise RuntimeError("AZURE_SEARCH_API_KEY is not set")

        from azure.core.credentials import AzureKeyCredential

        self._credential = AzureKeyCredential(self.api_key)

    def _index_client(self):
        from azure.search.documents.indexes import SearchIndexClient

        return SearchIndexClient(endpoint=self.endpoint, credential=self._credential)

    def _search_client(self):
        from azure.search.documents import SearchClient

        return SearchClient(endpoint=self.endpoint, index_name=self.index_name, credential=self._credential)

    def ensure_index(self) -> None:
        """Create the index if it doesn't already exist (safe to call repeatedly)."""
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="heading", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name=VECTOR_FIELD,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=EMBEDDING_DIMENSIONS,
                vector_search_profile_name=PROFILE_NAME,
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name=ALGORITHM_NAME)],
            profiles=[
                VectorSearchProfile(name=PROFILE_NAME, algorithm_configuration_name=ALGORITHM_NAME),
            ],
        )

        index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
        self._index_client().create_or_update_index(index)

    def upload_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        documents = [
            {
                "id": chunk.id,
                "source": chunk.source,
                "heading": chunk.heading,
                "content": chunk.text,
                VECTOR_FIELD: vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        self._search_client().upload_documents(documents=documents)

    def search(self, question: str, query_vector: list[float], top_k: int = 4) -> list[dict]:
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields=VECTOR_FIELD)
        results = self._search_client().search(
            search_text=question,
            vector_queries=[vector_query],
            select=["id", "source", "heading", "content"],
            top=top_k,
        )
        return [dict(r) for r in results]
