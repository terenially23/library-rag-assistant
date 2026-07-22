"""
One-off ingestion script: chunk the synthetic library documents, embed them
locally, create/update the Azure AI Search index, and upload the chunks.

Run once after provisioning the Azure AI Search resource (see README):

    python ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.chunking import chunk_directory
from app.embeddings import embed_texts
from app.search_backend import SearchBackend

DOCS_DIR = Path(__file__).resolve().parent / "data" / "documents"


def main() -> None:
    chunks = chunk_directory(DOCS_DIR)
    print(f"Chunked {len(chunks)} passages from {DOCS_DIR}")

    print("Embedding locally (sentence-transformers)...")
    vectors = embed_texts([c.text for c in chunks])

    backend = SearchBackend()
    print(f"Ensuring index {backend.index_name!r} exists...")
    backend.ensure_index()

    print("Uploading chunks...")
    backend.upload_chunks(chunks, vectors)

    print("Done.")


if __name__ == "__main__":
    main()
