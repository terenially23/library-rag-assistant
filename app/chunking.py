"""
Splits the synthetic library documents into retrieval-sized chunks.

Each document is a short Markdown file with a single '# Heading' line
followed by one or more paragraphs. Chunking by paragraph keeps each chunk
focused on one idea (loan limits vs. renewal rules vs. fines, say) which
gives better retrieval precision than indexing the whole document as one
block, while staying simple enough not to need a real text-splitting
library for a corpus this small.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    text: str


def _heading_and_body(raw: str) -> tuple[str, str]:
    lines = raw.strip().splitlines()
    if lines and lines[0].startswith("#"):
        return lines[0].lstrip("#").strip(), "\n".join(lines[1:]).strip()
    return "", raw.strip()


def chunk_document(path: Path) -> list[Chunk]:
    heading, body = _heading_and_body(path.read_text(encoding="utf-8"))
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

    chunks = []
    for i, paragraph in enumerate(paragraphs):
        chunks.append(
            Chunk(
                id=f"{path.stem}-{i}",
                source=path.name,
                heading=heading,
                text=paragraph,
            )
        )
    return chunks


def chunk_directory(directory: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_document(path))
    return chunks
