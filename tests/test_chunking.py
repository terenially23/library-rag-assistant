import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import chunk_directory, chunk_document


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_chunk_document_splits_on_paragraphs():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(
            Path(tmp),
            "example.md",
            "# Example Heading\n\nFirst paragraph.\n\nSecond paragraph.\n",
        )
        chunks = chunk_document(path)

    assert len(chunks) == 2
    assert all(c.heading == "Example Heading" for c in chunks)
    assert chunks[0].text == "First paragraph."
    assert chunks[1].text == "Second paragraph."
    assert chunks[0].source == "example.md"
    assert chunks[0].id == "example-0"
    assert chunks[1].id == "example-1"


def test_chunk_document_handles_missing_heading():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp), "no-heading.md", "Just a paragraph with no heading line.")
        chunks = chunk_document(path)

    assert len(chunks) == 1
    assert chunks[0].heading == ""
    assert chunks[0].text == "Just a paragraph with no heading line."


def test_chunk_directory_processes_all_markdown_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write(tmp_path, "a.md", "# A\n\nContent A.")
        _write(tmp_path, "b.md", "# B\n\nContent B.")
        _write(tmp_path, "ignore.txt", "should not be picked up")

        chunks = chunk_directory(tmp_path)

    sources = {c.source for c in chunks}
    assert sources == {"a.md", "b.md"}
