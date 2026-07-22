import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag import dedupe_sources, format_context, ask


class FakeSearchBackend:
    def __init__(self, hits):
        self._hits = hits
        self.last_query_vector = None

    def search(self, question, query_vector, top_k=4):
        self.last_query_vector = query_vector
        return self._hits


class FakeLLMBackend:
    def __init__(self, response="a generated answer"):
        self.response = response
        self.last_system_prompt = None
        self.last_user_prompt = None

    def generate(self, system_prompt, user_prompt):
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.response


SAMPLE_HITS = [
    {"id": "borrowing-policy-0", "source": "borrowing-policy.md", "heading": "Borrowing Policy", "content": "Undergrads may borrow 12 items."},
    {"id": "borrowing-policy-1", "source": "borrowing-policy.md", "heading": "Borrowing Policy", "content": "Fines are 10p per day."},
    {"id": "opening-hours-0", "source": "opening-hours.md", "heading": "Opening Hours", "content": "Open 08:00-23:00."},
]


def test_format_context_numbers_and_labels_each_hit():
    context = format_context(SAMPLE_HITS)
    assert "[1] (Borrowing Policy) Undergrads may borrow 12 items." in context
    assert "[3] (Opening Hours) Open 08:00-23:00." in context


def test_dedupe_sources_collapses_same_document():
    sources = dedupe_sources(SAMPLE_HITS)
    assert sources == [
        {"source": "borrowing-policy.md", "heading": "Borrowing Policy"},
        {"source": "opening-hours.md", "heading": "Opening Hours"},
    ]


def test_ask_returns_answer_and_deduped_sources():
    search = FakeSearchBackend(SAMPLE_HITS)
    llm = FakeLLMBackend(response="You can borrow up to 12 items.")

    result = ask("How many books can I borrow?", search_backend=search, llm_backend=llm)

    assert result.error is None
    assert result.answer == "You can borrow up to 12 items."
    assert len(result.sources) == 2
    assert "Undergrads may borrow 12 items." in llm.last_system_prompt


def test_ask_returns_error_when_no_hits():
    search = FakeSearchBackend([])
    llm = FakeLLMBackend()

    result = ask("Some question with no matches", search_backend=search, llm_backend=llm)

    assert result.error == "No relevant documents found"
    assert result.sources == []
    assert llm.last_user_prompt is None  # LLM should never be called
