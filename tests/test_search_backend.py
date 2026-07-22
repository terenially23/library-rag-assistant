import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.search_backend import SearchBackend


def test_requires_endpoint(monkeypatch):
    monkeypatch.delenv("AZURE_SEARCH_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match="AZURE_SEARCH_ENDPOINT"):
        SearchBackend()


def test_requires_api_key(monkeypatch):
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://example.search.windows.net")
    monkeypatch.delenv("AZURE_SEARCH_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AZURE_SEARCH_API_KEY"):
        SearchBackend()


def test_defaults_index_name(monkeypatch):
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://example.search.windows.net")
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "test-key")
    monkeypatch.delenv("AZURE_SEARCH_INDEX", raising=False)
    backend = SearchBackend()
    assert backend.index_name == "library-docs"


def test_explicit_args_override_env(monkeypatch):
    monkeypatch.delenv("AZURE_SEARCH_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_SEARCH_API_KEY", raising=False)
    backend = SearchBackend(
        endpoint="https://explicit.search.windows.net",
        api_key="explicit-key",
        index_name="explicit-index",
    )
    assert backend.endpoint == "https://explicit.search.windows.net"
    assert backend.index_name == "explicit-index"
