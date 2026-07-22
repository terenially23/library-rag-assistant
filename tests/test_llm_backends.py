import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm_backends import AzureOpenAIBackend, GroqBackend, OllamaBackend, OpenAIBackend, get_backend


def test_get_backend_defaults_to_groq(monkeypatch):
    monkeypatch.delenv("RAG_LLM_BACKEND", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert isinstance(get_backend(), GroqBackend)


def test_get_backend_selects_ollama(monkeypatch):
    monkeypatch.setenv("RAG_LLM_BACKEND", "ollama")
    assert isinstance(get_backend(), OllamaBackend)


def test_get_backend_selects_openai(monkeypatch):
    monkeypatch.setenv("RAG_LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert isinstance(get_backend(), OpenAIBackend)


def test_get_backend_selects_azure_openai(monkeypatch):
    monkeypatch.setenv("RAG_LLM_BACKEND", "azure_openai")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano-deployment")
    assert isinstance(get_backend(), AzureOpenAIBackend)


def test_get_backend_rejects_unknown(monkeypatch):
    monkeypatch.setenv("RAG_LLM_BACKEND", "not-a-real-backend")
    with pytest.raises(ValueError):
        get_backend()


def test_groq_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        GroqBackend()


def test_openai_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAIBackend()


def test_groq_backend_uses_groq_base_url(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    backend = GroqBackend()
    assert str(backend.client.base_url).startswith("https://api.groq.com")
    assert backend.model == "llama-3.1-8b-instant"


@pytest.mark.parametrize(
    "missing_env",
    ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"],
)
def test_azure_openai_backend_requires_each_setting(monkeypatch, missing_env):
    required = {
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
        "AZURE_OPENAI_DEPLOYMENT": "gpt-5-nano-deployment",
    }
    for key, value in required.items():
        if key == missing_env:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError):
        AzureOpenAIBackend()
