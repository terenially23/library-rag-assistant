"""
LLM backend abstraction for the RAG *generation* step -- turning retrieved
document chunks into a natural-language answer. Retrieval itself is handled
separately by Azure AI Search (app/search_backend.py); this only covers
"write the answer given this context", selected via environment variables,
same pattern as the sibling nl2sql-transactions-assistant project.

    RAG_LLM_BACKEND=groq (default)
        GROQ_API_KEY=...             (required; free key from console.groq.com)
        GROQ_MODEL=llama-3.1-8b-instant  (default)

    RAG_LLM_BACKEND=openai
        OPENAI_API_KEY=...           (required)
        OPENAI_MODEL=gpt-4o-mini     (default)

    RAG_LLM_BACKEND=azure_openai
        AZURE_OPENAI_API_KEY=...       (required)
        AZURE_OPENAI_ENDPOINT=...      (required; e.g. https://<resource>.openai.azure.com/)
        AZURE_OPENAI_DEPLOYMENT=...    (required; your deployment name, not the base model name)
        AZURE_OPENAI_API_VERSION=2024-10-21  (default)

    RAG_LLM_BACKEND=ollama
        OLLAMA_MODEL=llama3.1        (default: llama3.1)
        OLLAMA_HOST=http://localhost:11434  (default)
"""
import os
from abc import ABC, abstractmethod

import requests


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw text completion for the given prompts."""


class OllamaBackend(LLMBackend):
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"]


class _ChatCompletionsBackend(LLMBackend):
    """Shared `generate` for any backend that speaks the OpenAI chat-completions API."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


class _OpenAICompatibleBackend(_ChatCompletionsBackend):
    """
    Shared implementation for OpenAI and any OpenAI-API-compatible provider
    (e.g. Groq), which differ only in base URL, default model, and which
    env var holds the API key.
    """

    def __init__(self, model: str, api_key: str | None, api_key_env_var: str, base_url: str | None = None):
        from openai import OpenAI

        resolved_key = api_key or os.environ.get(api_key_env_var)
        if not resolved_key:
            raise RuntimeError(f"{api_key_env_var} is not set")
        self.model = model
        self.client = OpenAI(api_key=resolved_key, base_url=base_url)


class OpenAIBackend(_OpenAICompatibleBackend):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(
            model=model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            api_key_env_var="OPENAI_API_KEY",
        )


class GroqBackend(_OpenAICompatibleBackend):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(
            model=model or os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=api_key,
            api_key_env_var="GROQ_API_KEY",
            base_url="https://api.groq.com/openai/v1",
        )


class AzureOpenAIBackend(_ChatCompletionsBackend):
    """
    Azure OpenAI Service backend. Requests are routed by *deployment name*
    (chosen when deploying a model in your Azure resource), not by the
    underlying base model name.
    """

    def __init__(
        self,
        deployment: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
    ):
        from openai import AzureOpenAI

        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        resolved_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        resolved_deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not resolved_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not set")
        if not resolved_endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set")
        if not resolved_deployment:
            raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is not set")

        self.model = resolved_deployment
        self.client = AzureOpenAI(
            api_key=resolved_key,
            azure_endpoint=resolved_endpoint,
            api_version=api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )


def get_backend() -> LLMBackend:
    backend_name = os.environ.get("RAG_LLM_BACKEND", "groq").lower()
    if backend_name == "ollama":
        return OllamaBackend()
    if backend_name == "groq":
        return GroqBackend()
    if backend_name == "openai":
        return OpenAIBackend()
    if backend_name == "azure_openai":
        return AzureOpenAIBackend()
    raise ValueError(
        f"Unknown RAG_LLM_BACKEND: {backend_name!r} (expected 'ollama', 'groq', 'openai', or 'azure_openai')"
    )
