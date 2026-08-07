"""Ollama provider (OpenAI-compatible local server). Ported from
rag-foundry's providers/ollama/ollama_provider.py."""

from openai import OpenAI

from rag_agent.providers.base import BaseLLMProvider, ProviderInitializationException


class OllamaProvider(BaseLLMProvider):

    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, host: str = DEFAULT_HOST):
        self.host = host
        try:
            self.client = OpenAI(api_key="ollama", base_url=f"{self.host.rstrip('/')}/v1")
        except Exception as exc:
            raise ProviderInitializationException(f"Failed to initialize Ollama client: {exc}") from exc

    def generate(self, model: str, messages: list, **kwargs):
        return self.client.chat.completions.create(model=model, messages=messages, **kwargs)

    def health(self) -> dict:
        return {"provider": "ollama", "host": self.host}
