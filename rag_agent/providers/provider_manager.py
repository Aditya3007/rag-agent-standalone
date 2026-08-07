"""Provider construction + process-wide registry.

Ported/consolidated from rag-foundry's providers/provider_factory.py and
providers/provider_manager.py, trimmed to the two provider types the
pinned configs actually use (groq, ollama).
"""

import os

from rag_agent.config import ProviderConfig, ProviderType
from rag_agent.providers.groq_provider import GroqProvider
from rag_agent.providers.ollama_provider import OllamaProvider


def _resolve_api_keys(provider_name: str, config: ProviderConfig) -> list[str]:
    """Resolve credentials for a provider from its configured environment
    variable. A single env var may contain multiple comma-separated keys."""
    explicit_keys = config.params.get("api_keys")
    if explicit_keys:
        return list(explicit_keys)

    env_var = config.api_key_env
    if not env_var:
        raise ValueError(
            f"Provider '{provider_name}' has no 'api_key_env' configured "
            f"and no 'params.api_keys' fallback."
        )

    env_value = os.getenv(env_var)
    if env_value:
        return [key.strip() for key in env_value.split(",") if key.strip()]

    raise ValueError(
        f"No API keys found for provider '{provider_name}'. "
        f"Expected environment variable '{env_var}' to be set "
        f"(or 'params.api_keys' to be provided)."
    )


def _create_provider(provider_name: str, provider_type: ProviderType, config: ProviderConfig):
    provider_type = ProviderType(provider_type)

    if provider_type == ProviderType.OLLAMA:
        host = config.params.get("host")
        kwargs = {"host": host} if host is not None else {}
        return OllamaProvider(**kwargs)

    if provider_type == ProviderType.GROQ:
        api_keys = _resolve_api_keys(provider_name, config)
        kwargs = {"api_keys": api_keys}
        cooldown_seconds = config.params.get("cooldown_seconds")
        if cooldown_seconds is not None:
            kwargs["cooldown_seconds"] = cooldown_seconds
        return GroqProvider(**kwargs)

    raise ValueError(f"Unsupported provider type: {provider_type}")


class ProviderManager:
    """Process-wide singleton registry of constructed providers, keyed by name."""

    _providers: dict = {}

    @classmethod
    def register(cls, provider_name: str, provider_type, config):
        """Idempotent: an already-registered provider is reused."""
        if provider_name in cls._providers:
            return
        cls._providers[provider_name] = _create_provider(provider_name, provider_type, config)

    @classmethod
    def get_provider(cls, provider_name: str):
        if provider_name not in cls._providers:
            raise ValueError(f"Provider '{provider_name}' not registered")
        return cls._providers[provider_name]

    @classmethod
    def clear(cls):
        cls._providers.clear()
