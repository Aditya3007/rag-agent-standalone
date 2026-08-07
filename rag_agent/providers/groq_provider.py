"""Groq provider. Ported from rag-foundry's providers/groq/groq_provider.py.

Supports rotating across multiple comma-separated API keys with per-key
cooldown on 429 rate limits.
"""

from datetime import datetime, timedelta

from groq import Groq, RateLimitError

from rag_agent.providers.base import (
    BaseLLMProvider,
    KeyState,
    AllKeysExhaustedException,
    ProviderInitializationException,
)


class GroqProvider(BaseLLMProvider):

    DEFAULT_COOLDOWN_SECONDS = 60

    def __init__(self, api_keys: list[str], cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS):
        if not api_keys:
            raise ProviderInitializationException("At least one API key is required.")

        self.cooldown_seconds = cooldown_seconds
        self.current_index = 0
        self.keys = [KeyState(api_key=key) for key in api_keys]
        self.clients = {}

        try:
            for key in api_keys:
                self.clients[key] = Groq(api_key=key)
        except Exception as exc:
            raise ProviderInitializationException(f"Failed to initialize Groq clients: {exc}") from exc

    def _mark_rate_limited(self, key_state: KeyState, retry_after_seconds: int | None = None):
        cooldown_seconds = (
            retry_after_seconds if retry_after_seconds is not None else self.cooldown_seconds
        )
        key_state.cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
        key_state.total_429s += 1

    @staticmethod
    def _retry_after_seconds(exc: RateLimitError):
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None

        if retry := headers.get("retry-after"):
            return int(float(retry))

        if reset := headers.get("x-ratelimit-reset-requests"):
            if reset.endswith("ms"):
                return max(1, round(float(reset[:-2]) / 1000))
            if reset.endswith("s"):
                return int(float(reset[:-1]))

        return None

    def _get_next_available_key(self) -> KeyState:
        total_keys = len(self.keys)
        for offset in range(total_keys):
            idx = (self.current_index + offset) % total_keys
            key_state = self.keys[idx]
            if key_state.available:
                self.current_index = idx
                return key_state
        raise AllKeysExhaustedException("All Groq API keys are currently rate limited.")

    def generate(self, model: str, messages: list, **kwargs):
        attempted_keys = set()

        while len(attempted_keys) < len(self.keys):
            key_state = self._get_next_available_key()

            if key_state.api_key in attempted_keys:
                break
            attempted_keys.add(key_state.api_key)

            client = self.clients[key_state.api_key]

            try:
                key_state.total_requests += 1
                response = client.chat.completions.create(model=model, messages=messages, **kwargs)
                key_state.total_successes += 1
                return response

            except RateLimitError as exc:
                key_state.total_failures += 1
                retry_after = self._retry_after_seconds(exc)
                self._mark_rate_limited(key_state=key_state, retry_after_seconds=retry_after)

            except Exception:
                key_state.total_failures += 1
                raise

        raise AllKeysExhaustedException("No available Groq API keys.")

    def health(self) -> dict:
        return {
            "provider": "groq",
            "current_index": self.current_index,
            "keys": [
                {
                    "key_suffix": key.api_key[-4:],
                    "available": key.available,
                    "cooldown_until": key.cooldown_until,
                    "requests": key.total_requests,
                    "successes": key.total_successes,
                    "failures": key.total_failures,
                    "429s": key.total_429s,
                }
                for key in self.keys
            ],
        }
