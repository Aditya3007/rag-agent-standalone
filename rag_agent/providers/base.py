"""Base LLM provider interface. Ported from rag-foundry's providers/base/*.py."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


class BaseLLMProvider(ABC):

    @abstractmethod
    def generate(self, model: str, messages: list, **kwargs):
        """Generate a response from the model."""
        pass

    @abstractmethod
    def health(self) -> dict:
        """Provider diagnostics."""
        pass


@dataclass
class KeyState:
    api_key: str
    cooldown_until: datetime | None = None

    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_429s: int = 0

    @property
    def available(self) -> bool:
        if self.cooldown_until is None:
            return True
        return datetime.now() >= self.cooldown_until


class ProviderException(Exception):
    """Base provider exception."""


class ProviderInitializationException(ProviderException):
    """Raised when provider initialization fails."""


class AllKeysExhaustedException(ProviderException):
    """Raised when no API keys are currently available."""
