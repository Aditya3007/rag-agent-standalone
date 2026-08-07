"""Base chunking strategy. Ported from rag-foundry's rag/modules/chunking/base.py."""

from abc import ABC, abstractmethod
import re

from rag_agent.models.document import Document
from rag_agent.models.chunk import Chunk


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        pass

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]
