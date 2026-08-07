"""Chunk model. Ported verbatim from rag-foundry's rag/models/chunk.py."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
