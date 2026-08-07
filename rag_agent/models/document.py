"""Canonical document model. Ported verbatim from rag-foundry's rag/models/document.py."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Document:
    title: str
    content: str
    metadata: Dict[str, Any]
