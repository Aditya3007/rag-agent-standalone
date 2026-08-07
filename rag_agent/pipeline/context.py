"""Retrieval pipeline context. Ported verbatim from rag-foundry's
rag/pipeline/context.py."""

from dataclasses import dataclass, field


@dataclass
class RetrievalContext:
    """Shared state passed between retrieval pipeline stages."""

    query: str
    dense_queries: list[str] = field(default_factory=list)
    sparse_queries: list[str] = field(default_factory=list)
    search_results: list[list[dict]] = field(default_factory=list)
    fused_results: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
