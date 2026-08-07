"""Query transform result type. Ported from rag-foundry's
rag/modules/query_transform/base.py."""

from dataclasses import dataclass, field


@dataclass
class QueryTransformResult:
    """Result of query transformation with separate dense and sparse query lists."""
    dense_queries: list[str]
    sparse_queries: list[str]
    metadata: dict = field(default_factory=dict)
