"""No-op query transform strategy. Ported verbatim from rag-foundry's
rag/modules/query_transform/strategies/noop/strategy.py."""

from rag_agent.query_transform.base import QueryTransformResult


class NoOpQueryTransformStrategy:
    """No-op query transform: passes query through unchanged."""

    def __init__(self, config):
        self.config = config

    def transform(self, query: str) -> QueryTransformResult:
        return QueryTransformResult(dense_queries=[query], sparse_queries=[query])
