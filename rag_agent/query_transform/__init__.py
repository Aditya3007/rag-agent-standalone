from rag_agent.query_transform.base import QueryTransformResult
from rag_agent.query_transform.noop import NoOpQueryTransformStrategy
from rag_agent.query_transform.multi_query import MultiQueryQueryTransformStrategy
from rag_agent.config import QueryTransformType


def create_query_transform(query_transform_config, provider=None):
    if query_transform_config is None or query_transform_config.type == QueryTransformType.NOOP:
        config = query_transform_config.config if query_transform_config else None
        return NoOpQueryTransformStrategy(config)
    if query_transform_config.type == QueryTransformType.MULTI_QUERY:
        return MultiQueryQueryTransformStrategy(query_transform_config.config, provider=provider)
    raise ValueError(f"Unsupported query transform type: {query_transform_config.type}")


__all__ = [
    "create_query_transform",
    "QueryTransformResult",
    "NoOpQueryTransformStrategy",
    "MultiQueryQueryTransformStrategy",
]
