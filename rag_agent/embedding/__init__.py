from rag_agent.embedding.sentence_transformer import SentenceTransformerEmbeddingStrategy
from rag_agent.config import EmbeddingType

_STRATEGIES = {
    EmbeddingType.SENTENCE_TRANSFORMER: SentenceTransformerEmbeddingStrategy,
}


def create_embedder(embedding_config):
    strategy_cls = _STRATEGIES[embedding_config.type]
    return strategy_cls(embedding_config.config)


__all__ = ["create_embedder", "SentenceTransformerEmbeddingStrategy"]
