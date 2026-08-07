from rag_agent.vectorstore.faiss_store import FaissVectorStoreStrategy
from rag_agent.config import VectorStoreType

_STRATEGIES = {
    VectorStoreType.FAISS: FaissVectorStoreStrategy,
}


def create_vector_store(vector_store_config):
    strategy_cls = _STRATEGIES[vector_store_config.type]
    return strategy_cls(vector_store_config.config)


__all__ = ["create_vector_store", "FaissVectorStoreStrategy"]
