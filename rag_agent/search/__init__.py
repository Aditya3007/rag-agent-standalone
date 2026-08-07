from rag_agent.search.dense import DenseSearchStrategy
from rag_agent.search.sparse import SparseSearchStrategy
from rag_agent.search.bm25_store import BM25Store
from rag_agent.config import SearchType


def create_search_strategy(search_config, embedder, vector_store, bm25_store):
    if search_config.type == SearchType.DENSE:
        return DenseSearchStrategy(search_config.config, embedder=embedder, vector_store=vector_store)
    if search_config.type == SearchType.SPARSE:
        return SparseSearchStrategy(search_config.config, vector_store=vector_store, bm25_store=bm25_store)
    raise ValueError(f"Unsupported search type: {search_config.type}")


__all__ = ["create_search_strategy", "DenseSearchStrategy", "SparseSearchStrategy", "BM25Store"]
