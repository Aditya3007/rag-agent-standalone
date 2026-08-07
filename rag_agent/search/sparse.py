"""Sparse (BM25) search strategy. Ported verbatim from rag-foundry's
rag/modules/search/strategies/sparse/strategy.py."""

import logging

logger = logging.getLogger(__name__)


class SparseSearchStrategy:
    """Sparse (BM25) search strategy.

    Accepts either a concrete BM25Store or a callable that returns one
    (lazy reference), so the store can be built after index construction.
    """

    def __init__(self, config, vector_store, bm25_store=None):
        self.config = config
        self.vector_store = vector_store
        self._bm25_store_ref = bm25_store

    @property
    def bm25_store(self):
        ref = self._bm25_store_ref
        if callable(ref):
            return ref()
        return ref

    def search(self, queries: list[str]):
        store = self.bm25_store
        if not store:
            logger.warning("BM25 store not available — sparse search returning empty results")
            return []

        all_results = {}

        for query in queries:
            sparse_scores = store.search(query, self.config.top_k)
            max_score = max([score for _, score in sparse_scores] + [1e-6])

            for idx, score in sparse_scores:
                chunk_idx = int(idx)
                normalized_score = score / max_score
                if chunk_idx not in all_results or normalized_score > all_results[chunk_idx]["score"]:
                    all_results[chunk_idx] = {
                        "index": chunk_idx,
                        "chunk": self.vector_store.chunks[chunk_idx],
                        "score": float(normalized_score),
                        "sparse_score": float(normalized_score),
                    }

        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)[:self.config.top_k]
        return sorted_results
