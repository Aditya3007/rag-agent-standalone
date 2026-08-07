"""BM25 sparse index wrapper using rank_bm25. Ported verbatim from
rag-foundry's rag/modules/search/bm25_store.py."""

import logging
from typing import List, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Store:
    """Lightweight BM25 index over a list of text chunks."""

    def __init__(self, chunks):
        self.chunks = chunks
        tokenized = [self._tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(tokenized)
        logger.info("BM25Store built with %d chunks", len(chunks))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(idx, score) for idx, score in ranked if score > 0]
