"""FAISS vector store strategy. Ported verbatim from rag-foundry's
vectorstore/strategies/faiss/strategy.py."""

import faiss
import numpy as np


class FaissVectorStoreStrategy:
    """FAISS vector store strategy."""

    def __init__(self, config):
        self.config = config
        self.index = faiss.IndexFlatIP(self.config.dimension)
        self.chunks = []

    def add(self, embeddings, chunks):
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query_embedding, top_k):
        return self.index.search(query_embedding, top_k)
