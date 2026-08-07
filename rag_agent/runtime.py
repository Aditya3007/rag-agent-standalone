"""Lazily-loaded, process-wide model cache. Ported verbatim from
rag-foundry's rag/runtime/model_cache.py."""

from functools import lru_cache

from sentence_transformers import CrossEncoder, SentenceTransformer


@lru_cache(maxsize=None)
def get_sentence_transformer(model_name: str):
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


@lru_cache(maxsize=None)
def get_cross_encoder(model_name: str):
    print(f"Loading reranker model: {model_name}")
    return CrossEncoder(model_name)
