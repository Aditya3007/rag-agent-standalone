"""Sentence-transformer embedding strategy. Ported verbatim from
rag-foundry's embedding/strategies/sentence_transformer/strategy.py."""

from rag_agent.runtime import get_sentence_transformer


class SentenceTransformerEmbeddingStrategy:
    """Embedding strategy using Sentence Transformers."""

    def __init__(self, config):
        self.config = config

        model_name = self.config.model_name or self.config.model
        if not model_name:
            raise ValueError(
                "SentenceTransformerEmbeddingStrategy requires "
                "'model_name' (or 'model') in the embedding config."
            )

        self.model = get_sentence_transformer(model_name)

    def embed(self, texts, is_query=False):
        if isinstance(texts, str):
            texts = [texts]

        if is_query and self.config.query_instruction:
            texts = [f"{self.config.query_instruction}{t}" for t in texts]

        return self.model.encode(texts, normalize_embeddings=True)
