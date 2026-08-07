"""Cross-encoder reranking strategy. Ported verbatim from rag-foundry's
rag/modules/reranking/strategies/cross_encoder/strategy.py."""

from rag_agent.runtime import get_cross_encoder


class CrossEncoderRerankingStrategy:
    """Cross-encoder reranking strategy using sentence-transformers."""

    def __init__(self, config):
        self.config = config

        if not self.config.model_name:
            raise ValueError("CrossEncoderRerankingStrategy requires 'model_name' in the reranker config.")

        self.model = get_cross_encoder(self.config.model_name)

    def rerank(self, query, texts):
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs)
        return [float(score) for score in scores]
