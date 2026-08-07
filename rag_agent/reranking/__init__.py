from rag_agent.reranking.cross_encoder import CrossEncoderRerankingStrategy
from rag_agent.config import RerankingType

_STRATEGIES = {
    RerankingType.CROSS_ENCODER: CrossEncoderRerankingStrategy,
}


def create_reranker(reranking_config):
    if reranking_config is None:
        return None
    strategy_cls = _STRATEGIES[reranking_config.type]
    return strategy_cls(reranking_config.config)


__all__ = ["create_reranker", "CrossEncoderRerankingStrategy"]
