from rag_agent.fusion.noop import NoOpFusionStrategy
from rag_agent.fusion.rrf import RRFFusionStrategy
from rag_agent.fusion.weighted_sum import WeightedSumFusionStrategy
from rag_agent.config import FusionType

_STRATEGIES = {
    FusionType.NOOP: NoOpFusionStrategy,
    FusionType.RRF: RRFFusionStrategy,
    FusionType.WEIGHTED_SUM: WeightedSumFusionStrategy,
}


def create_fusion(fusion_config):
    strategy_cls = _STRATEGIES[fusion_config.type]
    return strategy_cls(fusion_config.config)


__all__ = ["create_fusion", "NoOpFusionStrategy", "RRFFusionStrategy", "WeightedSumFusionStrategy"]
