from rag_agent.generation.default import DefaultGenerationStrategy
from rag_agent.config import GenerationType

_STRATEGIES = {
    GenerationType.DEFAULT: DefaultGenerationStrategy,
}


def create_generator(generation_config, provider):
    strategy_cls = _STRATEGIES[generation_config.strategy]
    return strategy_cls(generation_config.config, provider=provider)


__all__ = ["create_generator", "DefaultGenerationStrategy"]
