from rag_agent.expansion.noop import NoopExpansionStrategy
from rag_agent.expansion.sibling import SiblingExpansionStrategy
from rag_agent.config import ExpansionType


def create_expansion(expansion_config, vector_store=None):
    if expansion_config is None or expansion_config.type == ExpansionType.NOOP:
        config = expansion_config.config if expansion_config else None
        return NoopExpansionStrategy(config)
    if expansion_config.type == ExpansionType.SIBLING:
        return SiblingExpansionStrategy(expansion_config.config, vector_store=vector_store)
    raise ValueError(f"Unsupported expansion type: {expansion_config.type}")


__all__ = ["create_expansion", "NoopExpansionStrategy", "SiblingExpansionStrategy"]
