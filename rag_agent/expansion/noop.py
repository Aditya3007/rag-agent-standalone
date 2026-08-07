"""No-op post-fusion expansion strategy. Ported verbatim from rag-foundry's
rag/modules/expansion/strategies/noop/strategy.py."""


class NoopExpansionStrategy:
    """Expansion strategy that passes fused results through unchanged."""

    def __init__(self, config):
        self.config = config

    def expand(self, fused_results: list[dict]) -> list[dict]:
        return fused_results
