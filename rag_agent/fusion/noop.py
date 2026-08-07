"""No-op fusion strategy. Ported verbatim from rag-foundry's
rag/modules/fusion/strategies/noop/strategy.py."""


class NoOpFusionStrategy:
    """No-op fusion: passes first search list through unchanged."""

    def __init__(self, config):
        self.config = config

    def fuse(self, search_results: list[list[dict]]) -> list[dict]:
        if not search_results:
            return []
        return list(search_results[0])
