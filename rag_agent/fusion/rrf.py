"""Reciprocal rank fusion strategy. Ported verbatim from rag-foundry's
rag/modules/fusion/strategies/rrf/strategy.py."""


class RRFFusionStrategy:
    """Reciprocal rank fusion strategy."""

    def __init__(self, config):
        self.config = config

    def fuse(self, search_results: list[list[dict]]) -> list[dict]:
        combined_scores: dict[int, float] = {}
        representative: dict[int, dict] = {}

        for ranked_list in search_results:
            for rank, result in enumerate(ranked_list):
                chunk_idx = result["index"]
                combined_scores[chunk_idx] = (
                    combined_scores.get(chunk_idx, 0.0) + 1.0 / (self.config.k + rank + 1)
                )
                if chunk_idx not in representative:
                    representative[chunk_idx] = dict(result)

        sorted_indices = sorted(
            combined_scores.items(), key=lambda item: item[1], reverse=True
        )[:self.config.top_k]

        fused = []
        for chunk_idx, score in sorted_indices:
            result = dict(representative[chunk_idx])
            result["score"] = float(score)
            result["rrf_score"] = float(score)
            fused.append(result)

        return fused
