"""Weighted sum fusion strategy. Ported verbatim from rag-foundry's
rag/modules/fusion/strategies/weighted_sum/strategy.py."""


class WeightedSumFusionStrategy:
    """Weighted sum fusion strategy."""

    def __init__(self, config):
        self.config = config

    def fuse(self, search_results: list[list[dict]]) -> list[dict]:
        weights = self.config.weights
        if not weights:
            weights = [1.0] * len(search_results)
        if len(weights) != len(search_results):
            raise ValueError("fusion.weights length must match the number of search strategies")

        combined_scores: dict[int, float] = {}
        representative: dict[int, dict] = {}

        for weight, ranked_list in zip(weights, search_results):
            if not ranked_list:
                continue

            max_score = max(result["score"] for result in ranked_list) + 1e-6
            for result in ranked_list:
                chunk_idx = result["index"]
                normalized_score = weight * (result["score"] / max_score)
                combined_scores[chunk_idx] = (
                    combined_scores.get(chunk_idx, 0.0) + normalized_score
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
            result["fusion_score"] = float(score)
            fused.append(result)

        return fused
