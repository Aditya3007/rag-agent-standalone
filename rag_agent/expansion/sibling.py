"""Sibling expansion strategy. Ported verbatim from rag-foundry's
rag/modules/expansion/strategies/sibling/strategy.py.

Pulls in sibling chunks sharing the same source sample (e.g. a financial
table that scores low on embedding similarity but sits next to a
narrative chunk that did retrieve well), so the reranker gets a chance to
judge them directly against the query instead of losing them to
embedding similarity alone.
"""


class SiblingExpansionStrategy:

    def __init__(self, config, vector_store=None):
        self.config = config
        self.vector_store = vector_store

    def expand(self, fused_results: list[dict]) -> list[dict]:
        if not fused_results or self.vector_store is None:
            return fused_results

        all_chunks = self.vector_store.chunks
        sample_key = self.config.sample_index_key

        present_indices = {r["index"] for r in fused_results}
        min_score = min((r["score"] for r in fused_results), default=0.0)

        expanded = list(fused_results)
        for result in fused_results:
            sample_value = result["chunk"].metadata.get(sample_key)
            if sample_value is None:
                continue

            added = 0
            for idx, chunk in enumerate(all_chunks):
                if added >= self.config.max_siblings_per_result:
                    break
                if idx in present_indices:
                    continue
                if chunk.metadata.get(sample_key) != sample_value:
                    continue
                if not chunk.text.strip():
                    continue

                present_indices.add(idx)
                expanded.append({
                    "index": idx,
                    "chunk": chunk,
                    "score": min_score,
                    "sibling_expanded": True,
                })
                added += 1

        return expanded
