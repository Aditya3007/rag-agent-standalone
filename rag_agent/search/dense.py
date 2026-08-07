"""Dense search strategy. Ported verbatim from rag-foundry's
rag/modules/search/strategies/dense/strategy.py."""

from rag_agent.models.chunk import Chunk


class DenseSearchStrategy:
    """Dense vector search strategy."""

    def __init__(self, config, embedder, vector_store):
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store

    def search(self, queries: list[str]):
        all_results = {}

        for query in queries:
            query_embedding = self.embedder.embed(query, is_query=True)

            if hasattr(query_embedding[0], "__len__"):
                query_embedding = query_embedding[0]

            distances, indices = self.vector_store.search(
                query_embedding.reshape(1, -1),
                self.config.top_k,
            )

            for idx, distance in zip(indices[0], distances[0]):
                if idx >= 0:
                    if (
                        self.config.min_similarity is not None
                        and distance < self.config.min_similarity
                    ):
                        continue
                    chunk_idx = int(idx)
                    if chunk_idx not in all_results or distance > all_results[chunk_idx]["score"]:
                        all_results[chunk_idx] = {
                            "index": chunk_idx,
                            "chunk": self.vector_store.chunks[chunk_idx],
                            "score": float(distance),
                            "dense_score": float(distance),
                        }

        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)[:self.config.top_k]

        if self.config.context_window > 0:
            sorted_results = self._expand_with_neighbors(sorted_results)

        return sorted_results

    def _expand_with_neighbors(self, results):
        all_chunks = self.vector_store.chunks
        total = len(all_chunks)
        seen_indices = set()
        expanded = []

        for r in results:
            idx = r["index"]
            doc_id = r["chunk"].metadata.get("doc_id")

            neighbor_indices = []
            for offset in range(-self.config.context_window, self.config.context_window + 1):
                ni = idx + offset
                if 0 <= ni < total and ni not in seen_indices:
                    neighbor = all_chunks[ni]
                    if neighbor.metadata.get("doc_id") == doc_id:
                        neighbor_indices.append(ni)
                        seen_indices.add(ni)

            neighbor_indices.sort()

            merged_text = "\n".join(all_chunks[ni].text for ni in neighbor_indices)
            merged_chunk = Chunk(text=merged_text, metadata=r["chunk"].metadata)

            expanded.append({
                **r,
                "chunk": merged_chunk,
                "neighbor_indices": neighbor_indices,
            })

        return expanded
