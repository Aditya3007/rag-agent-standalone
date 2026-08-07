"""Retrieval pipeline: orchestrates query transform, search, fusion,
expansion, and rerank stages. Ported verbatim from rag-foundry's
rag/pipeline/retrieval_pipeline.py."""

from rag_agent.pipeline.context import RetrievalContext


class RetrievalPipeline:
    """Orchestrates query transform, search, fusion, expansion, and rerank stages."""

    def __init__(self, query_transform, search_pipeline, fusion, expansion=None, reranker=None):
        self.query_transform = query_transform
        self.search_pipeline = search_pipeline
        self.fusion = fusion
        self.expansion = expansion
        self.reranker = reranker

    def retrieve(self, query: str) -> list[dict]:
        ctx = RetrievalContext(query=query)

        transform_result = self.query_transform.transform(query)
        ctx.dense_queries = transform_result.dense_queries
        ctx.sparse_queries = transform_result.sparse_queries

        ctx = self.search_pipeline.run(ctx)

        ctx.fused_results = self.fusion.fuse(ctx.search_results)

        if self.expansion:
            ctx.fused_results = self.expansion.expand(ctx.fused_results)

        if self.reranker:
            candidate_texts = [result["chunk"].text for result in ctx.fused_results]
            rerank_scores = self.reranker.rerank(query, candidate_texts)

            reranked = []
            for candidate, score in zip(ctx.fused_results, rerank_scores):
                result = dict(candidate)
                result["rerank_score"] = float(score)
                reranked.append(result)

            top_k = self.reranker.config.top_k or getattr(self.reranker.config, "top_n", 5)
            reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
            ctx.results = reranked[:top_k]
        else:
            ctx.results = ctx.fused_results

        return ctx.results
