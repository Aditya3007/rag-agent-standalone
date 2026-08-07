"""Search pipeline: runs multiple search strategies and collects results.
Ported verbatim from rag-foundry's rag/pipeline/search_pipeline.py."""

from rag_agent.pipeline.context import RetrievalContext


class SearchPipeline:
    """Runs one or more search strategies and collects their ranked lists."""

    def __init__(self, strategies: list):
        if not strategies:
            raise ValueError("SearchPipeline requires at least one search strategy")
        self.strategies = strategies

    def run(self, ctx: RetrievalContext) -> RetrievalContext:
        ctx.search_results = []

        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__.lower()

            if "dense" in strategy_name:
                queries = ctx.dense_queries if ctx.dense_queries else [ctx.query]
            elif "sparse" in strategy_name:
                queries = ctx.sparse_queries if ctx.sparse_queries else [ctx.query]
            else:
                queries = [ctx.query]

            result = strategy.search(queries)
            ctx.search_results.append(result)

        return ctx
