from rag_agent.chunking.sentence import SentenceChunkingStrategy
from rag_agent.chunking.finance_table_aware import FinanceTableAwareChunkingStrategy
from rag_agent.config import ChunkingType

_STRATEGIES = {
    ChunkingType.SENTENCE: SentenceChunkingStrategy,
    ChunkingType.FINANCE_TABLE_AWARE: FinanceTableAwareChunkingStrategy,
}


def create_chunker(chunking_config):
    strategy_cls = _STRATEGIES[chunking_config.type]
    return strategy_cls(chunking_config.config)


__all__ = ["create_chunker", "SentenceChunkingStrategy", "FinanceTableAwareChunkingStrategy"]
