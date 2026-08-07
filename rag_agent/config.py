"""Trimmed RAGConfig dataclasses for the standalone agent.

Ported from rag-foundry's ``rag/config/config.py`` plus the per-module
``config.py`` files under ``rag/modules/*``, keeping only the strategy
configs actually used by the 4 pinned "winning" configs (see
``rag-agent-standalone/README.md`` for which strategy each domain uses).
Unused strategies (e.g. cohere/voyage/jina/mixedbread reranking,
hyde/step_back query transforms, fixed_window/token/semantic chunking)
are intentionally not ported.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Mode(str, Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"


class ProviderType(str, Enum):
    GROQ = "groq"
    OLLAMA = "ollama"


class ChunkingType(str, Enum):
    SENTENCE = "sentence"
    FINANCE_TABLE_AWARE = "finance_table_aware"


class EmbeddingType(str, Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"


class VectorStoreType(str, Enum):
    FAISS = "faiss"


class SearchType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"


class QueryTransformType(str, Enum):
    NOOP = "noop"
    MULTI_QUERY = "multi_query"


class FusionType(str, Enum):
    NOOP = "noop"
    RRF = "rrf"
    WEIGHTED_SUM = "weighted_sum"


class ExpansionType(str, Enum):
    NOOP = "noop"
    SIBLING = "sibling"


class RerankingType(str, Enum):
    CROSS_ENCODER = "cross_encoder"


class GenerationType(str, Enum):
    DEFAULT = "default"


class EvaluationType(str, Enum):
    TRACE = "trace"


def _coerce(value: Any, config_cls: type) -> Any:
    """Build ``config_cls`` from ``value``, silently dropping unknown keys.

    Mirrors rag-foundry's registry field-filtering behavior (see
    core/registry.py): a YAML config may contain keys that don't match
    the strategy's actual dataclass fields (e.g. the pinned pubmedqa
    config's fusion.config has ``dense_weight``/``sparse_weight``, but
    ``WeightedSumFusionConfig`` only defines ``top_k``/``weights``).
    Rather than erroring, those keys are dropped and the dataclass
    defaults apply — replicating the exact (if surprising) behavior of
    the winning config as benchmarked.
    """
    if value is None:
        return config_cls()
    if isinstance(value, dict):
        field_names = {f.name for f in fields(config_cls)}
        filtered = {k: v for k, v in value.items() if k in field_names}
        return config_cls(**filtered)
    return value


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    type: ProviderType
    api_key_env: Optional[str] = None
    params: dict = None

    def __post_init__(self):
        self.type = ProviderType(self.type)
        if self.params is None:
            self.params = {}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass
class SentenceChunkingConfig:
    max_words: int = 100
    overlap_sentences: int = 1


@dataclass
class FinanceTableAwareChunkingConfig:
    max_words: int = 300
    overlap_sentences: int = 1
    max_rows_per_table_chunk: int | None = None
    max_context_words: int | None = None


_CHUNKING_CONFIG_CLASSES = {
    ChunkingType.SENTENCE: SentenceChunkingConfig,
    ChunkingType.FINANCE_TABLE_AWARE: FinanceTableAwareChunkingConfig,
}


@dataclass
class ChunkingConfig:
    type: ChunkingType
    config: Any = None

    def __post_init__(self):
        self.type = ChunkingType(self.type)
        config_cls = _CHUNKING_CONFIG_CLASSES[self.type]
        self.config = _coerce(self.config, config_cls)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@dataclass
class SentenceTransformerEmbeddingConfig:
    model_name: Optional[str] = None
    model: Optional[str] = None
    dimension: int = 768
    query_instruction: Optional[str] = None


_EMBEDDING_CONFIG_CLASSES = {
    EmbeddingType.SENTENCE_TRANSFORMER: SentenceTransformerEmbeddingConfig,
}


@dataclass
class EmbeddingConfig:
    type: EmbeddingType
    config: Any = None

    def __post_init__(self):
        self.type = EmbeddingType(self.type)
        config_cls = _EMBEDDING_CONFIG_CLASSES[self.type]
        self.config = _coerce(self.config, config_cls)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

@dataclass
class FaissVectorStoreConfig:
    dimension: int = 768


_VECTOR_STORE_CONFIG_CLASSES = {
    VectorStoreType.FAISS: FaissVectorStoreConfig,
}


@dataclass
class VectorStoreConfig:
    type: VectorStoreType
    config: Any = None

    def __post_init__(self):
        self.type = VectorStoreType(self.type)
        config_cls = _VECTOR_STORE_CONFIG_CLASSES[self.type]
        self.config = _coerce(self.config, config_cls)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass
class DenseSearchConfig:
    top_k: int = 5
    context_window: int = 0
    min_similarity: float | None = None


@dataclass
class SparseSearchConfig:
    top_k: int = 5


_SEARCH_CONFIG_CLASSES = {
    SearchType.DENSE: DenseSearchConfig,
    SearchType.SPARSE: SparseSearchConfig,
}


@dataclass
class SearchStrategyConfig:
    type: SearchType
    config: Any = None

    def __post_init__(self):
        self.type = SearchType(self.type)
        config_cls = _SEARCH_CONFIG_CLASSES[self.type]
        self.config = _coerce(self.config, config_cls)


@dataclass
class SearchPipelineConfig:
    searches: list

    def __post_init__(self):
        if not self.searches:
            raise ValueError("search.searches must contain at least one search")
        self.searches = [
            item if isinstance(item, SearchStrategyConfig) else SearchStrategyConfig(**item)
            for item in self.searches
        ]


# ---------------------------------------------------------------------------
# Query transform
# ---------------------------------------------------------------------------

@dataclass
class NoOpQueryTransformConfig:
    pass


@dataclass
class MultiQueryQueryTransformConfig:
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    num_queries: int = 4
    max_tokens: int = 128


_QUERY_TRANSFORM_CONFIG_CLASSES = {
    QueryTransformType.NOOP: NoOpQueryTransformConfig,
    QueryTransformType.MULTI_QUERY: MultiQueryQueryTransformConfig,
}


@dataclass
class QueryTransformConfig:
    type: QueryTransformType
    provider: str | None = None
    config: Any = None

    def __post_init__(self):
        self.type = QueryTransformType(self.type)
        config_cls = _QUERY_TRANSFORM_CONFIG_CLASSES[self.type]
        self.config = _coerce(self.config, config_cls)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

@dataclass
class NoOpFusionConfig:
    pass


@dataclass
class RRFFusionConfig:
    top_k: int = 5
    k: int = 60


@dataclass
class WeightedSumFusionConfig:
    top_k: int = 5
    weights: list = field(default_factory=list)


_FUSION_CONFIG_CLASSES = {
    FusionType.NOOP: NoOpFusionConfig,
    FusionType.RRF: RRFFusionConfig,
    FusionType.WEIGHTED_SUM: WeightedSumFusionConfig,
}


@dataclass
class FusionConfig:
    type: FusionType
    config: Any = None

    def __post_init__(self):
        self.type = FusionType(self.type)
        config_cls = _FUSION_CONFIG_CLASSES[self.type]
        self.config = _coerce(self.config, config_cls)


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

@dataclass
class NoopExpansionConfig:
    pass


@dataclass
class SiblingExpansionConfig:
    max_siblings_per_result: int = 3
    sample_index_key: str = "sample_index"


_EXPANSION_CONFIG_CLASSES = {
    ExpansionType.NOOP: NoopExpansionConfig,
    ExpansionType.SIBLING: SiblingExpansionConfig,
}


@dataclass
class ExpansionConfig:
    type: ExpansionType = ExpansionType.NOOP
    config: Any = None

    def __post_init__(self):
        self.type = ExpansionType(self.type)
        config_cls = _EXPANSION_CONFIG_CLASSES[self.type]
        self.config = _coerce(self.config, config_cls)


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

@dataclass
class CrossEncoderRerankingConfig:
    model_name: str = None
    top_k: int = None


_RERANKING_CONFIG_CLASSES = {
    RerankingType.CROSS_ENCODER: CrossEncoderRerankingConfig,
}


@dataclass
class RerankingConfig:
    type: RerankingType
    config: Any = None

    def __post_init__(self):
        self.type = RerankingType(self.type)
        config_cls = _RERANKING_CONFIG_CLASSES[self.type]
        self.config = _coerce(self.config, config_cls)


# ---------------------------------------------------------------------------
# Retrieval (composes query_transform, search, fusion, expansion, rerank)
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    search: SearchPipelineConfig
    query_transform: Optional[QueryTransformConfig] = None
    fusion: Optional[FusionConfig] = None
    expansion: Optional[ExpansionConfig] = None
    rerank: Optional[RerankingConfig] = None

    def __post_init__(self):
        if isinstance(self.search, dict):
            self.search = SearchPipelineConfig(**self.search)
        if isinstance(self.query_transform, dict):
            self.query_transform = QueryTransformConfig(**self.query_transform)
        if isinstance(self.fusion, dict):
            self.fusion = FusionConfig(**self.fusion)
        if isinstance(self.expansion, dict):
            self.expansion = ExpansionConfig(**self.expansion)
        if isinstance(self.rerank, dict):
            self.rerank = RerankingConfig(**self.rerank)
        if len(self.search.searches) > 1 and self.fusion is None:
            raise ValueError(
                "retrieval.fusion is required when search.searches "
                "contains more than one strategy"
            )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass
class DefaultGenerationConfig:
    model: str = None
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: str = None
    user_prompt: str = None
    reasoning_effort: str | None = None


_GENERATION_CONFIG_CLASSES = {
    GenerationType.DEFAULT: DefaultGenerationConfig,
}


@dataclass
class GenerationConfig:
    strategy: GenerationType = GenerationType.DEFAULT
    provider: str = None
    config: Any = None

    def __post_init__(self):
        self.strategy = GenerationType(self.strategy)
        config_cls = _GENERATION_CONFIG_CLASSES[self.strategy]
        self.config = _coerce(self.config, config_cls)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class TRACeEvaluationConfig:
    model: str = None
    temperature: float = 0.0
    max_tokens: int = 2000


_EVALUATION_CONFIG_CLASSES = {
    EvaluationType.TRACE: TRACeEvaluationConfig,
}


@dataclass
class EvaluationConfig:
    type: EvaluationType
    provider: str = None
    config: Any = None

    def __post_init__(self):
        self.type = EvaluationType(self.type)
        config_cls = _EVALUATION_CONFIG_CLASSES[self.type]
        self.config = _coerce(self.config, config_cls)


# ---------------------------------------------------------------------------
# Cache / logging
# ---------------------------------------------------------------------------

@dataclass
class CacheConfig:
    enabled: bool = True
    cache_dir: str = "./cache"


@dataclass
class LoggingConfig:
    enabled: bool = True
    level: str = "INFO"
    show_progress: bool = True


# ---------------------------------------------------------------------------
# Top-level RAGConfig
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """Complete RAG pipeline configuration for a single pinned domain config."""

    providers: Dict[str, ProviderConfig]
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig

    mode: Mode = Mode.DEV
    name: str = "default"

    cache: CacheConfig = field(default_factory=CacheConfig)

    # Optional per-config data processing (runs inside build_index, before
    # chunking) — e.g. tatqa's sibling_entity_context step.
    data_processing: Optional[Dict[str, Any]] = None

    logging_config: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        self.mode = Mode(self.mode)
        self.cache = _coerce(self.cache, CacheConfig)
        self.logging_config = _coerce(self.logging_config, LoggingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        return cls(
            name=data.get("name", "default"),
            mode=data.get("mode", Mode.DEV),
            providers={
                name: ProviderConfig(**provider_data)
                for name, provider_data in data.get("providers", {}).items()
            },
            chunking=ChunkingConfig(**data["chunking"]),
            embedding=EmbeddingConfig(**data["embedding"]),
            vector_store=VectorStoreConfig(**data["vector_store"]),
            retrieval=RetrievalConfig(**data["retrieval"]),
            generation=GenerationConfig(**data.get("generation", {})),
            cache=data.get("cache"),
            data_processing=data.get("data_processing"),
            logging_config=data.get("logging_config"),
        )
