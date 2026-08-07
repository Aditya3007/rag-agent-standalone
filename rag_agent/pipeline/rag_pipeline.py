"""Complete RAG pipeline for a single pinned domain config. Trimmed/ported
from rag-foundry's rag/pipeline/rag_pipeline.py: registries are replaced
by simple factory dispatch over only the strategies the 4 pinned configs
use (see rag_agent/{chunking,embedding,vectorstore,search,fusion,
expansion,query_transform,reranking,generation}/__init__.py).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from rag_agent.cache.cache_manager import CacheManager
from rag_agent.chunking import create_chunker
from rag_agent.embedding import create_embedder
from rag_agent.vectorstore import create_vector_store
from rag_agent.search import create_search_strategy
from rag_agent.search.bm25_store import BM25Store
from rag_agent.fusion import create_fusion
from rag_agent.expansion import create_expansion
from rag_agent.query_transform import create_query_transform
from rag_agent.reranking import create_reranker
from rag_agent.generation import create_generator
from rag_agent.providers.provider_manager import ProviderManager
from rag_agent.pipeline.retrieval_pipeline import RetrievalPipeline
from rag_agent.pipeline.search_pipeline import SearchPipeline
from rag_agent.models.document import Document
from rag_agent.models.query_result import QueryResult

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Chunk -> embed -> index -> retrieve -> generate, for one pinned RAGConfig."""

    def __init__(self, config, jsonl_path: Optional[Path] = None):
        self.config = config
        self.jsonl_path = jsonl_path

        self.cache_manager = CacheManager(self.config.cache)
        self.bm25_store = None

        self._initialize_providers()
        self._initialize_strategies()

    def _initialize_providers(self):
        for provider_name, provider_config in self.config.providers.items():
            ProviderManager.register(
                provider_name=provider_name,
                provider_type=provider_config.type,
                config=provider_config,
            )

    def _initialize_strategies(self):
        self.chunker = create_chunker(self.config.chunking)
        self.embedder = create_embedder(self.config.embedding)
        self.vector_store = create_vector_store(self.config.vector_store)

        # Query transform
        query_transform_config = self.config.retrieval.query_transform
        provider = None
        if query_transform_config and query_transform_config.provider:
            provider = ProviderManager.get_provider(query_transform_config.provider)
        query_transform = create_query_transform(query_transform_config, provider=provider)

        # Search pipeline (BM25 store is built lazily during build_index)
        search_strategies = [
            create_search_strategy(
                search_config,
                embedder=self.embedder,
                vector_store=self.vector_store,
                bm25_store=lambda: self.bm25_store,
            )
            for search_config in self.config.retrieval.search.searches
        ]
        search_pipeline = SearchPipeline(search_strategies)

        # Fusion
        fusion = create_fusion(self.config.retrieval.fusion)

        # Post-fusion expansion (optional)
        expansion = create_expansion(self.config.retrieval.expansion, vector_store=self.vector_store)

        # Reranker (optional)
        reranker = create_reranker(self.config.retrieval.rerank)

        self.retriever = RetrievalPipeline(
            query_transform=query_transform,
            search_pipeline=search_pipeline,
            fusion=fusion,
            expansion=expansion,
            reranker=reranker,
        )

        # Generation
        generation_provider = ProviderManager.get_provider(self.config.generation.provider)
        self.generator = create_generator(self.config.generation, provider=generation_provider)

    def build_index(self, documents: list[Document]):
        """Build the vector index, reusing cached stages where possible."""
        logger.info("Processing %d documents...", len(documents))
        cache = self.cache_manager

        if self.config.data_processing:
            from rag_agent.dataset.processors import ProcessingPipeline
            processing_pipeline = ProcessingPipeline.from_config(self.config.data_processing)
            documents = processing_pipeline.run(documents)

        datasource_hash = cache.datasource_hash(documents)

        # 1. Chunking
        chunk_key = cache.get_chunk_cache_key(datasource_hash, self.config.chunking)
        chunks = cache.load_chunk_cache(chunk_key)
        if chunks is not None:
            logger.info("[Chunk Cache] HIT key=%s", chunk_key)
        else:
            logger.info("[Chunk Cache] MISS key=%s", chunk_key)
            chunks = []
            for doc in documents:
                chunks.extend(self.chunker.chunk(doc))
            cache.save_chunk_cache(chunk_key, chunks, datasource_hash, self.config.chunking)
        logger.debug("Chunk cache key=%s (%d chunks)", chunk_key, len(chunks))

        # 2. Embedding
        embedding_key = cache.get_embedding_cache_key(chunk_key, self.config.embedding)
        embeddings = cache.load_embedding_cache(embedding_key)
        if embeddings is not None:
            logger.info("[Embedding Cache] HIT key=%s", embedding_key)
        else:
            logger.info("[Embedding Cache] MISS key=%s", embedding_key)
            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedder.embed(texts)
            embeddings = np.array(embeddings).astype("float32")
            cache.save_embedding_cache(embedding_key, embeddings, chunk_key, self.config.embedding)
        embeddings = np.asarray(embeddings).astype("float32")
        logger.debug("Embedding cache key=%s (%d vectors)", embedding_key, len(embeddings))

        # 3. Vector index (retrieval config is intentionally NOT part of the key)
        index_key = cache.get_index_cache_key(embedding_key, self.config.vector_store)
        cached_index = cache.load_index_cache(index_key)
        if cached_index is not None:
            logger.info("[Index Cache] HIT key=%s", index_key)
            self.vector_store.index = cached_index
            self.vector_store.chunks = list(chunks)
        else:
            logger.info("[Index Cache] MISS key=%s", index_key)
            self.vector_store.add(embeddings, chunks)
            cache.save_index_cache(index_key, self.vector_store.index, embedding_key, self.config.vector_store)
        logger.debug("Index cache key=%s", index_key)

        logger.info("Vector store ready with %d chunks", len(self.vector_store.chunks))

        # 4. BM25 index (for sparse search)
        has_sparse = any(sc.type == "sparse" for sc in self.config.retrieval.search.searches)
        if has_sparse:
            self.bm25_store = BM25Store(chunks)
            logger.info("BM25 store ready with %d chunks", len(chunks))

    def query(self, query: str) -> QueryResult:
        """Run complete RAG query and return QueryResult."""
        retrieved = self.retriever.retrieve(query)

        retrieved_docs = []
        for r in retrieved:
            doc = {k: v for k, v in r.items() if k != "chunk"}
            doc["text"] = r["chunk"].text
            doc["metadata"] = r["chunk"].metadata
            retrieved_docs.append(doc)

        context = "\n\n".join(
            f"[Document {i + 1}]\n{doc['text']}" for i, doc in enumerate(retrieved_docs)
        )
        answer = self.generator.generate(query=query, context=context)

        retrieved_doc_objects = [
            Document(
                title=doc.get("title", ""),
                content=doc.get("text", ""),
                metadata=doc.get("metadata", {}),
            )
            for doc in retrieved_docs
        ]

        return QueryResult(
            query=query,
            retrieved_docs=retrieved_doc_objects,
            answer=answer,
            metadata={},
        )
