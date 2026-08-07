"""RagAgent: query -> domain routing -> pinned pipeline -> answer + scores.

Eagerly builds (or reuses, via the content-addressed CacheManager) a
RAGPipeline per domain at construction time, using each domain's pinned
"winning" config. ``ask()`` routes the query with an LLM, retrieves +
generates using that domain's pipeline, and evaluates the answer with a
single shared TRACe evaluator instance (same evaluator for every domain).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_agent.agent.domain_registry import (
    DEFAULT_CONFIG_DIR,
    DomainSpec,
    load_domains,
    load_evaluation_config,
    load_rgb_evaluation_config,
)
from rag_agent.agent.domain_router import LLMDomainRouter
from rag_agent.dataset.huggingface_loader import DatasetLoadingConfig, HuggingFaceLoader
from rag_agent.dataset.parsers import DataProcessor, create_parser
from rag_agent.dataset.processors import ProcessingPipeline
from rag_agent.evaluation.trace import TRACeEvaluationStrategy
from rag_agent.evaluation.rgb import RGBEvaluationConfig, RGBEvaluationStrategy
from rag_agent.pipeline.rag_pipeline import RAGPipeline
from rag_agent.pipeline.remote_pipeline import RemoteBackedPipeline
from rag_agent.providers.provider_manager import ProviderManager

logger = logging.getLogger(__name__)

# When set, retrieval is delegated to the standalone retrieval FastAPI
# service (see api/main.py) instead of building the full chunk/embed/
# index pipeline in this process. See README.md "Retrieval API".
RAG_RETRIEVAL_API_URL_ENV = "RAG_RETRIEVAL_API_URL"


@dataclass
class AgentResult:
    """Everything about one ``ask()`` call: routing decision, answer,
    retrieved docs, and every score TRACe returned."""

    domain: str
    config_name: str
    query: str
    answer: str
    retrieved_docs: List[Dict[str, Any]]
    scores: Dict[str, Any]
    rgb_scores: Dict[str, Any] = field(default_factory=dict)
    latencies: Dict[str, float] = field(default_factory=dict)


def _resolve_parser(data_parser):
    """Build a parser from either a plain string or {type, config} dict."""
    if isinstance(data_parser, dict):
        return create_parser(data_parser["type"], data_parser.get("config"))
    return create_parser(data_parser)


class RagAgent:
    """Domain-routed RAG agent covering pubmedqa/covidqa/finqa/tatqa."""

    def __init__(
        self,
        config_dir: Path | str = DEFAULT_CONFIG_DIR,
        router_provider_name: str = "groq",
        router_model: str = "llama-3.3-70b-versatile",
        retrieval_api_url: Optional[str] = None,
    ):
        config_dir = Path(config_dir)
        self.domains: Dict[str, DomainSpec] = load_domains(config_dir)

        # If set (directly or via RAG_RETRIEVAL_API_URL), retrieval is
        # delegated to the standalone retrieval API instead of building
        # the full chunk/embed/index pipeline in this process — see
        # README.md "Retrieval API".
        self.retrieval_api_url = retrieval_api_url or os.environ.get(RAG_RETRIEVAL_API_URL_ENV)
        if self.retrieval_api_url:
            logger.info("Using remote retrieval API at %s", self.retrieval_api_url)

        logger.info("Building pipelines for %d domains: %s", len(self.domains), list(self.domains))
        for key, spec in self.domains.items():
            spec.pipeline = self._build_domain_pipeline(spec, retrieval_api_url=self.retrieval_api_url)
            logger.info("Domain '%s' ready (config=%s)", key, spec.rag_config.name)

        # Router and evaluator both reuse a provider already registered by
        # building the domain pipelines above (all 4 pinned configs
        # declare a 'groq' provider), so no separate credentials are needed.
        router_provider = ProviderManager.get_provider(router_provider_name)
        self.router = LLMDomainRouter(
            provider=router_provider,
            model=router_model,
            domains=self.domains,
        )

        eval_config = load_evaluation_config(config_dir)
        eval_provider = ProviderManager.get_provider(eval_config.provider)
        self.evaluator = TRACeEvaluationStrategy(eval_config.config, provider=eval_provider)

        # Supplementary RGB-benchmark-style scoring (rejection/factual-error
        # detection on the generated answer itself, no dataset ground
        # truth involved — see evaluation.yaml's ``rgb:`` section and
        # rag_agent/evaluation/rgb.py), reported alongside TRACe.
        rgb_config = load_rgb_evaluation_config(config_dir)
        rgb_provider = ProviderManager.get_provider(rgb_config.provider)
        self.rgb_evaluator = RGBEvaluationStrategy(
            RGBEvaluationConfig(**rgb_config.config), provider=rgb_provider
        )

    @staticmethod
    def _build_domain_pipeline(spec: DomainSpec, retrieval_api_url: Optional[str] = None):
        """Build this domain's retrieval+generation pipeline.

        If ``retrieval_api_url`` is set, skips loading/chunking/embedding
        the corpus locally entirely and returns a ``RemoteBackedPipeline``
        that calls the retrieval API instead (see api/main.py, which must
        already have this domain's index built). Otherwise loads this
        domain's HF corpus slice, parses it, and builds its index locally,
        exactly as before.
        """
        if retrieval_api_url:
            return RemoteBackedPipeline(
                spec.rag_config, domain_key=spec.key, retrieval_api_url=retrieval_api_url
            )

        loader_cfg = spec.data_loader
        loader = HuggingFaceLoader(
            dataset_name=loader_cfg["dataset_name"],
            subset=loader_cfg.get("subset"),
            split=loader_cfg.get("split", "test"),
            config=DatasetLoadingConfig(limit=loader_cfg.get("limit")),
        )
        raw_data = loader.load()

        parser = _resolve_parser(spec.data_parser)
        documents = DataProcessor(parser_strategy=parser).process_dataset(raw_data)

        if spec.data_processing:
            documents = ProcessingPipeline.from_config(spec.data_processing).run(documents)

        pipeline = RAGPipeline(spec.rag_config)
        pipeline.build_index(documents)
        return pipeline

    def ask(self, query: str) -> AgentResult:
        """Route ``query`` to a domain, answer it, and evaluate the answer."""
        total_start = time.time()

        domain_key = self.router.classify(query)
        spec = self.domains[domain_key]
        pipeline = spec.pipeline

        retrieval_start = time.time()
        retrieved = pipeline.retriever.retrieve(query)
        retrieval_ms = (time.time() - retrieval_start) * 1000

        context = "\n\n".join(
            f"[Document {i + 1}]\n{r['chunk'].text}" for i, r in enumerate(retrieved)
        )

        generation_start = time.time()
        answer = pipeline.generator.generate(query=query, context=context)
        generation_ms = (time.time() - generation_start) * 1000

        eval_start = time.time()
        scores = self.evaluator.evaluate(query=query, retrieved_docs=retrieved, response=answer)
        eval_ms = (time.time() - eval_start) * 1000

        # RGB scoring runs purely on the query + generated answer (no
        # retrieved_docs dependency — see rag_agent/evaluation/rgb.py).
        rgb_eval_start = time.time()
        rgb_scores = self.rgb_evaluator.evaluate(query=query, response=answer)
        rgb_eval_ms = (time.time() - rgb_eval_start) * 1000

        total_ms = (time.time() - total_start) * 1000

        retrieved_docs = [
            {
                **{k: v for k, v in r.items() if k != "chunk"},
                "text": r["chunk"].text,
                "metadata": r["chunk"].metadata,
            }
            for r in retrieved
        ]

        return AgentResult(
            domain=domain_key,
            config_name=spec.rag_config.name,
            query=query,
            answer=answer,
            retrieved_docs=retrieved_docs,
            scores=scores,
            rgb_scores=rgb_scores,
            latencies={
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "evaluation_ms": eval_ms,
                "rgb_evaluation_ms": rgb_eval_ms,
                "total_ms": total_ms,
            },
        )

    @staticmethod
    def print_result(result: AgentResult) -> None:
        """Pretty-print an AgentResult: domain, config, answer, retrieved
        docs, and every score field."""
        print(f"\n{'=' * 70}")
        print(f"Domain routed to: {result.domain}  (config: {result.config_name})")
        print("=" * 70)

        print(f"\nQuery: {result.query}")

        print(f"\n--- Retrieved Documents ({len(result.retrieved_docs)}) ---")
        for i, doc in enumerate(result.retrieved_docs[:5], 1):
            print(f"\n{i}. {doc['text'][:200]}...")

        print("\n--- Answer ---")
        print(result.answer)

        print("\n--- TRACe Scores ---")
        for key, value in result.scores.items():
            if key == "judge_output":
                continue
            print(f"  {key}: {value}")

        print("\n--- RGB Scores ---")
        for key, value in result.rgb_scores.items():
            print(f"  {key}: {value}")

        print("\n--- Latencies ---")
        for key, value in result.latencies.items():
            print(f"  {key}: {value:.2f}ms")
