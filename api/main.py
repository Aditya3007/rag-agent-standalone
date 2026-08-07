"""Standalone retrieval FastAPI service.

Exposes each registered domain's retrieval pipeline (query transform ->
dense+sparse search -> fusion -> [expansion] -> [rerank]) over HTTP, so
retrieval (chunking/embedding/vector-store/BM25 — the heavy, stateful
part of the agent) can be deployed and scaled independently from
generation/routing/evaluation.

``RagAgent`` (and the Gradio UI, via RagAgent) becomes a client of this
service whenever the ``RAG_RETRIEVAL_API_URL`` environment variable is
set — see rag_agent/pipeline/remote_pipeline.py and README.md
"Retrieval API". Without that env var, RagAgent still builds/queries
the full pipeline in-process exactly as before; this service is purely
additive.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

or:
    python -m api.main
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from rag_agent.agent.domain_registry import DEFAULT_CONFIG_DIR, DomainSpec, load_domains
from rag_agent.dataset.huggingface_loader import DatasetLoadingConfig, HuggingFaceLoader
from rag_agent.dataset.parsers import DataProcessor, create_parser
from rag_agent.dataset.processors import ProcessingPipeline
from rag_agent.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_domains: Dict[str, DomainSpec] = {}


def _resolve_parser(data_parser):
    """Build a parser from either a plain string or {type, config} dict."""
    if isinstance(data_parser, dict):
        return create_parser(data_parser["type"], data_parser.get("config"))
    return create_parser(data_parser)


def _build_domain_pipeline(spec: DomainSpec) -> RAGPipeline:
    """Load this domain's HF corpus slice, parse it, and build its index.

    Mirrors RagAgent._build_domain_pipeline's local-build path — this
    service is where that indexing work actually lives when retrieval
    is delegated remotely.
    """
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_dir = Path(os.environ.get("RAG_CONFIG_DIR", DEFAULT_CONFIG_DIR))
    domains = load_domains(config_dir)

    logger.info("Building retrieval pipelines for %d domains: %s", len(domains), list(domains))
    for key, spec in domains.items():
        spec.pipeline = _build_domain_pipeline(spec)
        logger.info("Domain '%s' retrieval index ready (config=%s)", key, spec.rag_config.name)

    _domains.update(domains)
    yield
    _domains.clear()


app = FastAPI(title="RAG Retrieval API", version="1.0.0", lifespan=lifespan)


class RetrieveRequest(BaseModel):
    domain: str
    query: str


class RetrievedDoc(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None
    rerank_score: Optional[float] = None


class RetrieveResponse(BaseModel):
    domain: str
    query: str
    results: List[RetrievedDoc]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "domains_loaded": list(_domains.keys())}


@app.get("/domains")
def list_domains() -> Dict[str, Any]:
    return {
        "domains": [
            {"key": key, "description": spec.description, "config_name": spec.rag_config.name}
            for key, spec in _domains.items()
        ]
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    spec = _domains.get(req.domain)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown domain '{req.domain}'. Available: {list(_domains)}",
        )

    retrieved = spec.pipeline.retriever.retrieve(req.query)
    results = [
        RetrievedDoc(
            text=r["chunk"].text,
            metadata=r["chunk"].metadata,
            score=r.get("score"),
            rerank_score=r.get("rerank_score"),
        )
        for r in retrieved
    ]
    return RetrieveResponse(domain=req.domain, query=req.query, results=results)


if __name__ == "__main__":
    import uvicorn

    # Cloud Run (and similar PaaS) inject $PORT; RAG_API_PORT is kept as a
    # local-dev override, falling back to 8000 for bare-metal/local runs.
    port = int(os.environ.get("PORT", os.environ.get("RAG_API_PORT", 8000)))
    uvicorn.run(app, host="0.0.0.0", port=port)
