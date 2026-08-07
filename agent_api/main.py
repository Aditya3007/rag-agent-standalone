"""Standalone agent FastAPI service: exposes RagAgent.ask() over HTTP.

This is the domain-routed "brain" (routing -> retrieval -> generation ->
evaluation). By default retrieval runs in-process, exactly like
scripts/ask.py and ui/app.py. Set RAG_RETRIEVAL_API_URL to delegate
retrieval to the standalone retrieval API (api/main.py) running as its
own service instead — see README.md "Retrieval API" and "Agent API".

Run with:
    uvicorn agent_api.main:app --host 0.0.0.0 --port 8080

Listens on $PORT if set (Cloud Run injects this), defaulting to 8080,
when run directly via ``python -m agent_api.main``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from rag_agent.agent.rag_agent import RagAgent

logger = logging.getLogger(__name__)

_agent: Optional[RagAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    logger.info("Building RagAgent (all domain pipelines)...")
    _agent = RagAgent()
    logger.info("RagAgent ready (retrieval_api_url=%s).", _agent.retrieval_api_url)
    yield
    _agent = None


app = FastAPI(title="RAG Agent API", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    domain: str
    config_name: str
    query: str
    answer: str
    retrieved_docs: List[Dict[str, Any]]
    scores: Dict[str, Any]
    rgb_scores: Dict[str, Any]
    latencies: Dict[str, float]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if _agent is not None else "starting",
        "domains": list(_agent.domains.keys()) if _agent is not None else [],
        "retrieval_api_url": _agent.retrieval_api_url if _agent is not None else None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = _agent.ask(req.query)
    return AskResponse(
        domain=result.domain,
        config_name=result.config_name,
        query=result.query,
        answer=result.answer,
        retrieved_docs=result.retrieved_docs,
        scores=result.scores,
        rgb_scores=result.rgb_scores,
        latencies=result.latencies,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
