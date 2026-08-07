"""HTTP client that calls the standalone retrieval FastAPI service
(see api/main.py) instead of running query transform -> search -> fusion
-> expansion -> rerank in-process.

Exposes the same ``.retrieve(query) -> list[dict]`` interface as
``rag_agent.pipeline.retrieval_pipeline.RetrievalPipeline``, so callers
(``RagAgent.ask``, ``RAGPipeline.query``) don't need to know whether
retrieval happened locally or remotely: each result dict has a "chunk"
key exposing ``.text``/``.metadata``, plus any extra fields (``score``,
``rerank_score``) the API returned.
"""

from __future__ import annotations

from typing import Any, Dict, List

import requests


class _RemoteChunk:
    """Minimal stand-in for rag_agent.models.chunk.Chunk, carrying only
    the fields callers actually read off a retrieved result's "chunk"."""

    __slots__ = ("text", "metadata")

    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata


class RemoteRetrievalClient:
    """Calls ``POST {base_url}/retrieve`` for one fixed domain."""

    def __init__(self, base_url: str, domain: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.domain = domain
        self.timeout = timeout

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        response = requests.post(
            f"{self.base_url}/retrieve",
            json={"domain": self.domain, "query": query},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        results = []
        for item in payload["results"]:
            result = {k: v for k, v in item.items() if k not in ("text", "metadata")}
            result["chunk"] = _RemoteChunk(item.get("text", ""), item.get("metadata") or {})
            results.append(result)
        return results
