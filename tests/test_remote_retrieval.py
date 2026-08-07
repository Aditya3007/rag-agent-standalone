"""Unit tests for the remote-retrieval mode (RAG_RETRIEVAL_API_URL):
RemoteRetrievalClient's HTTP call is mocked, so these run with no
network access and no real retrieval API running.
"""

from unittest.mock import MagicMock, patch

from rag_agent.pipeline.remote_retrieval_client import RemoteRetrievalClient


def test_remote_retrieval_client_maps_response_to_chunk_shape():
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "domain": "pubmedqa",
        "query": "what is aspirin?",
        "results": [
            {"text": "doc one", "metadata": {"doc_id": "d0"}, "score": 0.9, "rerank_score": 0.8},
            {"text": "doc two", "metadata": {}, "score": 0.5, "rerank_score": None},
        ],
    }
    fake_response.raise_for_status.return_value = None

    with patch("rag_agent.pipeline.remote_retrieval_client.requests.post", return_value=fake_response) as mock_post:
        client = RemoteRetrievalClient(base_url="http://localhost:8000/", domain="pubmedqa")
        results = client.retrieve("what is aspirin?")

    mock_post.assert_called_once_with(
        "http://localhost:8000/retrieve",
        json={"domain": "pubmedqa", "query": "what is aspirin?"},
        timeout=60.0,
    )

    assert len(results) == 2
    assert results[0]["chunk"].text == "doc one"
    assert results[0]["chunk"].metadata == {"doc_id": "d0"}
    assert results[0]["score"] == 0.9
    assert results[0]["rerank_score"] == 0.8
    assert results[1]["chunk"].text == "doc two"
    assert results[1]["chunk"].metadata == {}


def test_remote_backed_pipeline_builds_generator_and_remote_retriever():
    from rag_agent.pipeline.remote_pipeline import RemoteBackedPipeline
    from rag_agent.agent.domain_registry import load_domains
    from pathlib import Path

    config_dir = Path(__file__).resolve().parent.parent / "configs"
    domains = load_domains(config_dir)
    spec = domains["covidqa"]

    with patch("rag_agent.pipeline.remote_pipeline.ProviderManager.register"), \
         patch("rag_agent.pipeline.remote_pipeline.ProviderManager.get_provider", return_value=MagicMock()):
        pipeline = RemoteBackedPipeline(
            spec.rag_config, domain_key="covidqa", retrieval_api_url="http://localhost:8000"
        )

    assert pipeline.retriever.domain == "covidqa"
    assert pipeline.retriever.base_url == "http://localhost:8000"
    assert pipeline.generator is not None
