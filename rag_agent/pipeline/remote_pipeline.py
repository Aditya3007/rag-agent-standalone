"""Thin pipeline used when retrieval is delegated to the standalone
retrieval FastAPI service (api/main.py) instead of running the full
RAGPipeline (chunking, embedding, vector store, BM25) in this process.

Only providers + a generator are built locally — no corpus is
downloaded/indexed here, since that already happened in the retrieval
API process. ``.retriever`` is an HTTP client hitting the API, matching
RAGPipeline's ``.retriever``/``.generator`` interface so RagAgent.ask()
doesn't need to know which mode is active.

Enabled via the ``RAG_RETRIEVAL_API_URL`` environment variable (see
RagAgent.__init__ and README.md).
"""

from rag_agent.generation import create_generator
from rag_agent.pipeline.remote_retrieval_client import RemoteRetrievalClient
from rag_agent.providers.provider_manager import ProviderManager


class RemoteBackedPipeline:
    """Drop-in replacement for RAGPipeline when retrieval runs remotely."""

    def __init__(self, config, domain_key: str, retrieval_api_url: str):
        self.config = config

        for provider_name, provider_config in config.providers.items():
            ProviderManager.register(
                provider_name=provider_name,
                provider_type=provider_config.type,
                config=provider_config,
            )

        self.retriever = RemoteRetrievalClient(retrieval_api_url, domain=domain_key)

        generation_provider = ProviderManager.get_provider(config.generation.provider)
        self.generator = create_generator(config.generation, provider=generation_provider)
