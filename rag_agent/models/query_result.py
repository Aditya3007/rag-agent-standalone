"""Query result model. Ported from rag-foundry's rag/models/query_result.py."""

from dataclasses import dataclass
from typing import Dict, Any

from rag_agent.models.document import Document


@dataclass
class QueryResult:
    query: str
    retrieved_docs: list[Document]
    answer: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert QueryResult to JSON-serializable dictionary."""
        retrieved_docs_dict = [
            {
                "title": doc.title,
                "content": doc.content,
                "metadata": doc.metadata,
            }
            for doc in self.retrieved_docs
        ]

        return {
            "query": self.query,
            "retrieved_docs": retrieved_docs_dict,
            "answer": self.answer,
            "metadata": self.metadata,
        }
