"""Document parsers. Consolidated/ported from rag-foundry's parsers/*.py
(noop_parser, title_passage_combined_parser, table_heading_parser) —
the 3 parsers used by the 4 pinned domain configs.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from rag_agent.models.document import Document


class NoopParser:
    """Passes through raw text as-is (used by pubmedqa, finqa).

    Produces a Document with empty title and the raw text as content.
    """

    def parse(self, raw_document: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        if not raw_document:
            raise ValueError("raw_document cannot be None or empty")

        final_metadata = {**(metadata or {}), "parser_type": "noop"}
        return Document(title="", content=raw_document.strip(), metadata=final_metadata)


@dataclass
class TitlePassageCombinedConfig:
    """include_labels=False (default): content is '<title>\\n\\n<passage>'.
    include_labels=True: content is 'Title: <title>\\nPassage: <passage>'."""
    include_labels: bool = False


class TitlePassageCombinedParser:
    """Merges 'Title: ... / Passage: ...' raw text into document content
    (used by covidqa)."""

    def __init__(self, config: TitlePassageCombinedConfig = None):
        if config is None:
            config = TitlePassageCombinedConfig()
        elif isinstance(config, dict):
            config = TitlePassageCombinedConfig(**config)
        self.config = config

    def parse(self, raw_document: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        if not raw_document:
            raise ValueError("raw_document cannot be None or empty")

        title_match = re.search(r"Title:\s*(.*?)\n", raw_document, re.DOTALL)
        passage_match = re.search(r"Passage:\s*(.*)", raw_document, re.DOTALL)

        title = title_match.group(1).strip() if title_match else ""
        passage = passage_match.group(1).strip() if passage_match else raw_document.strip()

        if title:
            if self.config.include_labels:
                content = f"Title: {title}\nPassage: {passage}"
            else:
                content = f"{title}\n\n{passage}"
        else:
            content = passage

        final_metadata = {**(metadata or {}), "parser_type": "title_passage_combined"}
        return Document(title=title, content=content, metadata=final_metadata)


_TRAILING_TABLE_LITERAL = re.compile(r"^(?P<title>.*?)(?P<table>\[\[.*\]\])$", re.DOTALL)


class TableHeadingParser:
    """Splits a heading off of an inline flattened table literal (used by
    tatqa, e.g. '25. Deferred income [[...]]')."""

    def parse(self, raw_document: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        if not raw_document:
            raise ValueError("raw_document cannot be None or empty")

        stripped = raw_document.strip()
        title = ""
        content = stripped

        match = _TRAILING_TABLE_LITERAL.match(stripped)
        if match and match.group("title").strip():
            title = match.group("title").strip()
            content = match.group("table")

        final_metadata = {**(metadata or {}), "parser_type": "table_heading"}
        return Document(title=title, content=content, metadata=final_metadata)


_PARSERS = {
    "noop": NoopParser,
    "title_passage_combined": TitlePassageCombinedParser,
    "table_heading": TableHeadingParser,
}


def create_parser(parser_type: str, config: Optional[dict] = None):
    parser_cls = _PARSERS[parser_type]
    return parser_cls(config) if config else parser_cls()


class DataProcessor:
    """Process raw ragbench samples into canonical Document objects.

    Ported from rag-foundry's data_sources/processors/data_processor.py.
    """

    def __init__(self, parser_strategy):
        self.parser_strategy = parser_strategy

    def process_dataset(self, dataset: list) -> list:
        documents = []
        for sample_idx, sample in enumerate(dataset):
            raw_docs = [
                doc.strip()
                for doc in sample.get("documents", [])
                if doc and doc.strip()
            ]
            for doc_idx, raw_doc in enumerate(raw_docs):
                metadata = {
                    "doc_id": f"sample_{sample_idx}_doc_{doc_idx}",
                    "sample_index": sample_idx,
                    "source": "ragbench",
                }
                document = self.parser_strategy.parse(raw_doc, metadata)
                documents.append(document)
        return documents
