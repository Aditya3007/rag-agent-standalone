"""Finance table-aware chunking strategy. Ported verbatim from rag-foundry's
rag/modules/chunking/strategies/finance_table_aware/strategy.py.

Detects flattened list-of-lists table literals (RAGBench-style financial
documents) and renders them as natural language before falling back to
sentence-based chunking, so dense retrieval can actually find the numbers
it needs.
"""

import ast
import re

from rag_agent.chunking.sentence import SentenceChunkingStrategy
from rag_agent.models.chunk import Chunk
from rag_agent.models.document import Document


def _fold_context(*parts: str, max_words: int | None = None) -> str:
    cleaned = []
    for part in parts:
        part = re.sub(r"\.(?!\d)", ";", part.strip()).strip().rstrip(";").strip()
        if part:
            cleaned.append(part)
    joined = "; ".join(cleaned)
    if max_words is not None:
        joined = " ".join(joined.split()[:max_words])
    return joined


def _parse_table_rows(stripped: str) -> tuple[list, list[str]] | None:
    if not (stripped.startswith("[[") and stripped.endswith("]]")):
        return None

    try:
        data = ast.literal_eval(stripped)
    except (ValueError, SyntaxError, MemoryError, TypeError):
        return None

    if not isinstance(data, list) or len(data) < 2:
        return None
    if not all(isinstance(row, list) for row in data):
        return None

    header, *rows = data
    row_texts = []
    for row in rows:
        if not row:
            continue
        label = str(row[0]).strip()
        label = re.sub(r"\s+\.", ".", label)
        pairs = []
        for col_name, value in zip(header[1:], row[1:]):
            col_name = str(col_name).strip() or "value"
            pairs.append(f"{col_name} = {value}")
        if not pairs:
            continue
        prefix = f"{label}: " if label else ""
        row_texts.append(f"{prefix}{', '.join(pairs)}")

    if not row_texts:
        return None
    return header, row_texts


def _render_row_group(row_texts: list[str], entity_context: str, max_context_words: int | None = None) -> str:
    context = _fold_context(entity_context, max_words=max_context_words)
    prefix = f"Table data ({context}): " if context else "Table data: "
    return prefix + "; ".join(row_texts) + "."


def render_flattened_table_groups(
    text: str,
    entity_context: str = "",
    max_rows_per_chunk: int | None = None,
    max_context_words: int | None = None,
) -> list[str] | None:
    parsed = _parse_table_rows(text.strip())
    if parsed is None:
        return None
    _header, row_texts = parsed

    if max_rows_per_chunk is None or len(row_texts) <= max_rows_per_chunk:
        return [_render_row_group(row_texts, entity_context, max_context_words)]

    groups = [
        row_texts[i:i + max_rows_per_chunk]
        for i in range(0, len(row_texts), max_rows_per_chunk)
    ]
    return [_render_row_group(group, entity_context, max_context_words) for group in groups]


def render_flattened_table(text: str, entity_context: str = "", max_context_words: int | None = None) -> str | None:
    groups = render_flattened_table_groups(text, entity_context=entity_context, max_context_words=max_context_words)
    return groups[0] if groups else None


class FinanceTableAwareChunkingStrategy(SentenceChunkingStrategy):
    """Sentence chunking with financial-table normalization.

    Detects flattened list-of-lists table literals and rewrites them as
    natural-language sentences (preserving every original value) before
    applying standard sentence-based chunking. Non-table documents are
    chunked exactly as SentenceChunkingStrategy would.
    """

    def __init__(self, config):
        super().__init__(config)

    def chunk(self, document: Document) -> list[Chunk]:
        entity_context = document.metadata.get("entity_context", "")
        context = _fold_context(document.title or "", entity_context)
        texts = render_flattened_table_groups(
            document.content,
            entity_context=context,
            max_rows_per_chunk=self.config.max_rows_per_table_chunk,
            max_context_words=self.config.max_context_words,
        )
        if texts is None:
            return super().chunk(document)

        return [
            Chunk(
                text=text,
                metadata={
                    **document.metadata,
                    "chunk_type": "finance_table",
                    "word_count": len(text.split()),
                    "title": document.title,
                    "is_table": True,
                    **({"table_row_group": i} if len(texts) > 1 else {}),
                },
            )
            for i, text in enumerate(texts)
        ]
