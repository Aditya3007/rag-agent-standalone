"""Document processing steps. Consolidated/ported from rag-foundry's
data_sources/processors/strategies/{deduplication,sibling_entity_context}
and data_sources/processors/pipeline.py — the 2 steps used by the 4
pinned domain configs (deduplication at the experiment/domain level for
all 4; sibling_entity_context per-config for tatqa only).
"""

import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from rag_agent.models.document import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

@dataclass
class DeduplicationConfig:
    strategy: str = "content_hash"  # or "title_content_hash"


class DeduplicationStep:
    """Remove duplicate documents based on content hashing. Keeps the
    first occurrence and discards subsequent duplicates."""

    def __init__(self, config: DeduplicationConfig):
        self.config = config

    def _hash_key(self, doc: Document) -> str:
        if self.config.strategy == "title_content_hash":
            raw = f"{doc.title}||{doc.content}"
        else:
            raw = doc.content
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def process(self, documents: List[Document]) -> List[Document]:
        seen = set()
        unique = []
        for doc in documents:
            key = self._hash_key(doc)
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        removed = len(documents) - len(unique)
        if removed:
            logger.info(
                "Deduplication (%s): removed %d duplicates, %d -> %d documents",
                self.config.strategy, removed, len(documents), len(unique),
            )
        return unique


# ---------------------------------------------------------------------------
# Sibling entity context (tatqa only)
# ---------------------------------------------------------------------------

@dataclass
class SiblingEntityContextConfig:
    max_words: int = 20
    sample_index_key: str = "sample_index"
    use_positional_proximity: bool = False
    proximity_weight: float = 0.5


_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for",
    "is", "are", "was", "were", "be", "been", "by", "with", "as", "that",
    "this", "these", "those", "from", "we", "our", "it", "its", "their",
    "which", "than", "have", "has", "had", "not", "but", "also", "may",
})


def _is_flattened_table(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("[[") and stripped.endswith("]]")


def _content_keywords(text: str) -> set:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


_DOC_INDEX_RE = re.compile(r"_doc_(\d+)$")


def _doc_index(doc: Document) -> int | None:
    doc_id = doc.metadata.get("doc_id")
    if not isinstance(doc_id, str):
        return None
    match = _DOC_INDEX_RE.search(doc_id)
    return int(match.group(1)) if match else None


def _best_entity_snippet(
    table_doc: Document,
    siblings: List[Document],
    max_words: int,
    use_positional_proximity: bool = False,
    proximity_weight: float = 0.5,
) -> str:
    table_content = table_doc.content
    table_keywords = _content_keywords(table_content)
    if not table_keywords:
        return ""

    table_idx = _doc_index(table_doc) if use_positional_proximity else None

    best_sentence, best_score, best_overlap = "", float("-inf"), 0
    for sibling in siblings:
        sibling_idx = _doc_index(sibling) if use_positional_proximity else None
        distance = (
            abs(sibling_idx - table_idx)
            if table_idx is not None and sibling_idx is not None
            else 0
        )
        for sentence in re.split(r"(?<=[.!?])\s+", sibling.content):
            sentence = sentence.strip()
            if not sentence:
                continue
            overlap = len(table_keywords & _content_keywords(sentence))
            score = overlap - proximity_weight * distance if use_positional_proximity else overlap
            if score > best_score:
                best_score, best_overlap, best_sentence = score, overlap, sentence

    if best_overlap == 0:
        if use_positional_proximity and table_idx is not None:
            nearest = min(siblings, key=lambda s: abs((_doc_index(s) or 0) - table_idx))
            best_sentence = nearest.content
        else:
            best_sentence = siblings[0].content if siblings else ""

    return " ".join(best_sentence.split()[:max_words])


class SiblingEntityContextStep:
    """Attach entity-identifying context to generic tables from a sibling
    document in the same source sample."""

    def __init__(self, config: SiblingEntityContextConfig):
        self.config = config

    def process(self, documents: List[Document]) -> List[Document]:
        by_sample: dict = defaultdict(list)
        for doc in documents:
            by_sample[doc.metadata.get(self.config.sample_index_key)].append(doc)

        enriched = []
        enriched_count = 0
        for doc in documents:
            sample_id = doc.metadata.get(self.config.sample_index_key)
            if _is_flattened_table(doc.content) and sample_id is not None:
                siblings = [
                    s for s in by_sample.get(sample_id, [])
                    if s is not doc and not _is_flattened_table(s.content)
                ]
                if siblings:
                    snippet = _best_entity_snippet(
                        doc, siblings, self.config.max_words,
                        use_positional_proximity=self.config.use_positional_proximity,
                        proximity_weight=self.config.proximity_weight,
                    )
                    if snippet:
                        doc = Document(
                            title=doc.title,
                            content=doc.content,
                            metadata={**doc.metadata, "entity_context": snippet},
                        )
                        enriched_count += 1
            enriched.append(doc)

        logger.info(
            "SiblingEntityContext: attached entity context to %d/%d table documents",
            enriched_count, len(documents),
        )
        return enriched


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_STEP_TYPES = {
    "deduplication": (DeduplicationStep, DeduplicationConfig),
    "sibling_entity_context": (SiblingEntityContextStep, SiblingEntityContextConfig),
}


class ProcessingPipeline:
    """Runs a sequence of processing steps on a document list."""

    def __init__(self, steps: list):
        self.steps = steps

    def run(self, documents: List[Document]) -> List[Document]:
        for step in self.steps:
            before = len(documents)
            documents = step.process(documents)
            logger.debug(
                "ProcessingPipeline [%s]: %d -> %d documents",
                step.__class__.__name__, before, len(documents),
            )
        return documents

    @classmethod
    def from_config(cls, config: dict) -> "ProcessingPipeline":
        """Build a pipeline from a ``data_processing: {steps: [...]}`` dict."""
        steps_config = config.get("steps", [])
        steps = []
        for step_cfg in steps_config:
            step_type = step_cfg.get("type")
            step_config = step_cfg.get("config", {})
            step_cls, config_cls = _STEP_TYPES[step_type]
            steps.append(step_cls(config_cls(**step_config)))
        return cls(steps)
