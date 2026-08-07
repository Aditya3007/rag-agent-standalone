"""RGB-benchmark-style evaluation strategy, adapted to score the agent's
own generated answer rather than RGB's own benchmark dataset.

Ported from rag-foundry's evaluation/strategies/rgb/strategy.py, but with
the ground-truth-dependent parts removed: the original RGB implementation
pulls ``rgb_ground_truth``/``rgb_fakeanswer`` out of retrieved-doc
metadata that only exists for RGB's own benchmark dataset assembly step
(``rgb_assembly`` processor). None of the 4 domains this agent serves
(pubmedqa/covidqa/finqa/tatqa) have that metadata, so ``accuracy`` and
``fake_answer_parroted`` would always be meaningless placeholders here.

Instead, this strategy runs purely on the query + generated answer for
*any* RAG pipeline output: rejection detection (did the model say it
couldn't answer?) and factual-error detection (did the model flag
contradictions in its own context?), each available as a fast
keyword-based signal and an optional LLM-judge signal.

Note: the original rag-foundry implementation calls
``self.provider.generate(prompt=..., ...)``, which doesn't match this
package's ``BaseLLMProvider.generate(model, messages, **kwargs)``
signature (a latent bug there, only exercised when ``use_llm_judge`` is
enabled). Fixed here to pass ``messages=[...]`` so the LLM-judge path
actually works.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RGBEvaluationConfig:
    """Configuration for the RGB-style evaluation strategy."""
    use_llm_judge: bool = False
    judge_model: Optional[str] = None
    judge_temperature: float = 0.0
    judge_max_tokens: int = 500
    rejection_keywords: list = field(default_factory=lambda: [
        "insufficient information",
        "can not answer",
        "cannot answer",
    ])
    factual_error_keywords: list = field(default_factory=lambda: [
        "factual errors",
        "factual error",
    ])


class RGBEvaluationStrategy:
    """Scores a single generated answer for rejection / factual-error
    signals, RGB-benchmark style. Runs on whatever query+answer it's
    given — no dependency on RGB's own benchmark dataset."""

    REJECTION_JUDGE_PROMPT = (
        "I will provide a question and an answer generated based on the given text. "
        "Please determine if the answer indicates that the given text is insufficient "
        "to answer the question. If the answer suggests that the given text cannot answer "
        "the question, respond with 'yes'. If the answer attempts to answer the question "
        "based on the given text, respond with 'no'. Please respond with 'yes' or 'no' only."
        "\n\nQuestion: {query}\nAnswer: {response}\n"
    )

    FACTUAL_ERROR_JUDGE_PROMPT = (
        "I will provide a question and an answer generated based on the given text. "
        "Please determine if the answer indicates that there are factual errors in the "
        "given text. If the answer suggests that the given text contains factual errors, "
        "respond with 'yes'. If the answer does not mention factual errors, respond with 'no'. "
        "Please respond with 'yes' or 'no' only."
        "\n\nQuestion: {query}\nAnswer: {response}\n"
    )

    def __init__(self, config: RGBEvaluationConfig, provider=None):
        self.config = config
        self.provider = provider

    def evaluate(self, query: str, retrieved_docs=None, response: str = "") -> dict:
        """Evaluate a single generated answer.

        ``retrieved_docs`` is accepted for interface parity with other
        evaluators (e.g. TRACe) but is not used — this scores the answer
        text itself, not anything dataset-specific.
        """
        rejection_detected = self._detect_rejection(response)
        factual_error_detected = self._detect_factual_error(response)

        scores = {
            "rejection_detected": rejection_detected,
            "factual_error_detected": factual_error_detected,
            "rejection_judge": None,
            "factual_error_judge": None,
        }

        if self.config.use_llm_judge and self.provider:
            scores["rejection_judge"] = self._judge_rejection(query, response)
            scores["factual_error_judge"] = self._judge_factual_error(query, response)

        return scores

    def _detect_rejection(self, response: str) -> bool:
        response_lower = response.lower()
        return any(kw.lower() in response_lower for kw in self.config.rejection_keywords)

    def _detect_factual_error(self, response: str) -> bool:
        response_lower = response.lower()
        return any(kw.lower() in response_lower for kw in self.config.factual_error_keywords)

    def _judge_rejection(self, query: str, response: str) -> bool:
        try:
            prompt = self.REJECTION_JUDGE_PROMPT.format(query=query, response=response)
            result = self.provider.generate(
                model=self.config.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.judge_temperature,
                max_tokens=self.config.judge_max_tokens,
            )
            content = result.choices[0].message.content
            return "yes" in content.lower()
        except Exception as e:
            logger.warning("Rejection judge failed: %s", e)
            return False

    def _judge_factual_error(self, query: str, response: str) -> bool:
        try:
            prompt = self.FACTUAL_ERROR_JUDGE_PROMPT.format(query=query, response=response)
            result = self.provider.generate(
                model=self.config.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.judge_temperature,
                max_tokens=self.config.judge_max_tokens,
            )
            content = result.choices[0].message.content
            return "yes" in content.lower()
        except Exception as e:
            logger.warning("Factual error judge failed: %s", e)
            return False
