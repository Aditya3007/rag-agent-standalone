"""Unit tests for RGBEvaluationStrategy — scores the agent's own
generated answer directly, with no dependency on RGB's benchmark dataset
ground truth. No network/provider calls needed for the keyword-based
signals (LLM-judge path is exercised separately via a fake provider)."""

from rag_agent.evaluation.rgb import RGBEvaluationConfig, RGBEvaluationStrategy


def test_rejection_detection_generic():
    evaluator = RGBEvaluationStrategy(RGBEvaluationConfig(use_llm_judge=False), provider=None)

    scores = evaluator.evaluate(
        query="What is the treatment?",
        response="The retrieved context does not provide sufficient information to answer this question.",
    )
    assert scores["rejection_detected"] is False  # keyword list doesn't match this exact phrasing
    assert scores["factual_error_detected"] is False
    assert scores["rejection_judge"] is None
    assert scores["factual_error_judge"] is None


def test_rejection_detection_matches_keyword():
    evaluator = RGBEvaluationStrategy(RGBEvaluationConfig(use_llm_judge=False), provider=None)

    scores = evaluator.evaluate(query="q", response="I cannot answer this question from the given text.")
    assert scores["rejection_detected"] is True


def test_factual_error_detection_matches_keyword():
    evaluator = RGBEvaluationStrategy(RGBEvaluationConfig(use_llm_judge=False), provider=None)

    scores = evaluator.evaluate(query="q", response="Note: there are factual errors in the provided documents.")
    assert scores["factual_error_detected"] is True


def test_evaluate_does_not_require_retrieved_docs():
    """RGB scoring here runs on query+answer only — retrieved_docs is
    accepted for interface parity but never touched."""
    evaluator = RGBEvaluationStrategy(RGBEvaluationConfig(), provider=None)
    scores = evaluator.evaluate(query="q", response="Some answer.")
    assert "rejection_detected" in scores
    assert "factual_error_detected" in scores


def test_llm_judge_uses_messages_call_signature():
    class FakeChoice:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})

    class FakeResult:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeProvider:
        def __init__(self):
            self.calls = []

        def generate(self, model, messages, **kwargs):
            self.calls.append((model, messages, kwargs))
            return FakeResult("yes")

    provider = FakeProvider()
    evaluator = RGBEvaluationStrategy(
        RGBEvaluationConfig(use_llm_judge=True, judge_model="fake"), provider=provider
    )
    scores = evaluator.evaluate(query="q", response="no info available")

    assert scores["rejection_judge"] is True
    assert scores["factual_error_judge"] is True
    assert len(provider.calls) == 2
    for _, messages, _ in provider.calls:
        assert messages[0]["role"] == "user"
