"""End-to-end smoke test for RagAgent.ask(). Network-gated: requires a
real GROQ_API_KEY (all 4 domains run on Groq's hosted API, no other
model server needed) plus downloading/embedding a real corpus, so it's
skipped by default.

Run explicitly with:
    RUN_AGENT_SMOKE_TEST=1 pytest tests/test_agent_smoke.py -v
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_AGENT_SMOKE_TEST"),
    reason="Set RUN_AGENT_SMOKE_TEST=1 to run this (downloads real corpora, calls real APIs).",
)


def test_ask_routes_and_scores():
    from rag_agent.agent.rag_agent import RagAgent

    agent = RagAgent()
    result = agent.ask("What is the mechanism of action of aspirin?")

    assert result.domain in agent.domains
    assert result.answer
    assert "relevance_score" in result.scores
    assert "utilization_score" in result.scores
    assert "completeness_score" in result.scores
    assert "adherence_score" in result.scores
    assert "rejection_detected" in result.rgb_scores
    assert "factual_error_detected" in result.rgb_scores
