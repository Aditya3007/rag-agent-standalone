"""Gradio UI for the standalone RAG domain-routing agent.

Launch with:
    python ui/app.py

The underlying RagAgent is built once (lazily, on the first query) and
reused for every subsequent request in the running process — building it
eagerly indexes all 4 domains, which can take a while on a cold cache
(see README.md "First-run cost").
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from rag_agent.agent.rag_agent import AgentResult, RagAgent

load_dotenv()

_agent: RagAgent | None = None
_agent_lock = threading.Lock()

EXAMPLE_QUESTIONS = [
    "Do surface porosity and pore size influence the mechanical properties of PEEK implants?",
    "What are the known risk factors for severe COVID-19 outcomes?",
    "What was the year-over-year change in net revenue?",
    "What was the net change in deferred income for the period?",
]


def _get_agent() -> RagAgent:
    """Build the RagAgent once (thread-safe) and reuse it afterward."""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = RagAgent()
    return _agent


def _retrieved_docs_dataframe(result: AgentResult) -> pd.DataFrame:
    rows = []
    for i, doc in enumerate(result.retrieved_docs, 1):
        rows.append({
            "#": i,
            "score": round(doc.get("score", 0.0), 4),
            "rerank_score": round(doc["rerank_score"], 4) if "rerank_score" in doc else None,
            "text": (doc["text"][:300] + "...") if len(doc["text"]) > 300 else doc["text"],
        })
    return pd.DataFrame(rows)


def ask(query: str, progress=gr.Progress()):
    """Handle one query: route -> answer -> evaluate -> format for the UI."""
    if not query or not query.strip():
        raise gr.Error("Please enter a question.")

    progress(0, desc="Preparing agent (first run builds/loads all 4 domain indices — this can take a while)...")
    agent = _get_agent()

    progress(0.5, desc="Routing, retrieving, generating, and evaluating...")
    result = agent.ask(query.strip())

    domain_md = f"**Domain:** `{result.domain}`  \n**Config:** `{result.config_name}`"
    trace_scores = {k: v for k, v in result.scores.items() if k != "judge_output"}

    return (
        domain_md,
        result.answer,
        _retrieved_docs_dataframe(result),
        trace_scores,
        result.rgb_scores,
        {k: round(v, 2) for k, v in result.latencies.items()},
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG Domain Agent") as demo:
        gr.Markdown(
            "# RAG Domain-Routing Agent\n"
            "Ask a question. It will be routed to the right domain "
            "(**pubmedqa**, **covidqa**, **finqa**, **tatqa**), answered "
            "using that domain's pinned best-performing pipeline, and "
            "evaluated with **TRACe** and **RGB**-style scoring."
        )

        with gr.Row():
            query_box = gr.Textbox(
                label="Question",
                placeholder="e.g. What is the mechanism of action of aspirin?",
                lines=2,
                scale=4,
            )
            ask_button = gr.Button("Ask", variant="primary", scale=1)

        gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=query_box, label="Example questions")

        domain_output = gr.Markdown(label="Routing")
        answer_output = gr.Textbox(label="Answer", lines=8, interactive=False)

        with gr.Row():
            trace_output = gr.JSON(label="TRACe Scores")
            rgb_output = gr.JSON(label="RGB Scores")
            latency_output = gr.JSON(label="Latencies (ms)")

        retrieved_output = gr.Dataframe(
            label="Retrieved Documents",
            headers=["#", "score", "rerank_score", "text"],
            wrap=True,
        )

        ask_button.click(
            fn=ask,
            inputs=query_box,
            outputs=[domain_output, answer_output, retrieved_output, trace_output, rgb_output, latency_output],
        )
        query_box.submit(
            fn=ask,
            inputs=query_box,
            outputs=[domain_output, answer_output, retrieved_output, trace_output, rgb_output, latency_output],
        )

    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="0.0.0.0", server_port=7860)
