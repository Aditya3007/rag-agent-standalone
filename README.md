# RAG Agent (Standalone)

A self-contained, independently deployable RAG agent that:

1. Classifies an incoming query into one of 4 domains — **pubmedqa**,
   **covidqa**, **finqa**, **tatqa** — using an LLM (retrieval itself
   differs per domain, so routing happens before retrieval).
2. Runs that domain's **pinned, best-performing** retrieval + generation
   pipeline (chunk → embed → dense+sparse search → fuse → [expand] →
   rerank → generate).
3. Evaluates the generated answer with a single shared **TRACe**
   LLM-judge evaluator (same evaluator config for every domain), plus
   supplementary **RGB**-benchmark-style scoring (rejection detection,
   factual-error detection, and their optional LLM-judge variants).
4. Returns the answer plus **every** score: TRACe's relevance,
   utilization, completeness, adherence (+ full raw judge output), and
   RGB's accuracy/rejection/factual-error/fake-answer-parroted fields.

Unlike rag-foundry's original RGB evaluator (which scores against RGB's
own benchmark dataset's ground-truth/counterfactual metadata), the RGB
scoring here runs purely on the agent's own generated answer —
`rejection_detected`/`factual_error_detected` (+ optional LLM-judge
variants) — with no dependency on any dataset-specific ground truth, so
it applies uniformly regardless of which domain a query was routed to.

This package has **no runtime import dependency on the parent
`rag-foundry` repository** — every strategy implementation it needs was
copied and trimmed into `rag_agent/`, and the winning config per domain
is pinned as a literal YAML file under `configs/`. It still talks to
external services at runtime: the HuggingFace Hub (to load each domain's
corpus) and Groq's API (generation, query transform, routing, and
evaluation — no local model server required for any domain).

## How the pinned configs were chosen

Each domain in `rag-foundry/rag-experiments/<domain>-experiment/` had
many tuned config variants, evaluated and recorded in a
`reports/comparison.csv`. The winner per domain was picked with:

```
overall = relevance_mean + utilization_mean + completeness_mean + 1.5 * adherence_mean
```

(adherence weighted higher, matching that repo's own analysis docs,
which treat "don't hallucinate beyond the retrieved context" as the most
important signal). The winners, pinned here as-is:

| Domain | Config | File |
|---|---|---|
| pubmedqa | `pubmedqa_title_aware_v22_precise_citation` | `configs/pubmedqa.yaml` |
| covidqa | `covidqa_title_aware_v3` | `configs/covidqa.yaml` |
| finqa | `finqa_v8_strict_meta_free` | `configs/finqa.yaml` |
| tatqa | `tatqa_v1` | `configs/tatqa.yaml` |

If you re-run the analysis later and want a different winner, just
replace/edit the corresponding YAML file under `configs/` — no code
changes needed.

## Setup

```bash
cd rag-agent-standalone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY
```

## Usage

```bash
python scripts/ask.py "What is the mechanism of action of aspirin?"
```

Or from Python:

```python
from rag_agent.agent.rag_agent import RagAgent

agent = RagAgent()  # builds/loads all 4 domain indices
result = agent.ask("What is the mechanism of action of aspirin?")
agent.print_result(result)

print(result.domain)         # e.g. "pubmedqa"
print(result.answer)
print(result.scores)         # TRACe: relevance_score, utilization_score, completeness_score, adherence_score, judge_output
print(result.rgb_scores)     # RGB: rejection_detected, factual_error_detected, rejection_judge, factual_error_judge
```

## Retrieval API

Retrieval (chunking, embedding, vector store, BM25, and the full query
transform → search → fusion → expansion → rerank pipeline) can be run as
its own FastAPI service, separate from routing/generation/evaluation.
This lets you deploy/scale the stateful, resource-heavy retrieval layer
independently of the agent process(es) that call it.

Start the retrieval API (builds/loads all 4 domain indices on startup,
same cost as `RagAgent()` normally pays):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# or: python -m api.main
```

Endpoints:

* `GET /health` — readiness + which domains are loaded
* `GET /domains` — registered domains (key, description, pinned config name)
* `POST /retrieve` — `{"domain": "pubmedqa", "query": "..."}` → `{"results": [{"text", "metadata", "score", "rerank_score"}, ...]}`

Point the agent (CLI, Python, or the Gradio UI) at it by setting
`RAG_RETRIEVAL_API_URL` before constructing `RagAgent`:

```bash
export RAG_RETRIEVAL_API_URL=http://localhost:8000
python scripts/ask.py "What is the mechanism of action of aspirin?"
```

or in `.env`:

```
RAG_RETRIEVAL_API_URL=http://localhost:8000
```

When this is set, `RagAgent` skips loading/chunking/embedding any
corpus itself — it only registers providers and builds each domain's
generator locally, and calls the API over HTTP for retrieval
(`rag_agent/pipeline/remote_pipeline.py` /
`rag_agent/pipeline/remote_retrieval_client.py`). Routing, generation,
and evaluation are unaffected either way. Leave `RAG_RETRIEVAL_API_URL`
unset to keep the original fully in-process behavior — this is
purely additive.

## Agent API

The agent itself (routing → retrieval → generation → evaluation) can
also be run as its own FastAPI service instead of via the CLI/Gradio UI
— useful for calling it from another backend service, or for deploying
it separately from the retrieval API above.

```bash
uvicorn agent_api.main:app --host 0.0.0.0 --port 8080
# or: python -m agent_api.main
```

Endpoints:

* `GET /health` — readiness, loaded domains, and whether it's using a remote retrieval API
* `POST /ask` — `{"query": "..."}` → the full `AgentResult` as JSON (domain, config_name, answer, retrieved_docs, scores, rgb_scores, latencies)

Set `RAG_RETRIEVAL_API_URL` on this service (same env var as above) to
have it call the retrieval API instead of indexing corpora itself —
this is the intended split for the two-service GCP deployment below.

## Web UI (Gradio)

```bash
python ui/app.py
```

Opens a browser UI at `http://localhost:7860` with a question box, example
questions per domain, and panels for the routing decision, the answer,
retrieved documents, TRACe scores, RGB scores, and per-stage latencies.
The `RagAgent` is built once (lazily, on the first submitted question)
and reused for every subsequent question in that running process.

## First-run cost

`RagAgent()` eagerly builds an index for **all 4 domains** at
construction time: it downloads each domain's HuggingFace corpus slice,
chunks it, embeds every chunk, and builds a FAISS index (plus a BM25
index for sparse search). This can take a while the first time
(PubMedQA alone produces ~5-6k chunks). Every stage is cached
content-addressed under `./cache/` (`chunks/`, `embeddings/`,
`indexes/`), so subsequent runs — and container restarts, if `./cache`
is a mounted volume — reuse the cache and start quickly.

## No local model server required

All 4 domains run entirely on Groq's hosted API — no Ollama (or any
other local model server) is needed. Two of the four pinned "winning"
configs (`finqa`, `tatqa`) were originally benchmarked with a local
Ollama model (`gpt-oss:20b` for generation, `llama3.1:8b` for finqa's
query_transform); since Groq now hosts OpenAI's gpt-oss models directly,
those two configs use the Groq-hosted equivalents instead
(`openai/gpt-oss-20b`, `llama-3.1-8b-instant`), so only a single
`GROQ_API_KEY` is needed for every domain. `reasoning_effort: low` is
set on both gpt-oss-backed generation configs to keep its
chain-of-thought token usage small (see
`DefaultGenerationConfig.reasoning_effort`).

## Docker

```bash
docker build -t rag-agent-standalone .
docker run --rm -it \
  -e GROQ_API_KEY=... \
  -v $(pwd)/cache:/app/cache \
  rag-agent-standalone "What is the mechanism of action of aspirin?"
```

## Package layout

```
rag_agent/
  config.py, loader.py       # trimmed RAGConfig dataclasses + YAML loader
  providers/                 # groq provider client (+ provider_manager; ollama_provider.py kept but unused by the pinned configs)
  cache/                     # content-addressed chunk/embedding/index cache
  models/                    # Document, Chunk, QueryResult
  chunking/                  # sentence, finance_table_aware
  embedding/                 # sentence_transformer
  vectorstore/               # faiss
  search/                    # dense, sparse (+ bm25_store)
  fusion/                    # noop, rrf, weighted_sum
  expansion/                 # noop, sibling
  query_transform/           # noop, multi_query
  reranking/                 # cross_encoder
  generation/                # default
  evaluation/                # trace (TRACe LLM-judge)
  dataset/                   # HF loader, parsers (noop/title_passage_combined/table_heading), processors (deduplication/sibling_entity_context)
  pipeline/                  # RAGPipeline, RetrievalPipeline, SearchPipeline, RemoteBackedPipeline, RemoteRetrievalClient
  agent/                     # DomainSpec/domain_registry, LLMDomainRouter, RagAgent
configs/                     # pinned per-domain RAGConfig YAMLs + domains.yaml + evaluation.yaml
api/main.py                    # standalone retrieval FastAPI service (see "Retrieval API")
scripts/ask.py                # CLI entry point
ui/app.py                     # Gradio web UI
tests/                        # config round-trip + smoke tests
```

Only the specific strategies the 4 pinned configs actually use were
ported (e.g. no cohere/voyage/jina/mixedbread reranking, no hyde/step_back
query transforms, no fixed_window/token/semantic chunking) — this is a
deliberately minimal copy, not a full port of rag-foundry's pluggable
registries.
