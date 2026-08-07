# Standalone RAG domain-routing agent.
#
# One image, three ways to run it (override CMD/entrypoint per use case —
# this is also how the two Cloud Run services in README.md "Deploying to
# GCP" are built from a single image):
#
# Build:
#   docker build -t rag-agent-standalone .
#
# Run the CLI (requires GROQ_API_KEY; mount ./cache as a volume to
# persist embeddings/index across container restarts — the first run
# downloads and embeds all 4 domains' corpora, which can take a while):
#   docker run --rm -it \
#     -e GROQ_API_KEY=... \
#     -v $(pwd)/cache:/app/cache \
#     rag-agent-standalone "What is the mechanism of action of aspirin?"
#
# Run the Gradio UI instead (overrides the default CLI entrypoint):
#   docker run --rm -it \
#     -e GROQ_API_KEY=... \
#     -v $(pwd)/cache:/app/cache \
#     -p 7860:7860 \
#     --entrypoint python \
#     rag-agent-standalone ui/app.py
#
# Run the retrieval API (see README.md "Retrieval API"):
#   docker run --rm -it \
#     -e GROQ_API_KEY=... \
#     -v $(pwd)/cache:/app/cache \
#     -p 8000:8000 \
#     --entrypoint uvicorn \
#     rag-agent-standalone api.main:app --host 0.0.0.0 --port 8000
#
# Run the agent API (see README.md "Agent API"), pointed at the
# retrieval API above via RAG_RETRIEVAL_API_URL:
#   docker run --rm -it \
#     -e GROQ_API_KEY=... \
#     -e RAG_RETRIEVAL_API_URL=http://<retrieval-host>:8000 \
#     -p 8080:8080 \
#     --entrypoint uvicorn \
#     rag-agent-standalone agent_api.main:app --host 0.0.0.0 --port 8080
#
# All 4 domains run entirely on Groq's hosted API (including the
# finqa/tatqa configs, which use Groq's hosted openai/gpt-oss-20b rather
# than a local Ollama model) — no local model server needed, just
# GROQ_API_KEY.

FROM python:3.11-slim

WORKDIR /app

# System deps for faiss / torch / sentence-transformers wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-fetch the NLTK sentence tokenizer used by the TRACe evaluator.
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY rag_agent/ rag_agent/
COPY configs/ configs/
COPY scripts/ scripts/
COPY ui/ ui/
COPY api/ api/
COPY agent_api/ agent_api/

VOLUME ["/app/cache"]
EXPOSE 7860

ENTRYPOINT ["python", "scripts/ask.py"]
