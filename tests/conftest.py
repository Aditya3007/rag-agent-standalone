"""Pytest configuration: load .env so network-gated tests (e.g.
test_agent_smoke.py) can pick up GROQ_API_KEY/HF_TOKEN without requiring
callers to export them manually."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
