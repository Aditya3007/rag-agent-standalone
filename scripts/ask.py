#!/usr/bin/env python
"""CLI entry point for the standalone RAG domain-routing agent.

Usage:
    python scripts/ask.py "What is the mechanism of action of aspirin?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from rag_agent.agent.rag_agent import RagAgent


def main() -> None:
    load_dotenv()

    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "<question>"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print("Building domain pipelines (this can take a while on first run)...")
    agent = RagAgent()

    result = agent.ask(query)
    agent.print_result(result)


if __name__ == "__main__":
    main()
