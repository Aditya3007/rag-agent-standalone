"""LLM-based domain router: classifies a query into one of the known
domains via an LLM call, since retrieval itself differs per domain.
"""

import json
import re
from typing import Dict

from rag_agent.agent.domain_registry import DomainSpec


def _extract_json_candidate(raw: str) -> str:
    """Pull the JSON object out of a raw LLM response (reuses the same
    robust extraction approach as the TRACe evaluator)."""
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = raw.find("{")
    if start == -1:
        return raw
    candidate = raw[start:]
    depth = 0
    for i, ch in enumerate(candidate):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[: i + 1]
    return candidate


class DomainRoutingError(Exception):
    """Raised when the router can't confidently classify a query into a
    known domain. Deliberately not silently defaulted, so misrouting
    doesn't get masked."""


class LLMDomainRouter:
    """Classifies a query into one of the registered domains using an LLM."""

    def __init__(self, provider, model: str, domains: Dict[str, DomainSpec], temperature: float = 0.0):
        self.provider = provider
        self.model = model
        self.domains = domains
        self.temperature = temperature

    def _build_prompt(self, query: str) -> str:
        domain_lines = "\n".join(
            f'- "{key}": {spec.description}' for key, spec in self.domains.items()
        )
        valid_keys = ", ".join(f'"{k}"' for k in self.domains.keys())
        return f"""You are a routing assistant for a multi-domain question-answering system.
Given a user's question, decide which single knowledge domain it belongs to.

Available domains:
{domain_lines}

Question:
\"\"\"{query}\"\"\"

Respond with a JSON object matching this exact schema, and nothing else:
{{
  "domain": "<one of {valid_keys}>",
  "reason": "<one short sentence explaining the choice>"
}}"""

    def classify(self, query: str) -> str:
        """Return the domain key this query should be routed to.

        Raises DomainRoutingError if the model's answer can't be parsed
        or doesn't match a known domain.
        """
        prompt = self._build_prompt(query)

        response = self.provider.generate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        candidate = _extract_json_candidate(raw)

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise DomainRoutingError(f"Router did not return valid JSON:\n\n{raw}") from exc

        domain = parsed.get("domain")
        if domain not in self.domains:
            raise DomainRoutingError(
                f"Router returned unknown domain '{domain}'. "
                f"Known domains: {list(self.domains.keys())}. Raw response: {raw}"
            )

        return domain
