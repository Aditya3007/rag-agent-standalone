"""TRACe evaluation strategy. Ported verbatim from rag-foundry's
evaluation/strategies/trace/strategy.py.

Evaluates answer quality via an LLM judge, checking whether response
sentences are supported by the retrieved documents. This is the single
evaluator instance shared across all domains in the standalone agent.
"""

import json
import re


class TRACeEvaluationStrategy:
    """TRACe evaluation strategy using LLM-based evaluation."""

    def __init__(self, config, provider):
        self.config = config
        self.provider = provider

    def split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import nltk
        for resource in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                import ssl
                try:
                    _create_unverified = ssl._create_unverified_context
                except AttributeError:
                    nltk.download(resource, quiet=True)
                else:
                    _prev = ssl._create_default_https_context
                    ssl._create_default_https_context = _create_unverified
                    try:
                        nltk.download(resource, quiet=True)
                    finally:
                        ssl._create_default_https_context = _prev

        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text.strip())
        return [s.strip() for s in sentences if len(s.strip()) >= 10]

    def build_doc_sentences(self, retrieved_docs) -> list[list]:
        """Build document sentences with keys."""
        doc_sentences = []
        for doc_idx, doc in enumerate(retrieved_docs):
            chunk = doc.get("chunk")
            text = doc.get("text") or doc.get("content") or ""
            if not text and chunk is not None:
                text = getattr(chunk, "text", "")
            sentences = self.split_into_sentences(text)
            doc_sent_pairs = []

            for sent_idx, sentence in enumerate(sentences):
                key = f"d{doc_idx}s{sent_idx}"
                doc_sent_pairs.append([key, sentence])

            doc_sentences.append(doc_sent_pairs)
        return doc_sentences

    def build_response_sentences(self, response: str) -> list[list]:
        """Build response sentences with keys."""
        sentences = self.split_into_sentences(response)
        response_sentences = []
        for idx, sentence in enumerate(sentences):
            letter = chr(ord("a") + idx % 26)
            response_sentences.append([letter, sentence])
        return response_sentences

    def format_doc_sentences(self, doc_sentences) -> str:
        """Format document sentences for prompt."""
        formatted = ""
        for doc in doc_sentences:
            for key, sentence in doc:
                formatted += f"{key}. {sentence}\n"
        return formatted.strip()

    def format_response_sentences(self, resp_sentences) -> str:
        """Format response sentences for prompt."""
        formatted = ""
        for key, sentence in resp_sentences:
            formatted += f"{key}. {sentence}\n"
        return formatted.strip()

    @staticmethod
    def _extract_json_candidate(raw: str) -> str:
        """Pull the JSON object out of a raw LLM response.

        Prefers a fenced ```json``` block if present, otherwise finds the
        first balanced ``{...}`` span. If braces never balance (the
        response was cut off mid-object), returns everything from the
        first ``{`` onward so callers can attempt a repair.
        """
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
                    return candidate[:i + 1]
        return candidate

    @staticmethod
    def _repair_truncated_json(candidate: str) -> str | None:
        """Best-effort repair for a JSON object cut off mid-generation."""
        depth = 0
        in_string = False
        escape = False
        last_safe_cut = None
        last_safe_stack = None
        stack = []

        for i, ch in enumerate(candidate):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
                depth += 1
            elif ch in "}]":
                if stack:
                    stack.pop()
                depth -= 1
            elif ch == "," and depth == 1:
                last_safe_cut = i
                last_safe_stack = list(stack)

        if last_safe_cut is None:
            return None

        truncated = candidate[:last_safe_cut]
        closers = {"{": "}", "[": "]"}
        closing = "".join(closers[c] for c in reversed(last_safe_stack))
        return truncated + closing

    def _call_judge_and_parse(self, prompt: str, max_tokens: int, _attempt: int = 1) -> tuple[dict, str]:
        """Call the judge model and parse its JSON response.

        Retries once with a larger token budget if the response looks
        truncated, then falls back to a best-effort repair of the
        truncated JSON before giving up.
        """
        result = self.provider.generate(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.config.temperature,
        )

        choice = result.choices[0]
        raw = choice.message.content.strip()
        finish_reason = getattr(choice, "finish_reason", None)

        candidate = self._extract_json_candidate(raw)

        try:
            return json.loads(candidate), raw
        except json.JSONDecodeError:
            pass

        looks_truncated = finish_reason == "length" or not candidate.rstrip().endswith("}")
        if looks_truncated and _attempt == 1:
            return self._call_judge_and_parse(prompt, min(max_tokens * 2, 8000), _attempt=2)

        repaired = self._repair_truncated_json(candidate)
        if repaired is not None:
            try:
                return json.loads(repaired), raw
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Judge did not return valid JSON:\n\n{raw}\n\nParsed candidate:\n\n{candidate}"
        )

    def evaluate(self, query: str, retrieved_docs, response: str) -> dict:
        """Evaluate answer quality using LLM.

        Returns a dict with relevance_score, utilization_score,
        completeness_score, adherence_score, and the raw judge_output.
        """
        doc_sentences = self.build_doc_sentences(retrieved_docs)
        resp_sentences = self.build_response_sentences(response)

        doc_fmt = self.format_doc_sentences(doc_sentences)
        resp_fmt = self.format_response_sentences(resp_sentences)

        prompt = f"""I asked someone to answer a question based on one or more documents.
Your task is to review their response and assess whether or not each sentence in that response is supported by text in the documents. And if so, which sentences in the documents provide that support. You will also tell me which of the documents contain useful information for answering the question, and which of the documents the answer was sourced from.

Here are the documents, each of which is split into sentences. Alongside each sentence is associated key, such as '0a.' or '0b.' that you can use to refer to it:

'''
{doc_fmt}
'''

The question was:

'''
{query}
'''

Here is their response, split into sentences. Alongside each sentence is associated key, such as 'a.' or 'b.' that you can use to refer to it. Note that these keys are unique to the response, and are not related to the keys in the documents:

'''
{resp_fmt}
'''

You must respond with a JSON object matching this schema:
'''
{{
  "relevance_explanation": "string",
  "all_relevant_sentence_keys": ["string"],
  "overall_supported_explanation": "string",
  "overall_supported": boolean,
  "sentence_support_information": [
    {{
      "response_sentence_key": "string",
      "explanation": "string",
      "supporting_sentence_keys": ["string"],
      "fully_supported": boolean
    }}
  ],
  "all_utilized_sentence_keys": ["string"]
}}
'''

The all_relevant_sentence_keys field: list all document sentence keys relevant to the question (even if unused in the response).
The all_utilized_sentence_keys field: list all document sentence keys actually used to construct the answer.
The overall_supported field: true only if ALL response sentences are fully supported by the documents.

Keep every "explanation" field to one short sentence (under 20 words) — you have a limited output budget and the JSON must not be truncated.

Respond with valid JSON only. No text before or after the JSON."""

        judge_output, raw = self._call_judge_and_parse(prompt, self.config.max_tokens)

        total_doc_sentences = sum(len(doc) for doc in doc_sentences)
        relevant_keys = set(judge_output.get("all_relevant_sentence_keys", []))
        utilized_keys = set(judge_output.get("all_utilized_sentence_keys", []))
        relevant_utilized = relevant_keys & utilized_keys

        relevance_score = len(relevant_keys) / total_doc_sentences if total_doc_sentences > 0 else 0.0
        utilization_score = len(utilized_keys) / total_doc_sentences if total_doc_sentences > 0 else 0.0
        completeness_score = len(relevant_utilized) / len(relevant_keys) if relevant_keys else 0.0
        adherence_score = bool(judge_output.get("overall_supported", False))

        return {
            "relevance_score": round(relevance_score, 4),
            "utilization_score": round(utilization_score, 4),
            "completeness_score": round(completeness_score, 4),
            "adherence_score": adherence_score,
            "judge_output": judge_output,
        }
