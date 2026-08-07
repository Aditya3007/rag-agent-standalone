"""Default generation strategy. Ported verbatim from rag-foundry's
rag/modules/generation/strategies/default/strategy.py."""

import re

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_reasoning_trace(content: str) -> str:
    """Strip a closed <think>...</think> reasoning block, if present.

    Some reasoning-capable models emit their chain-of-thought inline in
    the response content rather than a separate structured field. Left
    in place, this gets scored by TRACe's judge as if it were part of
    the answer.
    """
    if "<think>" not in content:
        return content
    stripped = _THINK_BLOCK_RE.sub("", content)
    return stripped.strip() if stripped.strip() else content


class DefaultGenerationStrategy:
    """Default generation strategy that generates answers from context and query.

    Uses a provider (groq, ollama, ...) as a pluggable component.
    """

    def __init__(self, config, provider):
        self.config = config
        self.provider = provider

    def generate(self, query: str, context: str) -> str:
        messages = []

        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": """You are a knowledgeable and reliable AI assistant specializing in question answering.

Your primary objective is to provide accurate, concise, and well-supported answers based on the supplied context.

Guidelines:
- Base your answer only on the provided context.
- Do not use external knowledge or make unsupported assumptions.
- If the provided context does not contain enough information to answer the question, clearly state that the answer cannot be determined from the provided context.
- Do not fabricate facts or citations.
- Keep the answer focused on the user's question.
- Prefer factual correctness over completeness.
- If multiple pieces of retrieved context are relevant, synthesize them into a coherent answer.""",
            })

        user_prompt_content = f"""You are given a set of retrieved context passages and a question.

Retrieved Context:
--------------------
{context}
--------------------

Question:
{query}

Using only the retrieved context, answer the question as accurately and concisely as possible.

If the answer cannot be determined from the provided context, explicitly state that the information is not available in the retrieved context.

Answer:"""

        if self.config.user_prompt:
            user_prompt_content = self.config.user_prompt

        user_prompt = {
            "role": "user",
            "content": user_prompt_content.format(context=context, query=query),
        }

        messages.append(user_prompt)

        extra_kwargs = {}
        if self.config.reasoning_effort is not None:
            extra_kwargs["reasoning_effort"] = self.config.reasoning_effort

        response = self.provider.generate(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **extra_kwargs,
        )

        return _strip_reasoning_trace(response.choices[0].message.content)
