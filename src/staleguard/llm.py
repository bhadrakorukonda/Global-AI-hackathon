"""Structured-JSON LLM calls through Bifrost, with Pydantic validation and one retry on parse failure."""

import json
import logging
import re
from collections import defaultdict
from typing import Protocol, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from staleguard.prompts import render_prompt

__all__ = ["ChatBackend", "LLMCompletion", "LLMOutputError", "OpenAIChatBackend", "UsageTracker", "structured_call"]

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class LLMCompletion(BaseModel):
    text: str
    total_tokens: int = 0


class ChatBackend(Protocol):
    """Minimal chat interface so pipeline stages can be tested without the OpenAI SDK."""

    async def complete(self, model: str, messages: list[dict[str, str]], reasoning_effort: str | None) -> LLMCompletion: ...


class OpenAIChatBackend:
    """ChatBackend over the OpenAI SDK pointed at Bifrost (never at a provider directly)."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def complete(self, model: str, messages: list[dict[str, str]], reasoning_effort: str | None) -> LLMCompletion:
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
            **extra,  # type: ignore[arg-type]
        )
        return LLMCompletion(
            text=resp.choices[0].message.content or "",
            total_tokens=resp.usage.total_tokens if resp.usage else 0,
        )


class UsageTracker:
    """Accumulates total tokens per pipeline stage (feeds Report.tokens_by_stage)."""

    def __init__(self) -> None:
        self._tokens: defaultdict[str, int] = defaultdict(int)

    def add(self, stage: str, tokens: int) -> None:
        self._tokens[stage] += tokens

    def snapshot(self) -> dict[str, int]:
        return dict(self._tokens)


class LLMOutputError(RuntimeError):
    """The model returned invalid structured output twice in a row."""


def _parse(text: str, output_model: type[T]) -> T:
    match = _FENCE.match(text)
    return output_model.model_validate(json.loads(match.group(1) if match else text))


async def structured_call(
    backend: ChatBackend,
    *,
    model: str,
    prompt: str,
    variables: dict[str, str],
    output_model: type[T],
    stage: str,
    usage: UsageTracker,
    reasoning_effort: str | None = None,
) -> T:
    """Render `prompt`, call the model, validate against `output_model`; retry once with the error fed back."""
    messages = [{"role": "user", "content": render_prompt(prompt, **variables)}]
    last_error: Exception | None = None
    for attempt in (1, 2):
        completion = await backend.complete(model, messages, reasoning_effort)
        usage.add(stage, completion.total_tokens)
        try:
            return _parse(completion.text, output_model)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            log.warning("stage %s: invalid JSON on attempt %d: %s", stage, attempt, exc)
            messages = messages + [
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": f"That was not valid. Error: {exc}. Reply with only the corrected JSON."},
            ]
    raise LLMOutputError(f"stage {stage}: invalid structured output after retry: {last_error}")
