"""Structured-JSON LLM calls through Bifrost, with Pydantic validation and one retry on parse failure."""

import json
import logging
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from staleguard.prompts import render_prompt

__all__ = [
    "ChatBackend",
    "LLMCompletion",
    "LLMOutputError",
    "OpenAIChatBackend",
    "UsageTracker",
    "message_text",
    "structured_call",
]

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class LLMCompletion(BaseModel):
    text: str
    total_tokens: int = 0
    model_used: str | None = Field(default=None, description="provider/model that answered (differs on Bifrost failover).")


class ChatBackend(Protocol):
    """Minimal chat interface so pipeline stages can be tested without the OpenAI SDK."""

    async def complete(self, model: str, messages: list[dict[str, str]], reasoning_effort: str | None) -> LLMCompletion: ...


class OpenAIChatBackend:
    """ChatBackend over the OpenAI SDK pointed at Bifrost (never at a provider directly).

    `fallbacks` maps a model to Bifrost request-level fallbacks (body field `fallbacks`, verified live:
    docs/bifrost-notes.md). Bifrost reports the model that actually answered in extra_fields.routing_info.
    """

    def __init__(self, client: AsyncOpenAI, fallbacks: Mapping[str, Sequence[str]] | None = None) -> None:
        self._client = client
        self._fallbacks = {m: list(f) for m, f in (fallbacks or {}).items() if f}

    async def complete(self, model: str, messages: list[dict[str, str]], reasoning_effort: str | None) -> LLMCompletion:
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        extra_body = {"fallbacks": self._fallbacks[model]} if model in self._fallbacks else None
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},
            # Deterministic as possible: stable claim wording keeps verdict-cache keys stable across runs.
            temperature=0,
            extra_body=extra_body,
            **extra,  # type: ignore[arg-type]
        )
        return LLMCompletion(
            text=message_text(resp.choices[0].message.content) or "",
            total_tokens=resp.usage.total_tokens if resp.usage else 0,
            model_used=_routed_model(resp.model_extra) or model,
        )


def message_text(content: Any) -> str | None:
    """Text of an assistant message. Bifrost returns Gemini replies as a list of text parts, not a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "".join(parts) if parts else None
    return None


def _routed_model(extra: dict[str, Any] | None) -> str | None:
    info = ((extra or {}).get("extra_fields") or {}).get("routing_info") or {}
    provider, model = info.get("provider"), info.get("model")
    return f"{provider}/{model}" if provider and model else None


class UsageTracker:
    """Accumulates tokens and the answering model per stage (feeds Report.tokens_by_stage / models_by_stage)."""

    def __init__(self) -> None:
        self._tokens: defaultdict[str, int] = defaultdict(int)
        self._models: dict[str, str] = {}

    def add(self, stage: str, tokens: int, model: str | None = None) -> None:
        self._tokens[stage] += tokens
        if model:
            self._models[stage] = model

    def snapshot(self) -> dict[str, int]:
        return dict(self._tokens)

    def models(self) -> dict[str, str]:
        return dict(self._models)


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
        usage.add(stage, completion.total_tokens, completion.model_used or model)
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
