"""Test doubles for the pipeline's LLM and Tavily backends."""

import json
from pathlib import Path
from typing import Any

from staleguard.llm import LLMCompletion

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(rel: str) -> Any:
    return json.loads((FIXTURES / rel).read_text(encoding="utf-8"))


class FakeLLM:
    """Replies by prompt type (detected from the rendered prompt), in order. Records every call."""

    MARKERS = {
        "extract": "atomic factual claims",
        "classify": "Classify each claim",
        "judge": "careful fact-checker",
        "rewrite": "editing an assistant's answer",
    }

    def __init__(self, fail_stages: tuple[str, ...] = (), **replies: list[str | dict]) -> None:
        self.replies = {k: list(v) for k, v in replies.items()}
        self.fail_stages = fail_stages
        self.calls: list[tuple[str, str, list[dict[str, str]]]] = []

    def _stage(self, prompt: str) -> str:
        return next(stage for stage, marker in self.MARKERS.items() if marker in prompt)

    async def complete(self, model: str, messages: list[dict[str, str]], reasoning_effort: str | None) -> LLMCompletion:
        stage = self._stage(messages[0]["content"])
        self.calls.append((stage, model, messages))
        if stage in self.fail_stages:
            raise RuntimeError(f"{stage} backend down")
        reply = self.replies[stage].pop(0)
        return LLMCompletion(text=reply if isinstance(reply, str) else json.dumps(reply), total_tokens=100)

    def stages(self) -> list[str]:
        return [c[0] for c in self.calls]


class FakeTavily:
    """Returns a recorded Tavily response per query (or a default). Records queries and kwargs."""

    def __init__(self, responses: dict[str, Any] | None = None, default: Any = None, fail: bool = False) -> None:
        self.responses = responses or {}
        self.default = default if default is not None else {"results": []}
        self.fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((query, kwargs))
        if self.fail:
            raise TimeoutError("tavily timed out")
        return self.responses.get(query, self.default)
