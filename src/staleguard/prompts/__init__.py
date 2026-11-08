"""Versioned prompt templates (*.md). Placeholders are {{name}} so JSON examples need no escaping."""

import re
from functools import lru_cache
from pathlib import Path

__all__ = ["render_prompt"]

_DIR = Path(__file__).parent
_HEADER = re.compile(r"\A\s*<!--.*?-->\s*", re.DOTALL)
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


@lru_cache
def _load(name: str) -> str:
    """Read prompts/<name>.md, dropping the leading <!-- metadata --> comment."""
    return _HEADER.sub("", (_DIR / f"{name}.md").read_text(encoding="utf-8"))


def render_prompt(name: str, **variables: str) -> str:
    """Fill every {{placeholder}} in the named prompt. Missing or unused variables are errors."""
    template = _load(name)
    needed = set(_PLACEHOLDER.findall(template))
    if needed != set(variables):
        raise ValueError(f"prompt {name!r} needs {sorted(needed)}, got {sorted(variables)}")
    return _PLACEHOLDER.sub(lambda m: variables[m.group(1)], template)
