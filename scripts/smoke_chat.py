"""Smoke test: one plain chat completion through Bifrost using the OpenAI SDK.

Usage:  uv run python scripts/smoke_chat.py [--model groq/openai/gpt-oss-120b]
"""

import argparse
import asyncio
import json
import sys
import time

from openai import APIStatusError

from staleguard.bifrost_client import make_llm_client
from staleguard.config import get_settings


async def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=settings.small_model)
    args = parser.parse_args()

    client = make_llm_client(settings)
    print(f"POST {settings.bifrost_openai_base_url}/chat/completions  model={args.model}")
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with exactly: Bifrost OK"}],
        )
    except APIStatusError as exc:
        print(f"FAILED: HTTP {exc.status_code}\n{json.dumps(exc.body, indent=2)}", file=sys.stderr)
        return 1
    ms = (time.perf_counter() - t0) * 1000

    body = resp.to_dict()
    print(f"reply   : {resp.choices[0].message.content!r}")
    print(f"model   : {resp.model}")
    print(f"usage   : {resp.usage.model_dump() if resp.usage else None}")
    print(f"latency : {ms:.0f} ms")
    print(f"extra_fields: {json.dumps(body.get('extra_fields'), indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
