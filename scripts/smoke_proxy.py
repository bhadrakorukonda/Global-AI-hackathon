"""Smoke test for step 2: OpenAI SDK -> StaleGuard proxy -> Bifrost (semantic cache on).

Sends the same question twice and a paraphrase under a fresh cache key, then a streamed request.
Expect: miss -> direct hit -> semantic hit, all with x-staleguard-status: skipped.
Cleans up its cache entries afterwards.

Usage:  (StaleGuard running on :8000)  uv run python scripts/smoke_proxy.py
"""

import argparse
import asyncio
import sys
import time
import urllib.parse

import httpx
from openai import AsyncOpenAI

from staleguard.config import get_settings


async def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default="http://localhost:8000/v1")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.proxy, api_key="staleguard-local")
    cache_key = f"sg-smoke-{int(time.time())}"
    prompts = [
        ("miss", "What is the capital of Australia? One word."),
        ("direct hit", "What is the capital of Australia? One word."),
        ("semantic hit", "Which city is Australia's capital? Answer in one word."),
    ]
    ok = True
    for expected, prompt in prompts:
        raw = await client.chat.completions.with_raw_response.create(
            model=settings.small_model,
            messages=[{"role": "user", "content": prompt}],
            extra_headers={"x-bf-cache-key": cache_key},
            reasoning_effort="low",
        )
        resp = raw.parse()
        cd = (resp.to_dict().get("extra_fields") or {}).get("cache_debug") or {}
        got = "miss" if not cd.get("cache_hit") else f"{cd.get('hit_type')} hit"
        ok &= got == expected
        print(
            f"[{'ok ' if got == expected else 'BAD'}] expected={expected:12} got={got:12} "
            f"sim={cd.get('similarity')} cache_id={cd.get('cache_id')} "
            f"sg-status={raw.headers.get('x-staleguard-status')} reply={resp.choices[0].message.content!r}"
        )
        await asyncio.sleep(1)  # let Bifrost finish writing the entry

    stream = await client.chat.completions.create(
        model=settings.small_model, messages=[{"role": "user", "content": "Count 1 to 3."}], stream=True, reasoning_effort="low"
    )
    text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
    print(f"[ok ] stream passthrough: {text!r}")

    token = settings.bifrost_setup_token.get_secret_value() if settings.bifrost_setup_token else ""
    async with httpx.AsyncClient() as http:
        r = await http.delete(
            f"{settings.bifrost_url.rstrip('/')}/api/cache/clear-by-key/{urllib.parse.quote(cache_key, safe='')}",
            headers={"X-Bifrost-Setup-Token": token},
        )
    print(f"cleanup clear-by-key: HTTP {r.status_code}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
