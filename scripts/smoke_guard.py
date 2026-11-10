"""Smoke test for step 5, the headline flow: a stale answer in Bifrost's semantic cache gets caught and evicted.

1. Seed: ask Bifrost directly (no StaleGuard). The model answers from its training data and Bifrost caches it.
   This stands in for "an answer cached last month".
2. Ask a paraphrase through StaleGuard (annotate): semantic cache hit -> claim STALE -> entry evicted.
3. Confirm the eviction: re-ask Bifrost directly with x-bf-cache-no-store -> must be a miss.
4. Same question through StaleGuard in rewrite mode, then block mode.
Cleans up its cache key afterwards.

Usage:  (Bifrost up; StaleGuard on :8000)  uv run python scripts/smoke_guard.py
"""

import argparse
import asyncio
import sys
import time
import urllib.parse
from typing import Any

import httpx

from staleguard.config import get_settings
from staleguard.llm import message_text

SEED_Q = "What is the latest stable version of Python?"
PARAPHRASE_Q = "Which Python version is the newest stable release?"


def chat_body(model: str, question: str) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": question}], "reasoning_effort": "low"}


def cache_debug(body: dict[str, Any]) -> dict[str, Any]:
    return (body.get("extra_fields") or {}).get("cache_debug") or {}


def show_report(resp: httpx.Response) -> dict[str, Any]:
    body = resp.json()
    sg = body.get("staleguard") or {}
    cd = cache_debug(body)
    print(f"  HTTP {resp.status_code}  bifrost: hit={cd.get('cache_hit')} type={cd.get('hit_type')} sim={cd.get('similarity')}")
    print(f"  headers: status={resp.headers.get('x-staleguard-status')} action={resp.headers.get('x-staleguard-action')} "
          f"evicted={resp.headers.get('x-staleguard-cache-evicted')} error={resp.headers.get('x-staleguard-error')}")
    if sg:
        print(f"  staleguard: {sg['latency_ms']} ms, tavily {sg['tavily_calls']}, verdict-cache hits {sg['verdict_cache_hits']}, "
              f"tokens {sg['tokens_by_stage']}, models {sg['models_by_stage']}")
        for c in sg["claims"]:
            fix = f"  ->  {c['corrected_claim']}" if c.get("corrected_claim") else ""
            print(f"    {c['verdict']:12} conf={c['confidence']:.2f}  {c['claim']['text']}{fix}")
            for e in c.get("evidence", []):
                print(f"        - {e.get('published_date')}  {e['url']}")
    print("  answer: " + (message_text(body["choices"][0]["message"]["content"]) or "").replace("\n", "\n          "))
    return body


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # model output has non-cp1252 chars (e.g. U+2011) on Windows consoles
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default="http://localhost:8000")
    args = parser.parse_args()

    bifrost = settings.bifrost_url.rstrip("/")
    key = f"sg-guard-{int(time.time())}"
    model = settings.small_model
    token = settings.bifrost_setup_token.get_secret_value() if settings.bifrost_setup_token else ""
    ok = True

    async with httpx.AsyncClient(timeout=300) as http:
        print(f"1) seed Bifrost's cache directly (key {key})")
        seed = (await http.post(f"{bifrost}/v1/chat/completions", json=chat_body(model, SEED_Q), headers={"x-bf-cache-key": key})).json()
        print(f"  cached as {cache_debug(seed).get('cache_id')}: {message_text(seed['choices'][0]['message']['content'])!r}")
        await asyncio.sleep(2)  # let Bifrost finish writing the entry

        print("\n2) paraphrase through StaleGuard (annotate)")
        resp = await http.post(f"{args.proxy}/v1/chat/completions", json=chat_body(model, PARAPHRASE_Q), headers={"x-bf-cache-key": key})
        sg = show_report(resp).get("staleguard") or {}
        ok &= bool(sg.get("bifrost_cache_hit"))
        if sg.get("verdict_summary") != "stale":
            print("  NOTE: the model's answer was not stale this time; nothing to evict. Re-run, or the model is up to date.")
            ok = False
        else:
            ok &= bool(sg.get("cache_evicted"))

        print("\n3) confirm eviction: same seed question, direct to Bifrost, no-store")
        check = (await http.post(f"{bifrost}/v1/chat/completions", json=chat_body(model, SEED_Q),
                                 headers={"x-bf-cache-key": key, "x-bf-cache-no-store": "true"})).json()
        hit = cache_debug(check).get("cache_hit")
        print(f"  cache_hit={hit}  ({'evicted, as expected' if not hit else 'STILL CACHED'})")
        ok &= not hit

        for mode in ("rewrite", "block"):
            print(f"\n4) through StaleGuard, mode={mode}")
            resp = await http.post(f"{args.proxy}/v1/chat/completions", json=chat_body(model, PARAPHRASE_Q),
                                   headers={"x-bf-cache-key": key, "x-sg-mode": mode})
            show_report(resp)
            await asyncio.sleep(1)

        r = await http.delete(f"{bifrost}/api/cache/clear-by-key/{urllib.parse.quote(key, safe='')}",
                              headers={"X-Bifrost-Setup-Token": token})
        print(f"\ncleanup clear-by-key: HTTP {r.status_code}")
    print("\nRESULT:", "ok" if ok else "check output above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
