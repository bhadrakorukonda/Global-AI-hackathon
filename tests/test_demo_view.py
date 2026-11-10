"""Demo client (demo/guard_view.py): request headers, response parsing, cost footer. No network."""

import asyncio
import json

import httpx
from guard_view import ask_guarded, cost_rows, parse_exchange, seed_cache

CLAIM = {
    "claim": {"id": 1, "text": "The latest stable Python is 3.12.", "claim_class": "version"},
    "verdict": "STALE",
    "corrected_claim": "The latest stable Python is 3.15.",
    "confidence": 0.93,
    "evidence": [{"url": "https://www.python.org/downloads/", "title": "Downloads", "snippet": "…", "published_date": "2026-10-07"}],
}
REPORT = {
    "verdict_summary": "stale", "claims": [CLAIM], "claims_extracted": 3, "claims_checked": 1, "latency_ms": 4000,
    "tokens_by_stage": {"extract": 900, "judge": 1400}, "models_by_stage": {"judge": "groq/openai/gpt-oss-120b"},
    "tavily_calls": 1, "cache_evicted": True, "evicted_cache_id": "abc", "bifrost_cache_hit": True,
}


def body(answer: str = "Python 3.12 is the latest.", **report: object) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": [{"type": "text", "text": answer}]}}],
        "extra_fields": {"cache_debug": {"cache_hit": True, "hit_type": "semantic", "similarity": 0.96, "cache_id": "abc"}},
        "staleguard": REPORT | report,
    }


def test_parse_annotated_stale_cache_hit():
    ex = parse_exchange("q", "annotate", 200, {"x-staleguard-status": "stale"}, body(), wall_ms=5000)
    assert ex.status == "stale" and ex.model_answer == ex.final_answer == "Python 3.12 is the latest."
    assert ex.cache.hit_type == "semantic" and ex.report.cache_evicted and ex.upstream_ms == 1000
    assert ex.report.claims[0].corrected_claim.endswith("3.15.")


def test_parse_rewrite_keeps_original_on_the_left():
    b = body("Python 3.15 is the latest [1].", mode="rewrite", action="rewritten", original_answer="Python 3.12 is the latest.")
    ex = parse_exchange("q", "rewrite", 200, {}, b, wall_ms=5000)
    assert ex.model_answer == "Python 3.12 is the latest." and ex.final_answer.startswith("Python 3.15")
    assert ex.status == "stale"  # falls back to the report when the header is missing


def test_parse_fail_open_and_errors_never_raise():
    passthrough = {"choices": [{"message": {"content": "hi"}}]}
    ex = parse_exchange("q", "annotate", 200, {"x-staleguard-status": "skipped", "x-staleguard-error": "TimeoutError"}, passthrough, 10)
    assert ex.report is None and ex.status == "skipped" and ex.error == "TimeoutError" and ex.final_answer == "hi"
    assert parse_exchange("q", "annotate", 502, {}, {"error": "bifrost down"}, 10).error.startswith("HTTP 502")
    assert parse_exchange("q", "annotate", 200, {}, "<html>", 10).error.startswith("Unexpected response")


def test_ask_sends_policy_headers():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body(), headers={"x-staleguard-status": "stale"})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await ask_guarded(http, "http://sg/", model="m", question="q", mode="rewrite", cache_key="k",
                                     include_domains=["python.org", "nodejs.org"])

    ex = asyncio.run(go())
    req = seen[0]
    assert str(req.url) == "http://sg/v1/chat/completions" and json.loads(req.content)["model"] == "m"
    assert (req.headers["x-sg-mode"], req.headers["x-bf-cache-key"], req.headers["x-sg-include-domains"]) == (
        "rewrite", "k", "python.org,nodejs.org")
    assert ex.status == "stale" and ex.wall_ms >= 0


def test_ask_without_cache_key_sends_no_cache_header_and_reports_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "x-bf-cache-key" not in request.headers
        raise httpx.ConnectError("refused", request=request)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await ask_guarded(http, "http://sg", model="m", question="q", mode="annotate")

    ex = asyncio.run(go())
    assert ex.http_status == 0 and "Can't reach StaleGuard" in ex.error


def test_seed_cache_goes_straight_to_bifrost():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "bifrost" and request.headers["x-bf-cache-key"] == "k"
        return httpx.Response(200, json=body("Node 22."))

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await seed_cache(http, "http://bifrost:8080", model="m", question="q", cache_key="k")

    assert asyncio.run(go()) == ("Node 22.", "abc")


def test_cost_rows():
    ex = parse_exchange("q", "annotate", 200, {}, body(), 5000)
    rows = dict(cost_rows(ex.report))
    assert rows["Tokens"] == "2,300 (extract 900, judge 1,400)" and rows["Tavily searches"] == "1"
    assert rows["Claims"] == "1 checked of 3 extracted" and "groq/openai/gpt-oss-120b" in rows["Models"]
