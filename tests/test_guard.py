"""Proxy + verification (step 5): annotate / rewrite / block, Bifrost cache eviction, fail-open, failover.

Bifrost is an httpx.MockTransport replaying recorded chat responses; LLM and Tavily are faked
(Tavily replies are recorded live responses).
"""

import asyncio
import json
from datetime import date
from pathlib import Path

import httpx
import pytest
from fakes import FakeLLM, FakeTavily, load_fixture
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from staleguard import main
from staleguard.config import Settings
from staleguard.llm import OpenAIChatBackend, message_text
from staleguard.pipeline import Pipeline, PipelineDeps
from staleguard.pipeline.act import actionable_stale, block_message
from staleguard.schemas import Claim, ClaimClass, ClaimVerdict, Evidence, Report, Verdict

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 11, 9)
PY_CLAIM = "The latest stable version of Python is 3.12."
STALE_ANSWER = "The latest stable version of Python is 3.12. You can download it from python.org."
CHAT = {"model": "gemini/gemini-3.5-flash-lite", "messages": [{"role": "user", "content": "What is the latest Python?"}]}
CACHE_ID = "9a40b08a-e760-59ac-8a14-d9f6d6554892"  # cache_id in the recorded semantic-hit fixture


def bifrost_reply(name: str = "bifrost_chat_hit_semantic", content: object = STALE_ANSWER) -> dict:
    body = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    body["choices"][0]["message"]["content"] = content
    return body


def stale_llm(confidence: float = 0.9, **extra) -> FakeLLM:
    return FakeLLM(
        extract=[{"claims": [PY_CLAIM]}],
        judge=[{"verdicts": [{"id": 1, "verdict": "STALE", "corrected_claim": "The latest stable version of Python is 3.15.",
                              "evidence_ids": [2], "confidence": confidence, "rationale": "3.15 is out."}]}],
        **extra,
    )


class FakeBifrost:
    """Chat POSTs get `chat_reply`; cache DELETEs get `evict_status`. Records every request."""

    def __init__(self, chat_reply: dict, chat_status: int = 200, evict_status: int = 200) -> None:
        self.chat_reply, self.chat_status, self.evict_status = chat_reply, chat_status, evict_status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(self.evict_status, json={"message": "ok"})
        return httpx.Response(self.chat_status, json=self.chat_reply)

    @property
    def deletes(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "DELETE"]


@pytest.fixture
def setup():
    def _setup(bifrost: FakeBifrost, llm: FakeLLM | None, tavily: FakeTavily | None = None) -> TestClient:
        http = httpx.AsyncClient(base_url="http://bifrost.test", transport=httpx.MockTransport(bifrost))
        pipeline = None
        if llm is not None:
            tavily = tavily or FakeTavily(default=load_fixture("tavily/general_python_version.json"))
            pipeline = Pipeline(PipelineDeps(llm=llm, tavily=tavily, small_model="small", judge_model="judge", today=lambda: TODAY))
        settings = Settings(_env_file=None, bifrost_setup_token="setup-tok")
        main.app.dependency_overrides |= {
            main.get_bifrost_http: lambda: http,
            main.get_optional_pipeline: lambda: pipeline,
            main.get_settings: lambda: settings,
        }
        return TestClient(main.app)

    yield _setup
    main.app.dependency_overrides.clear()


def post(client: TestClient, mode: str | None = None, **headers: str) -> httpx.Response:
    hdrs = {"x-bf-cache-key": "demo"} | headers
    if mode:
        hdrs["x-sg-mode"] = mode
    return client.post("/v1/chat/completions", json=CHAT, headers=hdrs)


# ------------------------------------------------------------------ annotate + eviction


def test_stale_cache_hit_is_annotated_and_evicted(setup):
    bifrost = FakeBifrost(bifrost_reply())
    resp = post(setup(bifrost, stale_llm()))
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == STALE_ANSWER  # annotate leaves the answer alone
    sg = body["staleguard"]
    assert sg["verdict_summary"] == "stale" and sg["mode"] == "annotate" and sg["action"] == "none"
    assert sg["bifrost_cache_hit"] is True
    assert sg["cache_evicted"] is True and sg["evicted_cache_id"] == CACHE_ID
    assert sg["claims"][0]["corrected_claim"].endswith("3.15.")
    assert resp.headers["x-staleguard-status"] == "stale"
    assert resp.headers["x-staleguard-claims-checked"] == "1"
    assert resp.headers["x-staleguard-cache-evicted"] == "true"
    [delete] = bifrost.deletes
    assert delete.url.path == f"/api/cache/clear/{CACHE_ID}"
    assert delete.headers["x-bifrost-setup-token"] == "setup-tok"


def test_stale_answer_on_cache_miss_is_also_evicted(setup):
    bifrost = FakeBifrost(bifrost_reply("bifrost_chat_miss"))
    sg = post(setup(bifrost, stale_llm())).json()["staleguard"]
    assert sg["bifrost_cache_hit"] is False and sg["cache_evicted"] is True
    assert len(bifrost.deletes) == 1


def test_clean_answer_is_not_evicted(setup):
    llm = FakeLLM(extract=[{"claims": ["Canberra is the capital of Australia."]}],
                  classify=[{"items": [{"id": 1, "claim_class": "static"}]}])
    bifrost = FakeBifrost(bifrost_reply(content="Canberra is the capital of Australia."))
    resp = post(setup(bifrost, llm))
    assert resp.headers["x-staleguard-status"] == "clean"
    assert resp.json()["staleguard"]["cache_evicted"] is False
    assert bifrost.deletes == []


def test_no_cache_key_means_nothing_to_evict(setup):
    reply = bifrost_reply()
    del reply["extra_fields"]["cache_debug"]
    bifrost = FakeBifrost(reply)
    resp = setup(bifrost, stale_llm()).post("/v1/chat/completions", json=CHAT)
    sg = resp.json()["staleguard"]
    assert sg["verdict_summary"] == "stale" and sg["bifrost_cache_hit"] is None and sg["cache_evicted"] is False
    assert bifrost.deletes == []


def test_failed_eviction_does_not_fail_the_request(setup):
    bifrost = FakeBifrost(bifrost_reply(), evict_status=500)
    resp = post(setup(bifrost, stale_llm()))
    assert resp.status_code == 200
    assert resp.json()["staleguard"]["cache_evicted"] is False
    assert resp.headers["x-staleguard-cache-evicted"] == "false"


def test_gemini_list_content_is_verified(setup):
    parts = [{"type": "text", "text": "The latest stable version of Python is "}, {"type": "text", "text": "3.12."}]
    llm = stale_llm()
    resp = post(setup(FakeBifrost(bifrost_reply(content=parts)), llm))
    assert resp.json()["staleguard"]["verdict_summary"] == "stale"
    assert "The latest stable version of Python is 3.12." in llm.calls[0][2][0]["content"]


# ------------------------------------------------------------------ rewrite / block


def test_rewrite_mode_replaces_answer_and_keeps_original(setup):
    fixed = "The latest stable version of Python is 3.15 [1]. You can download it from python.org.\n\nSources:\n[1] History of Python — https://en.wikipedia.org/wiki/History_of_Python"
    llm = stale_llm(rewrite=[{"answer": fixed}])
    resp = post(setup(FakeBifrost(bifrost_reply()), llm), mode="rewrite")
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == fixed
    sg = body["staleguard"]
    assert sg["action"] == "rewritten" and sg["original_answer"] == STALE_ANSWER
    assert sg["tokens_by_stage"]["rewrite"] == 100 and sg["models_by_stage"]["rewrite"] == "judge"
    assert resp.headers["x-staleguard-action"] == "rewritten"
    rewrite_prompt = llm.calls[-1][2][0]["content"]
    assert "3.15" in rewrite_prompt and "https://en.wikipedia.org/wiki/History_of_Python" in rewrite_prompt
    assert "What is the latest Python?" in rewrite_prompt


def test_rewrite_failure_returns_annotated_original(setup):
    llm = stale_llm(fail_stages=("rewrite",))
    resp = post(setup(FakeBifrost(bifrost_reply()), llm), mode="rewrite")
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == STALE_ANSWER
    assert resp.json()["staleguard"]["action"] == "none"
    assert resp.headers["x-staleguard-error"] == "rewrite failed: RuntimeError"


def test_block_mode_replaces_answer_with_refusal(setup):
    llm = stale_llm()
    resp = post(setup(FakeBifrost(bifrost_reply()), llm), mode="block")
    content = resp.json()["choices"][0]["message"]["content"]
    assert content.startswith("StaleGuard withheld this answer")
    assert "3.15" in content and "https://en.wikipedia.org/wiki/History_of_Python" in content
    assert resp.json()["staleguard"]["action"] == "blocked"
    assert "rewrite" not in llm.stages()  # block never calls a model


@pytest.mark.parametrize("mode", ["rewrite", "block"])
def test_low_confidence_stale_is_reported_but_not_acted_on(setup, mode):
    llm = stale_llm(confidence=0.5)
    bifrost = FakeBifrost(bifrost_reply())
    resp = post(setup(bifrost, llm), mode=mode)
    sg = resp.json()["staleguard"]
    assert resp.json()["choices"][0]["message"]["content"] == STALE_ANSWER
    assert sg["verdict_summary"] == "stale" and sg["action"] == "none"
    assert sg["cache_evicted"] is True  # eviction ignores the threshold
    assert "rewrite" not in llm.stages()


# ------------------------------------------------------------------ skip / fail-open / validation


def test_verification_failure_fails_open(setup):
    llm = FakeLLM(fail_stages=("extract",))
    bifrost = FakeBifrost(bifrost_reply())
    resp = post(setup(bifrost, llm))
    assert resp.status_code == 200
    assert resp.json() == bifrost_reply()  # untouched, no staleguard key
    assert resp.headers["x-staleguard-status"] == "skipped"
    assert resp.headers["x-staleguard-error"] == "verification failed: RuntimeError"


def test_tool_call_reply_is_skipped_without_llm_calls(setup):
    llm = FakeLLM()
    resp = post(setup(FakeBifrost(bifrost_reply(content=None)), llm))
    assert resp.headers["x-staleguard-status"] == "skipped"
    assert llm.calls == []


def test_upstream_error_passes_through_unverified(setup):
    err = json.loads((FIXTURES / "bifrost_chat_error.json").read_text(encoding="utf-8"))
    llm = FakeLLM()
    resp = post(setup(FakeBifrost(err, chat_status=429), llm))
    assert resp.status_code == 429 and resp.headers["x-staleguard-status"] == "skipped"
    assert llm.calls == []


def test_without_tavily_answers_pass_through_skipped(setup):
    resp = post(setup(FakeBifrost(bifrost_reply()), llm=None))
    assert resp.headers["x-staleguard-status"] == "skipped"
    assert "staleguard" not in resp.json()


def test_invalid_mode_is_400_and_not_forwarded(setup):
    bifrost = FakeBifrost(bifrost_reply())
    resp = post(setup(bifrost, stale_llm()), mode="nuke")
    assert resp.status_code == 400 and "x-sg-mode" in resp.json()["error"]["message"]
    assert bifrost.requests == []


def test_include_domains_header_reaches_tavily(setup):
    tavily = FakeTavily(default=load_fixture("tavily/general_python_version.json"))
    post(setup(FakeBifrost(bifrost_reply()), stale_llm(), tavily), **{"x-sg-include-domains": "python.org, docs.python.org"})
    [(_, kwargs)] = tavily.calls
    assert kwargs["include_domains"] == ["python.org", "docs.python.org"]


def test_sg_headers_are_not_forwarded_to_bifrost(setup):
    bifrost = FakeBifrost(bifrost_reply())
    post(setup(bifrost, stale_llm()), mode="annotate")
    chat = bifrost.requests[0]
    assert "x-sg-mode" not in chat.headers and chat.headers["x-bf-cache-key"] == "demo"


# ------------------------------------------------------------------ act helpers


def _verdict(v: Verdict, conf: float, corrected: str | None = "new") -> ClaimVerdict:
    return ClaimVerdict(claim=Claim(id=1, text="old", claim_class=ClaimClass.VERSION), verdict=v, confidence=conf,
                        corrected_claim=corrected, evidence=[Evidence(url="https://u", title="T", snippet="s")])


def test_actionable_stale_filters_by_verdict_threshold_and_correction():
    claims = [_verdict(Verdict.STALE, 0.9), _verdict(Verdict.STALE, 0.6), _verdict(Verdict.STALE, 0.9, None),
              _verdict(Verdict.CURRENT, 0.99)]
    report = Report(verdict_summary="stale", claims=claims, claims_extracted=4, claims_checked=4, latency_ms=0,
                    tokens_by_stage={}, tavily_calls=0)
    assert actionable_stale(report, 0.7) == [claims[0]]


def test_block_message_lists_claim_correction_and_source():
    msg = block_message([_verdict(Verdict.STALE, 0.9)])
    assert '"old" is out of date. Current: new' in msg and "T — https://u" in msg


# ------------------------------------------------------------------ LLM backend: failover + Gemini parts


@pytest.mark.parametrize(
    ("content", "text"),
    [("plain", "plain"), ([{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}], "ab"),
     (None, None), ([], None)],
)
def test_message_text(content, text):
    assert message_text(content) == text


def test_backend_sends_bifrost_fallbacks_and_reports_routed_model():
    sent: list[dict] = []
    reply = bifrost_reply("bifrost_chat_miss", content='{"claims": []}')
    reply["extra_fields"]["routing_info"] = {"provider": "groq", "model": "openai/gpt-oss-20b", "is_fallback": True}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=reply)

    client = AsyncOpenAI(base_url="http://bifrost.test/v1", api_key="x",
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    backend = OpenAIChatBackend(client, fallbacks={"gemini/flash": ["groq/openai/gpt-oss-20b"], "other": []})
    out = asyncio.run(backend.complete("gemini/flash", [{"role": "user", "content": "hi"}], None))
    assert sent[0]["fallbacks"] == ["groq/openai/gpt-oss-20b"]
    assert out.model_used == "groq/openai/gpt-oss-20b" and out.text == '{"claims": []}'

    asyncio.run(backend.complete("other", [{"role": "user", "content": "hi"}], "low"))
    assert "fallbacks" not in sent[1] and sent[1]["reasoning_effort"] == "low"
