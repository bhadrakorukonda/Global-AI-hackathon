"""Proxy passthrough tests. Bifrost is mocked with httpx.MockTransport replaying recorded fixtures."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from staleguard import main
from staleguard.bifrost_client import parse_cache_debug

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class FakeBifrost:
    """Records requests and replies with a canned response."""

    def __init__(self, handler):
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)


@pytest.fixture
def make_client():
    def _make(handler) -> tuple[TestClient, FakeBifrost]:
        fake = FakeBifrost(handler)
        http = httpx.AsyncClient(base_url="http://bifrost.test", transport=httpx.MockTransport(fake))
        main.app.dependency_overrides[main.get_bifrost_http] = lambda: http
        return TestClient(main.app), fake

    yield _make
    main.app.dependency_overrides.clear()


CHAT = {"model": "gemini/gemini-3.6-flash", "messages": [{"role": "user", "content": "What is the capital of Australia?"}]}


def test_forwards_body_and_bifrost_headers_only(make_client):
    client, fake = make_client(lambda r: httpx.Response(200, json=fixture("bifrost_chat_miss")))
    resp = client.post(
        "/v1/chat/completions",
        json=CHAT,
        headers={"x-bf-cache-key": "demo", "x-bf-cache-ttl": "1h", "authorization": "Bearer sk-client", "x-other": "1"},
    )
    assert resp.status_code == 200
    sent = fake.requests[0]
    assert sent.url.path == "/v1/chat/completions"
    assert json.loads(sent.content) == CHAT
    assert sent.headers["x-bf-cache-key"] == "demo"
    assert sent.headers["x-bf-cache-ttl"] == "1h"
    assert "authorization" not in sent.headers
    assert "x-other" not in sent.headers


def test_response_body_passes_through_unchanged(make_client):
    body = fixture("bifrost_chat_hit_semantic")
    client, _ = make_client(lambda r: httpx.Response(200, json=body, headers={"x-bifrost-cache-hit": "true"}))
    resp = client.post("/v1/chat/completions", json=CHAT, headers={"x-bf-cache-key": "demo"})
    assert resp.json() == body
    assert resp.headers["x-bifrost-cache-hit"] == "true"
    assert resp.headers["x-staleguard-status"] == "skipped"


def test_bifrost_error_status_passes_through(make_client):
    body = fixture("bifrost_chat_error")
    client, _ = make_client(lambda r: httpx.Response(body["status_code"], json=body))
    resp = client.post("/v1/chat/completions", json=CHAT)
    assert resp.status_code == body["status_code"]
    assert resp.json()["error"]["message"] == body["error"]["message"]


def test_bifrost_unreachable_returns_502(make_client):
    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    client, _ = make_client(boom)
    resp = client.post("/v1/chat/completions", json=CHAT)
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "bifrost_unreachable"


def test_invalid_json_body_is_400(make_client):
    client, fake = make_client(lambda r: httpx.Response(200, json={}))
    resp = client.post("/v1/chat/completions", content=b"not json", headers={"content-type": "application/json"})
    assert resp.status_code == 400
    assert fake.requests == []


def test_stream_is_passed_through_and_marked_skipped(make_client):
    sse = b'data: {"choices":[{"delta":{"content":"Can"}}]}\n\ndata: [DONE]\n\n'
    client, fake = make_client(lambda r: httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"}))
    resp = client.post("/v1/chat/completions", json=CHAT | {"stream": True})
    assert resp.status_code == 200
    assert resp.content == sse
    assert resp.headers["x-staleguard-status"] == "skipped"
    assert json.loads(fake.requests[0].content)["stream"] is True


@pytest.mark.parametrize(
    ("name", "hit", "hit_type"),
    [("bifrost_chat_miss", False, None), ("bifrost_chat_hit_direct", True, "direct"), ("bifrost_chat_hit_semantic", True, "semantic")],
)
def test_parse_cache_debug_from_recorded_responses(name, hit, hit_type):
    cd = parse_cache_debug(fixture(name))
    assert cd is not None
    assert cd.cache_hit is hit
    assert cd.hit_type == hit_type
    assert cd.cache_id  # present on misses too


def test_parse_cache_debug_absent_without_cache_key():
    assert parse_cache_debug({"choices": [], "extra_fields": {"provider": "gemini"}}) is None
    assert parse_cache_debug(["not", "a", "dict"]) is None
