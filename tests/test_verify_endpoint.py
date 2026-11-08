"""/verify API-layer tests: request validation, 503 without Tavily, report serialization."""

import pytest
from fakes import FakeLLM, FakeTavily
from fastapi.testclient import TestClient

from staleguard import main
from staleguard.pipeline import Pipeline, PipelineDeps


@pytest.fixture
def client():
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_verify_returns_report(client):
    llm = FakeLLM(extract=[{"claims": ["A triangle has three sides."]}], classify=[{"items": [{"id": 1, "claim_class": "static"}]}])
    pipeline = Pipeline(PipelineDeps(llm=llm, tavily=FakeTavily(), small_model="s", judge_model="j"))
    main.app.dependency_overrides[main.get_pipeline] = lambda: pipeline
    resp = client.post("/verify", json={"text": "A triangle has three sides."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict_summary"] == "clean"
    assert body["tavily_calls"] == 0 and body["cache_evicted"] is False


def test_verify_rejects_empty_text(client):
    main.app.dependency_overrides[main.get_pipeline] = lambda: object()
    assert client.post("/verify", json={"text": ""}).status_code == 422


def test_verify_503_when_tavily_not_configured(client):
    main.app.state.pipeline = None
    resp = client.post("/verify", json={"text": "x"})
    assert resp.status_code == 503
