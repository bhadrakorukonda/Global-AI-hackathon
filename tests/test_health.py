from fastapi.testclient import TestClient

from staleguard import main
from staleguard.config import Settings, get_settings
from staleguard.schemas import DependencyHealth


def _client(monkeypatch, bifrost_ok: bool) -> TestClient:
    async def fake_check(settings: Settings, timeout_s: float = 3.0) -> DependencyHealth:
        return DependencyHealth(ok=bifrost_ok, url=f"{settings.bifrost_url}/health")

    monkeypatch.setattr(main, "check_bifrost", fake_check)
    main.app.dependency_overrides[get_settings] = lambda: Settings(bifrost_url="http://bifrost.test")
    return TestClient(main.app)


def test_health_ok_when_bifrost_up(monkeypatch):
    resp = _client(monkeypatch, bifrost_ok=True).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["bifrost"]["url"] == "http://bifrost.test/health"


def test_health_degraded_when_bifrost_down(monkeypatch):
    resp = _client(monkeypatch, bifrost_ok=False).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_openai_base_url_targets_bifrost():
    assert Settings(bifrost_url="http://localhost:8080/").bifrost_openai_base_url == "http://localhost:8080/v1"
