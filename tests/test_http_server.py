"""OpenAPI HTTP tool server."""
import pytest

pytest.importorskip("fastapi", reason="requires the 'server' extra")
from fastapi.testclient import TestClient  # noqa: E402

from sift import Sift  # noqa: E402
from sift.http_server import build_app  # noqa: E402


def _client(**kwargs) -> TestClient:
    s = Sift(retrieval="bm25")

    @s.tool("mail.gmail.read", description="Read emails from the inbox", params={}, returns=["id"])
    def _r():
        return {"id": "1", "extra": "x"}

    s.build_index()
    return TestClient(build_app(s, **kwargs))


def test_health():
    assert _client().get("/health").json()["status"] == "ok"


def test_search_returns_inline_schema():
    r = _client().post("/search_tools", json={"q": "read emails inbox"})
    assert r.status_code == 200 and "mail.gmail.read" in r.json()["result"]


def test_execute_filters_response():
    body = _client().post("/execute_tool", json={"path": "mail.gmail.read", "params": {}}).json()["result"]
    assert '"id"' in body and "extra" not in body  # returns whitelist applied


def test_openapi_exposes_meta_tools():
    spec = _client().get("/openapi.json").json()
    assert "/search_tools" in spec["paths"] and "/execute_tool" in spec["paths"]
    assert "/get_tool_schema" not in spec["paths"]  # folded into search_tools


def test_search_browse_via_path():
    body = _client().post("/search_tools", json={"path": ""}).json()["result"]
    assert "mail" in body  # browsing the root lists categories


def test_scope_limits_server():
    s = Sift(retrieval="bm25")

    @s.tool("mail.gmail.read", description="Read emails", params={}, returns=["id"])
    def _r():
        return {"id": "1"}

    @s.tool("web.search.run", description="Search the web", params={"q": "string:n::q"}, returns=["url"])
    def _w(q):
        return {"url": "u"}

    s.build_index()
    client = TestClient(build_app(s, scope=s.scope(allow=["mail.*"])))
    out = client.post("/execute_tool", json={"path": "web.search.run", "params": {"q": "x"}}).json()["result"]
    assert "not allowed" in out


def test_auth_required(monkeypatch):
    monkeypatch.setenv("SIFT_API_KEY", "secret")
    c = _client()
    assert c.post("/search_tools", json={"q": "x"}).status_code == 401
    ok = c.post("/search_tools", json={"q": "read emails"}, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
