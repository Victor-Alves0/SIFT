"""Per-model tool scoping (allowedTools) — allow/deny enforcement."""
import json

import pytest

from sift import Sift


def _sift() -> Sift:
    s = Sift(retrieval="bm25")  # offline, deterministic

    @s.tool("google_workspace.gmail.read", description="Read emails from the inbox",
            params={}, returns=["id"])
    def _r():
        return {"id": "1"}

    @s.tool("google_workspace.gmail.send", description="Send an email",
            params={"to": "string:n::to"}, returns=["id"], risk=True)
    def _s(to):
        return {"id": "2"}

    @s.tool("web.search.run", description="Search the web", params={"q": "string:n::q"},
            returns=["url"])
    def _w(q):
        return {"url": "u"}

    return s.build_index()


def test_scope_search_and_execute_allowed():
    v = _sift().scope(allow=["google_workspace.gmail.*"])
    out = v.search_compact("read email inbox")
    assert "google_workspace.gmail.read" in out and "web.search.run" not in out
    assert v.execute_tool("google_workspace.gmail.read")["id"] == "1"


def test_scope_execute_denied_raises():
    v = _sift().scope(allow=["google_workspace.gmail.read"])
    with pytest.raises(PermissionError):
        v.execute_tool("web.search.run", {"q": "x"})


def test_scope_dispatch_execute_denied():
    v = _sift().scope(allow=["web.*"])
    out = json.loads(v.dispatch("execute_tool", {"path": "google_workspace.gmail.read", "params": {}}))
    assert "not allowed" in out["error"]


def test_scope_search_via_dispatch_only_allowed():
    v = _sift().scope(allow=["web.*"])
    out = v.dispatch("search_tools", {"q": "search the web internet"})
    assert "web.search.run" in out and "gmail" not in out


def test_scope_deny_wins():
    v = _sift().scope(allow=["google_workspace.gmail.*"], deny=["*.send"])
    assert v.allowed("google_workspace.gmail.read")
    assert not v.allowed("google_workspace.gmail.send")


def test_scope_codemode_enforces_allow():
    v = _sift().scope(allow=["web.*"])
    ok = json.loads(v.run_code("output = call('web.search.run', q='x')"))
    assert ok["output"]["url"] == "u"
    denied = json.loads(v.run_code("output = call('google_workspace.gmail.read')"))
    assert "error" in denied and "not allowed" in denied["error"]


def test_scope_dispatch_run_code_enforced():
    v = _sift().scope(allow=["web.*"])
    out = json.loads(v.dispatch("run_code", {"code": "output = call('google_workspace.gmail.read')"}))
    assert "not allowed" in out["error"]


def test_scope_allow_risky_false_blocks_risky():
    v = _sift().scope(allow=["google_workspace.gmail.*"], allow_risky=False)
    assert v.allowed("google_workspace.gmail.read")
    assert not v.allowed("google_workspace.gmail.send")  # risk=True
    with pytest.raises(PermissionError):
        v.execute_tool("google_workspace.gmail.send", {"to": "x"})


# ---- browse must not disclose denied tools (information leak) ----

def test_scope_browse_hides_denied_function():
    v = _sift().scope(deny=["*.send"])
    listing = v.dispatch("search_tools", {"path": "google_workspace.gmail"})
    assert "google_workspace.gmail.read" in listing
    assert "send" not in listing            # schema of the denied tool is not shown


def test_scope_browse_hides_empty_category():
    v = _sift().scope(allow=["web.*"])
    cats = v.dispatch("search_tools", {"path": ""})
    assert "web" in cats
    assert "google_workspace" not in cats   # no visible tool -> category hidden


def test_scope_browse_denied_leaf_is_error():
    v = _sift().scope(deny=["*.send"])
    out = json.loads(v.dispatch("search_tools", {"path": "google_workspace.gmail.send"}))
    assert "error" in out and "not visible" in out["error"]


def test_scope_deprecated_alias_is_scoped_too():
    v = _sift().scope(deny=["*.send"])
    listing = v.dispatch("get_tool_schema", {"path": "google_workspace.gmail"})
    assert "send" not in listing
