"""Core behaviour: parsing, navigation, TOON, filtering, dispatch, discovery."""
import json

from sift.registry import parse_param


def test_parse_param():
    opt = parse_param("m", "number:o:10:max results")
    assert (opt.type, opt.required, opt.default, opt.desc) == ("number", False, "10", "max results")
    req = parse_param("q", "string:n::search query")
    assert (req.type, req.required, req.default) == ("string", True, "")


def test_toon_encode(sift):
    line = sift.get_tool_schema("google_workspace.gmail.read")
    assert "\n" not in line
    assert "google_workspace.gmail.read" in line
    assert "m:number:o:10" in line
    assert "r:id,subject,from,snippet,date" in line


def test_toon_risk_marker(sift):
    line = sift.get_tool_schema("google_workspace.gmail.send")
    assert line.endswith("|risk")
    assert "to:string:n" in line


def test_response_filtering(sift):
    res = sift.execute_tool("google_workspace.gmail.read", {"m": 1})
    assert "body" not in res  # not in the whitelist
    assert set(res) == {"id", "subject", "from", "snippet", "date"}


def test_required_param_missing(sift):
    try:
        sift.execute_tool("google_workspace.gmail.send", {"subject": "hi"})
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing required 'to'")


def test_navigation(sift):
    cats = sift.registry.categories()
    assert "google_workspace" in cats and "local" in cats
    assert "gmail" in sift.registry.services("google_workspace")
    assert "read" in sift.registry.functions("google_workspace.gmail")
    assert sift.registry.risky_paths() == {"google_workspace.gmail.send"}


def test_dispatch_search_returns_toon(sift):
    """search now returns matches WITH schema inline (TOON), not a JSON list."""
    out = sift.dispatch("search_tools", {"q": "read emails from gmail inbox"})
    assert "google_workspace.gmail.read" in out
    assert "r:" in out                      # schema is inline
    assert not out.lstrip().startswith("[")  # not a JSON list anymore


def test_dispatch_execute_filters(sift):
    out = sift.dispatch("execute_tool", {"path": "local.filesystem.read", "params": {"path": "/tmp/x"}})
    data = json.loads(out)
    assert data == {"path": "/tmp/x", "content": "hello"}


def test_dispatch_unknown_meta_tool(sift):
    out = json.loads(sift.dispatch("nope", {}))
    assert "error" in out


def test_search_ranks_gold(sift):
    res = sift.search_tools("read my last email", top_k=3)
    assert any(r.path == "google_workspace.gmail.read" for r in res)


def test_structured_param_colon_default():
    """The structured dict form supports defaults containing ':' (the string DSL can't)."""
    from sift.registry import ToolDef
    from sift import toon

    t = ToolDef("a.b.c", "desc",
                {"q": {"type": "string", "default": "is:unread", "desc": "query"}})
    assert t.params["q"].default == "is:unread"
    # TOON quotes colon-bearing defaults so they stay unambiguous on one line
    assert "q:string:o:'is:unread'" in toon.encode_function(t)
