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


def test_search_browse_categories(sift):
    """search_tools with a path (no q) browses the hierarchy — folded get_tool_schema."""
    out = sift.dispatch("search_tools", {"path": ""})
    assert "google_workspace" in out and "local" in out


def test_search_browse_service(sift):
    out = sift.dispatch("search_tools", {"path": "google_workspace.gmail"})
    assert "read" in out and "send" in out


def test_get_tool_schema_alias_backcompat(sift):
    out = sift.dispatch("get_tool_schema", {"path": ""})
    assert "google_workspace" in out


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


def test_set_response_invalidates_toon_cache(sift):
    before = sift.get_tool_schema("google_workspace.gmail.read")   # populates the cache
    assert "r:id,subject,from,snippet,date" in before
    sift.set_response("google_workspace.gmail.read", returns=["id", "subject"])
    after = sift.get_tool_schema("google_workspace.gmail.read")
    assert "r:id,subject" in after and "snippet" not in after      # not the stale line


def test_examples_improve_discovery():
    """`examples=` phrasings are indexed on the embedding side."""
    from conftest import FakeEmbedder
    from sift import Sift

    # embedding-only isolates the example-text effect (examples enrich the
    # dense side; the BM25 side stays lean by design)
    s = Sift(embedder=FakeEmbedder(), retrieval="embedding")
    s.add_tool("dev.repo.bisect", lambda: {"ok": 1}, description="Run a binary search over commits",
               examples=["find which commit broke the build"])
    s.add_tool("dev.repo.log", lambda: {"ok": 1}, description="Show the commit history")
    s.build_index()
    res = s.search_tools("which commit broke the build", top_k=1)
    assert res[0].path == "dev.repo.bisect"       # matched via the example phrasing


def test_subprocess_child_env_is_scrubbed(monkeypatch):
    """The sandbox child must not inherit parent secrets (API keys etc.)."""
    from sift.sandbox import SubprocessSandbox

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    env = SubprocessSandbox()._child_env()
    assert "OPENROUTER_API_KEY" not in env and "AWS_SECRET_ACCESS_KEY" not in env
    assert "PATH" in env                           # but it can still boot Python


def test_structured_param_colon_default():
    """The structured dict form supports defaults containing ':' (the string DSL can't)."""
    from sift.registry import ToolDef
    from sift import toon

    t = ToolDef("a.b.c", "desc",
                {"q": {"type": "string", "default": "is:unread", "desc": "query"}})
    assert t.params["q"].default == "is:unread"
    # TOON quotes colon-bearing defaults so they stay unambiguous on one line
    assert "q:string:o:'is:unread'" in toon.encode_function(t)
