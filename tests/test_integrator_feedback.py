"""Regressions from real-integrator feedback on 0.5.0 (errors that mislead
models, per-scope pins, async scope, watchdog vs long tools, on_risky, meta).
"""
import asyncio
import json
import time

import pytest

from sift import Sift


def _sift(**kw) -> Sift:
    s = Sift(retrieval="bm25", **kw)

    @s.tool("demo.math.add", description="add two numbers",
            params={"a": "integer:n::a", "b": "integer:n::b"})
    def add(a, b):
        return {"sum": a + b}

    @s.tool("mail.gmail.send", description="Send an email",
            params={"to": "string:n::recipient"}, returns=["ok"], risk=True)
    def send(to):
        return {"ok": True}

    return s.build_index()


# ------------------------------------------------- #3 errors that don't mislead

def test_missing_path_is_named_not_misclassified():
    out = json.loads(_sift().dispatch("execute_tool", {"tool": "demo.math.add"}))
    assert "requires a 'path'" in out["error"]      # what actually went wrong
    assert "None" not in out["error"]               # no repr noise
    assert "search_tools" in out["hint"]            # recovery pointer


def test_scope_missing_path_is_not_a_permission_error():
    view = _sift().scope(allow=["demo.*"])
    out = json.loads(view.dispatch("execute_tool", {"tool": "demo.math.add"}))
    assert "not allowed" not in out["error"]        # was: "tool None is not allowed..."
    assert "requires a 'path'" in out["error"] and "search_tools" in out["hint"]


def test_unknown_tool_error_is_clean_and_has_hint():
    out = json.loads(_sift().dispatch("execute_tool", {"path": "no.such.tool"}))
    assert out["error"] == "unknown tool 'no.such.tool'"   # no double-quoted KeyError
    assert "search_tools" in out["hint"]


# --------------------------------------------------------- #4 per-scope pinning

def test_scope_pin_is_local_to_the_scope():
    s = _sift()
    v1 = s.scope(allow=["demo.*"], pin=["demo.math.add"])
    v2 = s.scope(allow=["mail.*"])
    names1 = [t["function"]["name"] for t in v1.openai_tools()]
    names2 = [t["function"]["name"] for t in v2.openai_tools()]
    assert "demo__math__add" in names1
    assert "demo__math__add" not in names2          # no shared mutable state
    assert s._pinned == []                          # parent untouched
    assert "demo__math__add" in v1.system_prompt


def test_scope_pin_denied_by_own_rules_raises():
    with pytest.raises(ValueError, match="denied"):
        _sift().scope(allow=["mail.*"], pin=["demo.math.add"])


def test_scope_pin_executes_via_flat_name():
    v = _sift().scope(allow=["demo.*"], pin=["demo.math.add"])
    out = json.loads(v.dispatch("demo__math__add", {"a": 2, "b": 3}))
    assert out == {"sum": 5}


# ------------------------------------------------------------- #5 async on scope

def test_scope_has_async_surface():
    s = Sift(retrieval="bm25")

    @s.tool("mail.inbox.peek", description="peek inbox", params={}, returns=["n"])
    async def peek():
        await asyncio.sleep(0)
        return {"n": 1}

    s.build_index()
    view = s.scope(allow=["mail.*"])
    assert json.loads(asyncio.run(view.adispatch(
        "execute_tool", {"path": "mail.inbox.peek"})))["n"] == 1
    with pytest.raises(PermissionError):
        asyncio.run(view.aexecute_tool("other.x.y"))


def test_scope_adispatch_enforces_scope():
    view = _sift().scope(allow=["demo.*"])
    out = json.loads(asyncio.run(view.adispatch(
        "execute_tool", {"path": "mail.gmail.send", "params": {"to": "a@b"}})))
    assert "not allowed" in out["error"]


# ---------------------------------------- #6 watchdog ignores trusted tool time

def test_watchdog_survives_slow_parent_tool():
    """The subprocess timeout budgets the SNIPPET, not your own tools: a tool
    slower than the timeout must complete and its result must not be discarded."""
    from sift.sandbox import SubprocessSandbox

    s = Sift(retrieval="bm25", sandbox=SubprocessSandbox(timeout=1.5))

    @s.tool("research.deep.search", description="Slow deep search", params={}, returns=["hits"])
    def slow():
        time.sleep(2.5)          # longer than the sandbox timeout, runs in the PARENT
        return {"hits": 42}

    s.build_index()
    out = json.loads(s.run_code("output = call('research.deep.search')['hits']"))
    assert out.get("output") == 42, out


# --------------------------------------------------------------- #9 on_risky

def test_on_risky_blocks_until_confirmed():
    asked = []

    def guard(path, args):
        asked.append((path, args))
        return False              # human said no

    s = _sift(on_risky=guard)
    out = json.loads(s.dispatch("execute_tool",
                                {"path": "mail.gmail.send", "params": {"to": "a@b.c"}}))
    assert "not confirmed" in out["error"]
    assert asked == [("mail.gmail.send", {"to": "a@b.c"})]   # got path + prepared args


def test_on_risky_allows_and_skips_non_risky():
    calls = []
    s = _sift(on_risky=lambda p, a: calls.append(p) or True)
    assert json.loads(s.dispatch("execute_tool",
                                 {"path": "mail.gmail.send", "params": {"to": "x"}}))["ok"]
    s.dispatch("execute_tool", {"path": "demo.math.add", "params": {"a": 1, "b": 1}})
    assert calls == ["mail.gmail.send"]        # non-risky never hits the guard


# ------------------------------------------------------------------- #8 meta

def test_meta_dicts_exist_for_integrators():
    s = _sift()
    v = s.scope(allow=["demo.*"])
    s.meta["mode"] = "workspace"
    v.meta["aw_tools"] = [1, 2]
    assert s.meta["mode"] == "workspace" and v.meta["aw_tools"] == [1, 2]


# ------------------------------------------------------------ #7 language hint

def test_system_prompt_has_language_guidance():
    assert "language of the tool descriptions" in _sift().system_prompt
