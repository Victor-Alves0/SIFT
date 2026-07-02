"""Result-size cap (max_result_chars) and the observability hook."""
import json

import pytest

from sift import Sift


def _big_sift(**kw) -> Sift:
    s = Sift(retrieval="bm25", **kw)

    @s.tool("data.blob.get", description="Get a huge blob", params={}, returns=[])
    def _b():
        return {"data": "x" * 5000}

    return s.build_index()


def test_dispatch_caps_huge_results():
    s = _big_sift(max_result_chars=500)
    out = s.dispatch("execute_tool", {"path": "data.blob.get"})
    assert len(out) < 700                    # 500 + the truncation marker
    assert "truncated" in out and "set_response" in out


def test_cap_disabled_with_none():
    s = _big_sift(max_result_chars=None)
    out = s.dispatch("execute_tool", {"path": "data.blob.get"})
    assert len(out) > 5000 and "truncated" not in out


def test_run_code_output_is_capped_too():
    s = _big_sift(max_result_chars=500)
    out = s.dispatch("run_code", {"code": "output = call('data.blob.get')['data']"})
    assert len(out) < 700 and "truncated" in out


def test_observer_receives_search_and_execute():
    events = []
    s = _big_sift(observer=lambda ev, data: events.append((ev, data)))
    s.dispatch("search_tools", {"q": "get blob data"})
    s.dispatch("execute_tool", {"path": "data.blob.get"})
    kinds = [e[0] for e in events]
    assert kinds == ["search", "execute"]
    assert events[1][1]["ok"] is True and events[1][1]["path"] == "data.blob.get"
    assert "ms" in events[0][1]


def test_observer_sees_failures_and_never_breaks_the_loop():
    events = []

    def bad_then_record(ev, data):
        events.append((ev, data))
        raise RuntimeError("observer bug")   # must be swallowed

    s = _big_sift(observer=bad_then_record)
    out = json.loads(s.dispatch("execute_tool", {"path": "nope.nope.nope"}))
    assert "error" in out                    # the loop still returned normally
    assert events and events[0][1]["ok"] is False


def test_registry_add_rejects_duplicates():
    s = Sift(retrieval="bm25")
    s.add_tool("a.b.c", lambda: {}, description="first")
    with pytest.raises(ValueError, match="already registered"):
        s.add_tool("a.b.c", lambda: {}, description="shadow")
    s.add_tool("a.b.c", lambda: {"v": 2}, description="second", replace=True)  # explicit
    assert s.registry.tool("a.b.c").description == "second"
