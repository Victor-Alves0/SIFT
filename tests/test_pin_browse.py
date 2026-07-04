"""Two safe latency optimizations: browse-with-fallback and pinned hot tools."""
import json

import pytest

from sift import Sift


def _sift() -> Sift:
    s = Sift(retrieval="bm25")

    @s.tool("utils.time.now", description="Current date and time clock",
            params={"timezone": "string:o::IANA tz, empty=UTC"}, returns=["datetime", "tz"])
    def _now(timezone=""):
        return {"datetime": "2026-07-04 01:03 -03", "tz": timezone or "UTC", "unix": 1}

    @s.tool("crm.contacts.delete", description="Delete a contact from the CRM",
            params={"id": "string:n::id"}, returns=["ok"], risk=True)
    def _del(id):
        return {"ok": True}

    return s.build_index()


# ---------------------------------------------------------------- browse fallback

def test_browse_bad_guess_falls_back_to_search():
    s = _sift()
    # 'clock' is not a category, but it matches the time tool's text -> found in ONE call
    out = s.dispatch("search_tools", {"path": "clock"})
    assert "utils.time.now" in out
    assert "unknown category" not in out


def test_browse_valid_path_still_lists_level():
    s = _sift()
    assert "utils.time" in s.dispatch("search_tools", {"path": "utils"})   # real category
    cats = s.dispatch("search_tools", {"path": ""})
    assert "utils" in cats and "crm" in cats


def test_browse_bad_guess_with_no_match_is_graceful():
    s = _sift()
    out = s.dispatch("search_tools", {"path": "zzqq_nonexistent"})
    assert "no matching tools" in out and "unknown category" not in out


# ---------------------------------------------------------------------- pinning

def test_pin_exposes_first_class_spec():
    s = _sift().pin("utils.time.now")
    names = [t["function"]["name"] for t in s.openai_tools()]
    assert names == ["search_tools", "execute_tool", "utils__time__now"]
    spec = next(t["function"] for t in s.openai_tools()
                if t["function"]["name"] == "utils__time__now")
    assert spec["parameters"]["properties"]["timezone"]["type"] == "string"


def test_pinned_tool_executes_in_one_call_with_model_params():
    s = _sift().pin("utils.time.now")
    # the model calls the pinned tool DIRECTLY (no search) and still supplies params
    out = json.loads(s.dispatch("utils__time__now", {"timezone": "America/Sao_Paulo"}))
    assert out == {"datetime": "2026-07-04 01:03 -03", "tz": "America/Sao_Paulo"}


def test_pin_adds_system_prompt_hint():
    base = _sift()
    assert "no search needed" not in base.system_prompt
    pinned = base.pin("utils.time.now").system_prompt
    assert "no search needed" in pinned and "utils__time__now" in pinned


def test_pin_validates_existence():
    with pytest.raises(KeyError):
        _sift().pin("nope.nope.nope")


def test_pin_is_idempotent():
    s = _sift().pin("utils.time.now").pin("utils.time.now")
    assert s._pinned == ["utils.time.now"]


# ------------------------------------------------------------ pin under scope

def test_scope_hides_and_denies_pinned_tool():
    s = _sift().pin("utils.time.now", "crm.contacts.delete")
    view = s.scope(deny=["utils.*"])
    names = [t["function"]["name"] for t in view.openai_tools()]
    assert "utils__time__now" not in names            # hidden from the scoped surface
    assert "crm__contacts__delete" in names           # this pin is still allowed
    out = json.loads(view.dispatch("utils__time__now", {}))
    assert "not allowed" in out["error"]              # and execution is blocked


def test_scope_allows_permitted_pinned_tool():
    s = _sift().pin("utils.time.now")
    view = s.scope(allow=["utils.*"])
    out = json.loads(view.dispatch("utils__time__now", {"timezone": "UTC"}))
    assert out["tz"] == "UTC"


def test_pinned_async_via_adispatch():
    import asyncio

    s = Sift(retrieval="bm25")

    @s.tool("utils.clock.now", description="now", params={}, returns=["t"])
    async def _n():
        return {"t": 1}

    s.build_index().pin("utils.clock.now")
    out = json.loads(asyncio.run(s.adispatch("utils__clock__now", {})))
    assert out == {"t": 1}
