"""Async surface: aexecute_tool / adispatch and async-def tools."""
import asyncio
import json

import pytest

from sift import Sift


def _sift() -> Sift:
    s = Sift(retrieval="bm25")

    @s.tool("mail.gmail.read", description="Read emails", params={"m": "number:o:1:max"},
            returns=["id", "n"])
    async def _r(m=1):
        await asyncio.sleep(0)               # genuinely async executor
        return {"id": "1", "n": m, "secret": "drop"}

    @s.tool("files.disk.stat", description="Stat a file", params={}, returns=["ok"])
    def _sync():
        return {"ok": True}

    return s.build_index()


def test_async_tool_via_aexecute():
    s = _sift()
    res = asyncio.run(s.aexecute_tool("mail.gmail.read", {"m": 2}))
    assert res == {"id": "1", "n": 2}        # awaited + projected


def test_sync_tool_via_adispatch():
    s = _sift()
    out = json.loads(asyncio.run(s.adispatch("execute_tool", {"path": "files.disk.stat"})))
    assert out == {"ok": True}


def test_adispatch_search_still_works():
    s = _sift()
    out = asyncio.run(s.adispatch("search_tools", {"q": "read emails"}))
    assert "mail.gmail.read" in out


def test_sync_execute_of_async_tool_raises_helpfully():
    s = _sift()
    with pytest.raises(TypeError, match="aexecute_tool"):
        s.execute_tool("mail.gmail.read", {})


def test_adispatch_surfaces_errors_as_json():
    s = _sift()
    out = json.loads(asyncio.run(s.adispatch("execute_tool", {"path": "no.such.tool"})))
    assert "error" in out
