"""Prompted (text) tool calling + constrained-decoding helpers, offline."""
import json

from sift.adapters.prompted import _extract_json, run_agent, single_decision


def _scripted(replies):
    it = iter(replies)
    return lambda prompt: next(it)


def test_extract_json_plain():
    assert _extract_json('{"tool": "x", "args": {}}') == {"tool": "x", "args": {}}


def test_extract_json_fenced_with_prose():
    text = 'Sure!\n```json\n{"answer": "hi"}\n```\nhope that helps'
    assert _extract_json(text) == {"answer": "hi"}


def test_extract_json_none():
    assert _extract_json("no json here") is None


def test_prompted_full_loop(sift):
    gen = _scripted([
        '{"tool": "search_tools", "args": {"q": "read email"}}',
        '{"tool": "execute_tool", "args": {"path": "google_workspace.gmail.read", "params": {"m": 1}}}',
        '{"answer": "Your last email is from a@b.c."}',
    ])
    answer = run_agent(sift, gen, "what's my last email?")
    assert answer == "Your last email is from a@b.c."


def test_prompted_recovers_from_malformed(sift):
    gen = _scripted([
        "I think I should search...",                       # no JSON -> reprompt
        '{"tool": "execute_tool", "args": {"path": "google_workspace.gmail.read", "params": {}}}',
        '{"answer": "done"}',
    ])
    assert run_agent(sift, gen, "read email") == "done"


def test_single_decision(sift):
    gen = _scripted(['{"path": "google_workspace.gmail.read", "args": {"m": 1}}'])
    out = single_decision(sift, gen, "read my last email")
    assert out["path"] == "google_workspace.gmail.read"
    assert out["result"]["subject"] == "Hi"
    assert "body" not in out["result"]  # response filtering still applies


def test_constrain_helpers(sift):
    schema = sift.tool_call_schema()
    names = schema["oneOf"][0]["properties"]["tool"]["enum"]
    assert set(names) == {"search_tools", "execute_tool"}
    gbnf = sift.json_gbnf()
    assert "root" in gbnf and "object" in gbnf


def test_tool_call_schema_is_json_serialisable(sift):
    json.dumps(sift.tool_call_schema())  # must not raise
