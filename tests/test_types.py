"""Type coercion at the LLM→tool boundary.

LLMs routinely send every argument as a string ("3", "false", "[1,2]"), and
tools break in non-obvious ways when the string leaks through (float slice
indices, truthy "false", …). These pin the coercion contract of gateway._coerce
/ _prepare_args.
"""
import pytest

from sift import Sift


def _sift(**tool_kwargs) -> Sift:
    s = Sift(retrieval="bm25")
    s.add_tool("t.t.echo", lambda **kw: {"got": kw}, description="echo params back",
               **tool_kwargs)
    return s.build_index()


def _run(params_spec: dict, sent: dict) -> dict:
    return _sift(params=params_spec).execute_tool("t.t.echo", sent)["got"]


def test_number_integral_stays_int_for_slicing():
    s = Sift(retrieval="bm25")

    @s.tool("mail.inbox.read", description="Read emails newest first",
            params={"m": "number:o:10:max results"}, returns=["ids"])
    def _read(m=10):
        return {"ids": ["a", "b", "c"][:m]}   # slicing: float m would TypeError

    s.build_index()
    assert s.execute_tool("mail.inbox.read", {"m": 2})["ids"] == ["a", "b"]
    assert s.execute_tool("mail.inbox.read", {"m": "2"})["ids"] == ["a", "b"]


def test_number_keeps_real_floats():
    assert _run({"x": "number:n::value"}, {"x": "3.5"}) == {"x": 3.5}
    got = _run({"x": "number:n::value"}, {"x": 3.0})["x"]
    assert got == 3 and isinstance(got, int)   # integral float -> int


def test_integer_type():
    got = _run({"n": "integer:n::count"}, {"n": "7"})["n"]
    assert got == 7 and isinstance(got, int)
    assert _run({"n": "integer:n::count"}, {"n": 7.9}) == {"n": 7}


def test_boolean_strings():
    spec = {"flag": "boolean:n::a flag"}
    assert _run(spec, {"flag": "false"})["flag"] is False   # the dangerous one
    assert _run(spec, {"flag": "False"})["flag"] is False
    assert _run(spec, {"flag": "0"})["flag"] is False
    assert _run(spec, {"flag": "true"})["flag"] is True
    assert _run(spec, {"flag": True})["flag"] is True


def test_array_and_object_from_json_strings():
    assert _run({"a": "array:n::items"}, {"a": "[1, 2, 3]"}) == {"a": [1, 2, 3]}
    assert _run({"o": "object:n::payload"}, {"o": '{"k": 1}'}) == {"o": {"k": 1}}
    # native values pass through untouched
    assert _run({"a": "array:n::items"}, {"a": [1]}) == {"a": [1]}


def test_unparseable_value_passes_through():
    # the tool gets the raw value and can raise its own, more specific error
    assert _run({"n": "number:n::count"}, {"n": "many"}) == {"n": "many"}
    assert _run({"a": "array:n::items"}, {"a": "not json"}) == {"a": "not json"}


def test_explicit_empty_string_overrides_default():
    spec = {"q": {"type": "string", "default": "is:unread", "desc": "query"}}
    assert _run(spec, {"q": ""}) == {"q": ""}          # "" is a real value now
    assert _run(spec, {}) == {"q": "is:unread"}        # absent -> default
    assert _run(spec, {"q": None}) == {"q": "is:unread"}  # None -> default


def test_required_still_enforced_when_absent():
    with pytest.raises(ValueError):
        _run({"to": "string:n::recipient"}, {})
    with pytest.raises(ValueError):
        _run({"to": "string:n::recipient"}, {"to": None})
