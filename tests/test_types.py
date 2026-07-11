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


def test_unparseable_value_raises_clean_error():
    """Plausible garbage must NOT reach the tool ('x' * 4 == 'xxxx' propagates
    hallucination) — the model gets a structured, named error to retry from."""
    with pytest.raises(ValueError, match="'n'.*integer"):
        _run({"n": "integer:n::count"}, {"n": "x"})
    with pytest.raises(ValueError, match="'n'.*number"):
        _run({"n": "number:n::count"}, {"n": "many"})
    with pytest.raises(ValueError, match="'a'.*array"):
        _run({"a": "array:n::items"}, {"a": "not json"})
    with pytest.raises(ValueError, match="'f'.*boolean"):
        _run({"f": "boolean:n::flag"}, {"f": "maybe"})


def test_required_flag_r_is_accepted_and_typos_raise():
    """'r' reads as required (users assume it); an unknown flag raises instead of
    silently meaning optional."""
    from sift.registry import parse_param

    assert parse_param("a", "integer:r::val").required is True
    assert parse_param("a", "integer:n::val").required is True
    assert parse_param("a", "integer:o::val").required is False
    with pytest.raises(ValueError, match="req flag"):
        parse_param("a", "integer:x::val")


def test_bare_decorator_derives_params_from_signature():
    """A tool registered without params= must be CALLABLE, not a silent trap."""
    import json

    s = Sift(retrieval="bm25")

    @s.tool("demo.math.add", description="add two numbers")
    def add(a: int, b: int, precise: bool = False):
        return {"sum": a + b, "precise": precise}

    s.build_index()
    out = json.loads(s.dispatch("execute_tool",
                                {"path": "demo.math.add", "params": {"a": "1", "b": 2}}))
    assert out == {"sum": 3, "precise": False}      # bound AND coerced ("1" -> 1)

    schema = s.get_tool_schema("demo.math.add")
    assert "a:integer:n" in schema and "precise:boolean:o" in schema


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
