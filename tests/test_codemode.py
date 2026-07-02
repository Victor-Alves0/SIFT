"""Code mode: orchestrate tools in one turn, with a constrained namespace."""
import json


def test_code_tools_specs(sift):
    names = {t["function"]["name"] for t in sift.code_tools()}
    assert names == {"search_tools", "run_code"}


def test_run_code_single_call(sift):
    out = json.loads(sift.run_code("output = call('google_workspace.gmail.read', m=1)"))
    data = out["output"]
    assert data["subject"] == "Hi"        # executed
    assert "body" not in data             # response filtering still applies


def test_run_code_compose_many_calls(sift):
    code = (
        "paths = search('read email')\n"
        "msg = call('google_workspace.gmail.read', m=1)\n"
        "output = {'found': len(paths) > 0, 'subject': msg['subject']}\n"
    )
    out = json.loads(sift.run_code(code))["output"]
    assert out["found"] is True
    assert out["subject"] == "Hi"


def test_run_code_blocks_imports(sift):
    out = json.loads(sift.run_code("import os\noutput = os.getcwd()"))
    assert "error" in out and "import" in out["error"].lower()


def test_run_code_blocks_dunder_escape(sift):
    # the classic restricted-exec escape: reach object internals via dunder attrs
    out = json.loads(sift.run_code("output = ().__class__.__bases__"))
    assert "error" in out and "not allowed" in out["error"]


def test_run_code_blocks_dangerous_names(sift):
    for src in ("output = getattr(1, 'real')", "output = eval('1+1')", "output = open('x')"):
        out = json.loads(sift.run_code(src))
        assert "error" in out and "not allowed" in out["error"]


def test_run_code_line_budget_stops_infinite_loop(sift):
    out = json.loads(sift.run_code("while True:\n    x = 1"))
    assert "error" in out and "budget" in out["error"].lower()


def test_run_code_blocks_str_format_escape(sift):
    # str.format traverses attributes at runtime -> the classic escape
    out = json.loads(sift.run_code('output = "{0.__class__}".format(())'))
    assert "error" in out and "not allowed" in out["error"]


def test_dispatch_run_code(sift):
    out = json.loads(sift.dispatch("run_code", {"code": "output = call('local.filesystem.read', path='/x')"}))
    assert out["output"]["path"] == "/x"


def test_run_code_blocks_class_definitions(sift):
    # a clear policy error, not a cryptic NameError: __build_class__
    out = json.loads(sift.run_code("class A: pass\noutput = 1"))
    assert "error" in out and "class" in out["error"].lower()


def test_line_budget_ignores_tool_internals():
    """A heavy-but-legitimate tool must not exhaust the SNIPPET's line budget:
    only <sift-code> frames are counted (and tool frames aren't traced at all)."""
    import json as _json

    from sift import Sift
    from sift.sandbox import InProcessSandbox

    s = Sift(retrieval="bm25", sandbox=InProcessSandbox(max_lines=50))

    @s.tool("heavy.compute.run", description="Heavy computation", params={}, returns=["n"])
    def _heavy():
        n = 0
        for _ in range(5000):   # ~10k traced lines if tool frames counted
            n += 1
        return {"n": n}

    s.build_index()
    out = _json.loads(s.run_code("output = call('heavy.compute.run')['n']"))
    assert out.get("output") == 5000, out   # would blow the 50-line budget before the fix
