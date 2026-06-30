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
