"""Code mode: orchestrate tools in one turn, with a constrained namespace."""
import json
import re


def test_code_tools_specs(sift):
    """execute_tool belongs in code mode: forcing Python for a SINGLE call pays
    sandbox overhead (and a real parse-failure rate) to do what one JSON call does."""
    names = {t["function"]["name"] for t in sift.code_tools()}
    assert names == {"search_tools", "execute_tool", "run_code"}


def test_code_mode_execute_tool_dispatches(sift):
    out = json.loads(sift.dispatch("execute_tool", {"path": "local.filesystem.read",
                                                    "params": {"path": "/x"}}))
    assert out["path"] == "/x"          # no code written, nothing to compile


def test_code_prompt_tells_the_model_to_keep_output_small(sift):
    # the pattern behind code mode's headline savings: filter in the sandbox, not
    # in the context — intermediate values are free, `output` is re-sent every turn
    prompt = sift.code_system_prompt
    assert "output` SMALL" in prompt or "output SMALL" in prompt
    assert "execute_tool" in prompt


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


def test_trailing_expression_becomes_output(sift):
    """REPL shim: the model ends with the value it means to return, unassigned."""
    out = json.loads(sift.run_code("call('google_workspace.gmail.read', m=1)['subject']"))
    assert out["output"] == "Hi"          # was {"stdout": ""} — a hollow success


def test_trailing_expression_does_not_override_explicit_output(sift):
    code = ("output = 'explicit'\n"
            "call('google_workspace.gmail.read', m=1)\n")
    assert json.loads(sift.run_code(code))["output"] == "explicit"


def test_trailing_print_still_goes_to_stdout(sift):
    out = json.loads(sift.run_code("print('hello')"))
    assert out["stdout"].strip() == "hello" and "output" not in out


def test_no_result_returns_an_actionable_error(sift):
    out = json.loads(sift.run_code("x = 1 + 1"))
    assert "no result" in out["error"]
    assert "output" in out["hint"]
    assert "ran" not in out                # nothing executed -> nothing to warn about


def test_no_result_warns_about_calls_already_made(sift):
    """The retry hazard: the snippet ran a tool and discarded it. Saying only
    'assign to output' would invite re-running a side effect."""
    out = json.loads(sift.run_code("msg = call('google_workspace.gmail.read', m=1)"))
    assert "no result" in out["error"]
    assert "1 tool call(s) already executed" in out["ran"]


def test_explicit_empty_output_is_not_an_error(sift):
    """A tool that legitimately returns nothing must NOT look like a failure —
    an error here would push the model to repeat the call."""
    out = json.loads(sift.run_code("output = None"))
    assert out == {"output": None} and "error" not in out


def test_policy_error_teaches_the_policy(sift):
    out = json.loads(sift.run_code("import datetime\noutput = datetime.date.today()"))
    assert "import" in out["error"].lower()
    assert "call()" in out["hint"]         # tells it where datetime must come from


def test_sandbox_rules_are_derived_from_the_policy():
    from sift.codemode import CODE_SYSTEM_PROMPT
    from sift.sandbox import _SAFE_BUILTIN_NAMES, SANDBOX_RULES

    for name in ("len", "sorted", "print"):
        assert name in SANDBOX_RULES       # generated from _SAFE_BUILTIN_NAMES...
    assert "eval" not in SANDBOX_RULES     # ...so it cannot drift from what is allowed
    listed = set(re.findall(r"[a-z_]+", SANDBOX_RULES))
    assert set(_SAFE_BUILTIN_NAMES) <= listed
    assert SANDBOX_RULES in CODE_SYSTEM_PROMPT


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
