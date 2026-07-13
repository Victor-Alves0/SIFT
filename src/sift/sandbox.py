"""Pluggable sandbox backends for code mode.

``InProcessSandbox`` (default) runs the model's orchestration snippet in the
current process behind an AST policy + restricted builtins + a line budget —
fast, fine for TRUSTED catalogues.

``SubprocessSandbox`` runs it in a separate Python process with a wall-clock
watchdog (kills tight/native hangs the line budget can't) and, on Unix, CPU and
memory rlimits. Tool calls made by the snippet are proxied back to the parent
over stdio, so the untrusted code can't touch your tool objects or process
memory. It is stronger isolation — process + resource limits + no direct access
to the parent — but still not a VM: it does not by itself block network/filesystem
syscalls (add OS-level isolation — container/seccomp — for fully untrusted input).

Both backends share one contract:

    backend.run(code, call, search, schema) -> str   # JSON: {output|stdout|error}
"""
from __future__ import annotations

import ast
import builtins as _builtins
import contextlib
import io
import json
import os
import subprocess
import sys
import threading

# ---------------------------------------------------------------- policy

_SAFE_BUILTIN_NAMES = (
    "len range enumerate sorted reversed sum min max zip map filter list dict set "
    "tuple str int float bool round abs any all isinstance repr print"
).split()
_SAFE_BUILTINS = {n: getattr(_builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(_builtins, n)}

_BLOCKED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "memoryview", "breakpoint", "help",
    "exit", "quit", "object", "type", "super", "classmethod", "staticmethod",
})
# str.format / format_map traverse attributes at RUNTIME (e.g. "{0.__class__}".format(x)),
# which the AST dunder check wouldn't catch — block them explicitly.
_BLOCKED_ATTRS = frozenset({"format", "format_map", "mro"})

_LINE_BUDGET = 200_000

# The prompt text is DERIVED from the policy above so the two can never drift:
# a rule that is enforced but not communicated makes the model discover it by
# failing — and in tool calling every failure costs a whole context round-trip.
SANDBOX_RULES = (
    "Sandbox: no imports, no class definitions, no file/network access — the ONLY "
    "way out is call(). Anything you would import (datetime, requests, os, json...) "
    "must come from a tool instead. Available builtins: "
    + ", ".join(sorted(_SAFE_BUILTIN_NAMES)) + "."
)


class SandboxError(Exception):
    """Raised when code-mode source violates the sandbox policy."""


class _Guard(ast.NodeVisitor):
    def visit_Import(self, node):  # noqa: N802
        raise SandboxError("imports are not allowed in code mode")

    visit_ImportFrom = visit_Import

    def visit_ClassDef(self, node):  # noqa: N802
        # would fail at runtime anyway (__build_class__ is not exposed), but a
        # policy error beats a cryptic NameError
        raise SandboxError("class definitions are not allowed in code mode")

    def visit_Attribute(self, node):  # noqa: N802
        if node.attr.startswith("_") or node.attr in _BLOCKED_ATTRS:
            raise SandboxError(f"access to attribute {node.attr!r} is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node):  # noqa: N802
        name = node.id
        if name in _BLOCKED_NAMES or (name.startswith("__") and name.endswith("__")):
            raise SandboxError(f"use of name {name!r} is not allowed")
        self.generic_visit(node)


def _validate(code: str) -> ast.AST:
    try:
        tree = ast.parse(code, "<sift-code>", "exec")
    except SyntaxError as exc:
        raise SandboxError(f"syntax error: {exc}") from None
    _Guard().visit(tree)
    return tree


def _binds_output(tree: ast.AST) -> bool:
    return any(isinstance(n, ast.Name) and n.id == "output" and isinstance(n.ctx, ast.Store)
               for n in ast.walk(tree))


def _promote_trailing_expr(tree: ast.Module) -> bool:
    """REPL semantics: a bare expression on the last line becomes ``output``.

    Models routinely end a snippet with the value they mean to return (``[m["id"]
    for m in msgs]``) instead of assigning it. That is not ambiguity — it is what
    every REPL/notebook does — so honour it rather than burning a round-trip.
    Left alone when the snippet already binds ``output`` (explicit beats implicit)
    or when the last line is ``print(...)`` (stdout already carries it).

    Returns True if the snippet produces ``output`` at all.
    """
    if _binds_output(tree):
        return True
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return False
    last = tree.body[-1]
    value = last.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "print":
        return False
    assign = ast.Assign(targets=[ast.Name(id="output", ctx=ast.Store())], value=value)
    ast.copy_location(assign, last)      # keep line numbers: the budget tracer counts them
    ast.fix_missing_locations(assign)
    tree.body[-1] = assign
    return True


def _line_limiter(max_lines: int):
    count = [0]

    def tracer(frame, event, arg):
        # Budget the SNIPPET only: frames from real tool implementations (any
        # other filename) are neither counted nor locally traced — a heavy but
        # legitimate tool must not exhaust the snippet's budget (or pay tracing
        # overhead on every line it runs).
        if frame.f_code.co_filename != "<sift-code>":
            return None
        if event == "line":
            count[0] += 1
            if count[0] > max_lines:
                raise SandboxError("code exceeded the execution budget (possible infinite loop)")
        return tracer

    return tracer


def execute(code: str, call, search, schema, *, max_lines: int = _LINE_BUDGET) -> str:
    """Validate and run ``code`` with the tool helpers bound; returns a JSON
    string {output|stdout|error}. Used by both backends (and the subprocess child)."""
    try:
        tree = _validate(code)
    except SandboxError as exc:
        # a policy violation must teach the policy, or the model just guesses again
        return json.dumps({"error": f"SandboxError: {exc}", "hint": SANDBOX_RULES},
                          ensure_ascii=False)

    produces_output = _promote_trailing_expr(tree)

    calls = {"n": 0}

    def counting_call(_path: str, /, **params):
        calls["n"] += 1
        return call(_path, **params)

    ns = {"__builtins__": _SAFE_BUILTINS, "call": counting_call, "search": search,
          "schema": schema, "output": None}
    buf = io.StringIO()
    code_obj = compile(tree, "<sift-code>", "exec")
    had_trace = sys.gettrace()
    try:
        sys.settrace(_line_limiter(max_lines))
        with contextlib.redirect_stdout(buf):
            exec(code_obj, ns)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "stdout": buf.getvalue()},
                          default=str, ensure_ascii=False)
    finally:
        sys.settrace(had_trace)

    if ns.get("output") is not None:
        return json.dumps({"output": ns["output"]}, default=str, ensure_ascii=False)
    stdout = buf.getvalue()
    if stdout:
        return json.dumps({"stdout": stdout}, default=str, ensure_ascii=False)
    if produces_output:
        # the snippet DID set output; it is simply empty. Not an error — reporting
        # one here would invite a retry of calls that already ran (a re-sent email).
        return json.dumps({"output": None})

    # Nothing assigned, nothing printed: the snippet threw its own work away and
    # would otherwise return a hollow success ({"stdout": ""}) the model cannot
    # learn from. Say so, and say what already happened — a retry must not repeat
    # side effects.
    err = {"error": "no result: nothing was assigned to `output` and nothing was printed",
           "hint": "assign what you want back, e.g. output = call('some.tool.path', arg=1) "
                   "(a bare expression on the last line also becomes `output`)"}
    if calls["n"]:
        err["ran"] = (f"{calls['n']} tool call(s) already executed — they were NOT undone; "
                      "do not repeat any that have side effects")
    return json.dumps(err, ensure_ascii=False)


# ---------------------------------------------------------------- backends

class InProcessSandbox:
    """Runs in the current process (AST policy + line budget). Trusted catalogues."""

    def __init__(self, max_lines: int = _LINE_BUDGET) -> None:
        self.max_lines = max_lines

    def run(self, code: str, call, search, schema) -> str:
        return execute(code, call, search, schema, max_lines=self.max_lines)


class SubprocessSandbox:
    """Runs in an isolated child process with a wall-clock watchdog (+ Unix rlimits).
    Tool calls are proxied back to this (parent) process over stdio."""

    def __init__(self, *, timeout: float = 10.0, max_lines: int = _LINE_BUDGET,
                 cpu_seconds: int = 10, memory_mb: int = 512) -> None:
        self.timeout = timeout
        self.max_lines = max_lines
        self.cpu_seconds = cpu_seconds
        self.memory_mb = memory_mb

    def _preexec(self):  # pragma: no cover - Unix only
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_seconds, self.cpu_seconds))
            mem = self.memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except Exception:
            pass

    # Env vars the child actually needs to boot Python + import sift. Everything
    # else — API keys above all — must NOT leak into the process that runs
    # untrusted code.
    _ENV_KEEP = ("PATH", "PYTHONPATH", "PYTHONHOME", "SYSTEMROOT", "SYSTEMDRIVE",
                 "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL", "HOME", "USERPROFILE")

    def _child_env(self) -> dict:
        env = {k: os.environ[k] for k in self._ENV_KEEP if k in os.environ}
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def run(self, code: str, call, search, schema) -> str:
        handlers = {
            "call": lambda m: call(m["path"], **(m.get("params") or {})),
            "search": lambda m: search(m["q"], m.get("top_k", 5)),
            "schema": lambda m: schema(m["path"]),
        }
        kwargs = {}
        if os.name != "nt":
            kwargs["preexec_fn"] = self._preexec

        # launch the child as a FILE, not -m sift._sandbox_child: -m would import
        # the whole sift package (gateway -> embeddings -> numpy) into the child,
        # doubling its boot time; the script loads sandbox.py standalone instead
        child = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sandbox_child.py")
        proc = subprocess.Popen(
            [sys.executable, child],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self._child_env(), text=True, bufsize=1, **kwargs,
        )

        # drain stderr on a thread (a full pipe would deadlock the child); keep a
        # tail so an unexpected crash is diagnosable instead of a blind exit
        stderr_tail: list[str] = []

        def _drain():
            with contextlib.suppress(Exception):
                for line in proc.stderr:
                    stderr_tail.append(line)
                    del stderr_tail[:-20]

        drainer = threading.Thread(target=_drain, daemon=True)
        drainer.start()

        killed = {"v": False}

        def _watchdog():
            killed["v"] = True
            proc.kill()

        timer = threading.Timer(self.timeout, _watchdog)
        timer.start()
        try:
            proc.stdin.write(json.dumps({"code": code, "max_lines": self.max_lines}) + "\n")
            proc.stdin.flush()

            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                msg = json.loads(line)
                op = msg.get("op")
                if op == "done":
                    return msg["result"]
                # PAUSE the watchdog while the PARENT runs the (trusted) tool: the
                # timeout budgets the sandboxed snippet, not your own tools — a
                # long-running tool (deep search etc.) must not be killed mid-way
                # with its result thrown out. The child is blocked on readline
                # here, so no untrusted code runs while the clock is stopped.
                timer.cancel()
                try:
                    value = handlers[op](msg)
                    proc.stdin.write(json.dumps({"ok": True, "value": value}) + "\n")
                except Exception as exc:  # tool error -> surface to the sandboxed code
                    proc.stdin.write(json.dumps({"ok": False, "error": str(exc)}) + "\n")
                finally:
                    timer = threading.Timer(self.timeout, _watchdog)
                    timer.start()
                proc.stdin.flush()
        except OSError:
            pass   # child died mid-write (boot failure) -> the error path below
        finally:
            timer.cancel()
            with contextlib.suppress(Exception):
                proc.kill()

        if killed["v"]:
            return json.dumps({"error": "SandboxError: wall-clock timeout"})
        drainer.join(timeout=0.5)   # let the drain thread flush the last lines
        detail = ("".join(stderr_tail).strip()[-500:]) or "no stderr output"
        return json.dumps({"error": f"SandboxError: sandbox child exited unexpectedly ({detail})"},
                          ensure_ascii=False)
