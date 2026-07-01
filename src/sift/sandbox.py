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


class SandboxError(Exception):
    """Raised when code-mode source violates the sandbox policy."""


class _Guard(ast.NodeVisitor):
    def visit_Import(self, node):  # noqa: N802
        raise SandboxError("imports are not allowed in code mode")

    visit_ImportFrom = visit_Import

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


def _line_limiter(max_lines: int):
    count = [0]

    def tracer(frame, event, arg):
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
        return json.dumps({"error": f"SandboxError: {exc}"}, ensure_ascii=False)

    ns = {"__builtins__": _SAFE_BUILTINS, "call": call, "search": search,
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
    return json.dumps({"stdout": buf.getvalue()}, default=str, ensure_ascii=False)


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

    def run(self, code: str, call, search, schema) -> str:
        handlers = {
            "call": lambda m: call(m["path"], **(m.get("params") or {})),
            "search": lambda m: search(m["q"], m.get("top_k", 5)),
            "schema": lambda m: schema(m["path"]),
        }
        kwargs = {}
        if os.name != "nt":
            kwargs["preexec_fn"] = self._preexec

        proc = subprocess.Popen(
            [sys.executable, "-m", "sift._sandbox_child"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, **kwargs,
        )

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
                try:
                    value = handlers[op](msg)
                    proc.stdin.write(json.dumps({"ok": True, "value": value}) + "\n")
                except Exception as exc:  # tool error -> surface to the sandboxed code
                    proc.stdin.write(json.dumps({"ok": False, "error": str(exc)}) + "\n")
                proc.stdin.flush()
        finally:
            timer.cancel()
            with contextlib.suppress(Exception):
                proc.kill()

        if killed["v"]:
            return json.dumps({"error": "SandboxError: wall-clock timeout"})
        return json.dumps({"error": "SandboxError: sandbox child exited unexpectedly"})
