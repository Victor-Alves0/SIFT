"""Code mode — let the model orchestrate tools by writing code.

Instead of one round-trip per tool, the model emits a short Python snippet that
composes many tool calls in a SINGLE turn (the StackOne/Cloudflare "code mode"
pattern). This collapses the multi-turn overhead for composite tasks.

The snippet runs in a constrained namespace exposing ``call``, ``search`` and
``schema``, with a restricted ``__builtins__`` (no imports / file / eval). It is
NOT a hardened sandbox — the registered tool executors are your own code — but it
blocks the obvious escape hatches. Keep code mode for trusted catalogues.
"""
from __future__ import annotations

import ast
import builtins as _builtins
import contextlib
import io
import json
import sys

CODE_SYSTEM_PROMPT = """You orchestrate tools by WRITING CODE, composing many calls in a single turn.

Tools:
1. search_tools(q) — find tool paths (schema inline) by need.
2. run_code(code)  — runs Python NOW. Inside it you have:
     call(path, **params) -> dict   # execute a tool, returns its filtered result
     search(q) -> [paths]           # discovery; returns matching tool paths
     schema(path) -> str            # TOON schema of a tool/level
   Assign the data you want back to a variable named `output`.

Flow: optionally call search_tools to learn paths, then ONE run_code that performs
all the tool calls and sets `output`. Then answer the user concisely.
"risk" tools (send/delete): only run them if the user authorised it."""

_SAFE_BUILTIN_NAMES = (
    "len range enumerate sorted reversed sum min max zip map filter list dict set "
    "tuple str int float bool round abs any all isinstance repr print"
).split()
_SAFE_BUILTINS = {n: getattr(_builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(_builtins, n)}

# Names that must never be referenced, even though they aren't in the namespace —
# blocking them at the AST level stops the usual sandbox-escape tricks.
_BLOCKED_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "memoryview", "breakpoint", "help",
    "exit", "quit", "object", "type", "super", "classmethod", "staticmethod",
})
# Bound the number of executed lines so a runaway loop (while True) can't hang.
_LINE_BUDGET = 200_000


class SandboxError(Exception):
    """Raised when code-mode source violates the sandbox policy."""


# attribute names blocked even though they don't start with "_": str.format /
# format_map traverse attributes at RUNTIME (e.g. "{0.__class__}".format(x)), so
# the AST dunder check alone wouldn't catch them.
_BLOCKED_ATTRS = frozenset({"format", "format_map", "mro"})


class _Guard(ast.NodeVisitor):
    """Reject imports, dunder/private attribute access and dangerous names —
    the vectors used to break out of a restricted ``exec``."""

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


def _line_limiter():
    count = [0]

    def tracer(frame, event, arg):
        if event == "line":
            count[0] += 1
            if count[0] > _LINE_BUDGET:
                raise SandboxError("code exceeded the execution budget (possible infinite loop)")
        return tracer

    return tracer


def code_tool_specs() -> list[dict]:
    """Tool specs for code mode: search_tools + run_code."""
    return [
        {"type": "function", "function": {
            "name": "search_tools",
            "description": "Find tools by need. Returns matches with schema inline.",
            "parameters": {"type": "object",
                           "properties": {"q": {"type": "string", "description": "the need"}},
                           "required": ["q"]}}},
        {"type": "function", "function": {
            "name": "run_code",
            "description": ("Run Python that orchestrates tools in ONE turn. Available: "
                            "call(path, **params)->dict, search(q)->[paths], schema(path)->str. "
                            "Assign the final data to `output`."),
            "parameters": {"type": "object",
                           "properties": {"code": {"type": "string", "description": "python code"}},
                           "required": ["code"]}}},
    ]


def run_code(target, code: str, *, max_calls: int = 50) -> str:
    """Execute a tool-orchestration snippet; returns JSON {output|stdout|error}.

    ``target`` provides ``execute_tool`` / ``search_tools`` / ``get_tool_schema`` —
    pass a ``Sift`` (unscoped) or a ``SiftScope`` (so ``call`` obeys allow/deny).

    Threat model: this blocks the in-process exec escape vectors (imports, dunder
    attrs, dangerous names, str.format traversal) and bounds Python loops via a
    line budget. It does NOT bound a single C-level call (e.g. building a huge
    object) and is not a VM. Use code mode with trusted catalogues; for untrusted
    input run the host process under OS-level isolation.
    """
    budget = {"n": 0}

    # path/query are positional-only (/) so a tool param named 'path' or 'q'
    # never collides with these helper signatures.
    def call(_path: str, /, **params) -> dict:
        budget["n"] += 1
        if budget["n"] > max_calls:
            raise RuntimeError(f"exceeded {max_calls} tool calls")
        return target.execute_tool(_path, params)

    def search(_q: str, /, top_k: int = 5) -> list[str]:
        return [r.path for r in target.search_tools(_q, top_k)]

    def schema(_path: str, /) -> str:
        return target.get_tool_schema(_path)

    try:
        tree = _validate(code)  # AST policy: no imports / dunder attrs / dangerous names
    except SandboxError as exc:
        return json.dumps({"error": f"SandboxError: {exc}"}, ensure_ascii=False)

    ns = {"__builtins__": _SAFE_BUILTINS, "call": call, "search": search,
          "schema": schema, "output": None}
    buf = io.StringIO()
    code_obj = compile(tree, "<sift-code>", "exec")
    had_trace = sys.gettrace()
    try:
        sys.settrace(_line_limiter())
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
