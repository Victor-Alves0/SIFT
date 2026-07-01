"""Code mode — let the model orchestrate tools by writing code.

Instead of one round-trip per tool, the model emits a short Python snippet that
composes many tool calls in a SINGLE turn (the StackOne/Cloudflare "code mode"
pattern). This collapses the multi-turn overhead for composite tasks.

Execution goes through a pluggable sandbox backend (see :mod:`sift.sandbox`):
``InProcessSandbox`` (default, fast, for trusted catalogues) or
``SubprocessSandbox`` (isolated process + resource limits, tool calls proxied
back to the parent). The snippet runs in a constrained namespace exposing
``call``, ``search`` and ``schema``.
"""
from __future__ import annotations

from .sandbox import InProcessSandbox, SandboxError  # re-exported for back-compat

__all__ = ["CODE_SYSTEM_PROMPT", "code_tool_specs", "run_code", "SandboxError"]

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


def run_code(target, code: str, *, sandbox=None, max_calls: int = 50) -> str:
    """Execute a tool-orchestration snippet; returns JSON {output|stdout|error}.

    ``target`` provides ``execute_tool`` / ``search_tools`` / ``get_tool_schema``
    (a ``Sift`` — unscoped — or a ``SiftScope`` — so ``call`` obeys allow/deny).
    ``sandbox`` selects the backend (default: in-process).
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

    return (sandbox or InProcessSandbox()).run(code, call, search, schema)
