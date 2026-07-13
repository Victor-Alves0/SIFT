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

from .sandbox import SANDBOX_RULES, InProcessSandbox, SandboxError  # re-exported for back-compat

__all__ = ["CODE_SYSTEM_PROMPT", "SANDBOX_RULES", "code_tool_specs", "run_code", "SandboxError"]

# The sandbox's limits are stated up front (SANDBOX_RULES is generated FROM the
# policy itself, so it cannot go stale): every rule the model learns by failing
# costs a full context round-trip, which dwarfs the ~70 tokens of stating it.
CODE_SYSTEM_PROMPT = f"""You orchestrate tools by WRITING CODE, composing many calls in a single turn.

Tools:
1. search_tools(q) — find tool paths (schema inline) by need.
2. execute_tool(path, params) — run ONE tool. Use this when one call answers the
   request: it is cheaper and safer than writing code for it.
3. run_code(code) — for anything COMPOSITE (2+ calls, loops, conditionals, or
   filtering a big result). Runs Python NOW. Inside it you have:
     call(path, **params) -> dict   # execute a tool, returns its filtered result
     search(q) -> [paths]           # discovery; returns matching tool paths
     schema(path) -> str            # TOON schema of a tool/level
   Assign the data you want back to a variable named `output` — or just leave it as
   the last expression, which is promoted to `output` like in a REPL.

Keep `output` SMALL: filter, slice and aggregate inside the snippet. Everything you
put in `output` is re-sent with the whole conversation on every later turn, while
intermediate values stay in the sandbox for free. Return the 5 rows you need, the
count, the one field — never a raw tool payload you are not going to read.

{SANDBOX_RULES}

Flow: search_tools to learn paths → execute_tool for a single call, or ONE run_code
that performs all the calls and sets `output`. Then answer the user concisely.
"risk" tools (send/delete): only run them if the user authorised it."""


def code_tool_specs() -> list[dict]:
    """Tool specs for code mode: search_tools + execute_tool + run_code.

    ``execute_tool`` is here on purpose. Code mode without it forces the model to
    write Python even for a single call — paying sandbox overhead and a real
    parse-failure rate to do what one JSON call does. Code mode wins on COMPOSITE
    work (many calls, control flow, filtering a big result); for one call, direct
    execution is cheaper and cannot fail to compile.
    """
    return [
        {"type": "function", "function": {
            "name": "search_tools",
            "description": "Find tools by need. Returns matches with schema inline.",
            "parameters": {"type": "object",
                           "properties": {"q": {"type": "string", "description": "the need"}},
                           "required": ["q"]}}},
        {"type": "function", "function": {
            "name": "execute_tool",
            "description": ("Run ONE tool by path. Prefer this over run_code when a single "
                            "call answers the request — no code to write, nothing to compile."),
            "parameters": {"type": "object",
                           "properties": {
                               "path": {"type": "string", "description": "full tool path"},
                               "params": {"type": "object", "description": "tool arguments"}},
                           "required": ["path"]}}},
        {"type": "function", "function": {
            "name": "run_code",
            "description": ("Run Python that orchestrates SEVERAL tools in ONE turn (2+ calls, "
                            "loops, or filtering a big result). Available: call(path, **params)"
                            "->dict, search(q)->[paths], schema(path)->str. No imports. Assign "
                            "the final data to `output` (or leave it as the last expression), "
                            "and keep it small — filter inside the snippet."),
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
