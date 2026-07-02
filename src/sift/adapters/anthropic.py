"""Native Anthropic (Messages API) adapter.

Anthropic's tool format differs from OpenAI's: tools use ``input_schema`` (not a
``function`` wrapper), the system prompt is a separate argument, and tool calls
come back as ``tool_use`` content blocks answered with ``tool_result`` blocks.
This adapter bridges that so SIFT works with the native ``anthropic`` SDK.

    import anthropic
    from sift.adapters.anthropic import run_agent
    run_agent(sift, anthropic.Anthropic(), "claude-haiku-4.5", "what's my last email?")

Requires the ``anthropic`` extra:  pip install "sift-tools[anthropic]"
"""
from __future__ import annotations

from typing import Any


def anthropic_tools(sift) -> list[dict]:
    """The 2 meta-tools in Anthropic's tool format."""
    out = []
    for spec in sift.openai_tools():
        fn = spec["function"]
        out.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        })
    return out


def _text_of(content: Any) -> str:
    parts = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


# --------------------------------------------------------------------------
# Native tool search integration (defer_loading + custom client-side search)
#
# Anthropic's tool search tool ships regex and BM25 variants — but the API also
# supports a CUSTOM search tool that returns `tool_reference` blocks, which the
# API expands into full definitions exactly like the built-ins. That custom slot
# is where SIFT plugs in: the whole catalogue goes up as `defer_loading: true`
# tools, and discovery runs through SIFT's hybrid retrieval + active tool
# request instead of regex/BM25.
#
#     tools = deferred_tools(sift)
#     resp  = client.messages.create(model=..., tools=tools, ...)
#     # when Claude calls "search_tools", answer with tool_search_result(...)
#
# Tool names can't contain dots, so paths map `.` -> `__`
# (google_workspace.gmail.read -> google_workspace__gmail__read).
# --------------------------------------------------------------------------

_SEARCH_TOOL = {
    "name": "search_tools",
    "description": ("Search the tool catalog. Preferred: an active request via domain "
                    "(platform/area, e.g. 'email') + action (operation + target, e.g. "
                    "'read the latest message'). Or q for a plain query. Returns "
                    "references to the matching tools, which are then available to call."),
    "input_schema": {
        "type": "object",
        "properties": {
            "domain": {"type": "string", "description": "platform / permission area"},
            "action": {"type": "string", "description": "operation + target"},
            "q": {"type": "string", "description": "plain natural-language query"},
        },
    },
}


def deferred_tools(sift, *, keep: tuple[str, ...] = ()) -> list[dict]:
    """The full catalogue as Anthropic ``defer_loading`` tools + SIFT's search
    tool (non-deferred). Paths in ``keep`` stay non-deferred (your 3–5 most-used
    tools, callable without a search)."""
    from ..registry import input_schema_for
    from ..session import promoted_name

    reg = getattr(sift, "registry", None) or sift._sift.registry
    tools: list[dict] = [dict(_SEARCH_TOOL)]
    for tool in reg.tools():
        tools.append({
            "name": promoted_name(tool.path),
            "description": tool.description + (" [risk: confirm first]" if tool.risk else ""),
            "input_schema": input_schema_for(tool),
            "defer_loading": tool.path not in keep,
        })
    return tools


def tool_search_result(sift, tool_use_id: str, args: dict, *, top_k: int = 5) -> dict:
    """The ``tool_result`` block answering a ``search_tools`` call: standard
    content with ``tool_reference`` blocks, which the API expands server-side."""
    from ..session import promoted_name

    domain = (args.get("domain") or "").strip()
    action = (args.get("action") or "").strip()
    if domain or action:
        results = sift.search_request(domain, action, top_k)
    else:
        results = sift.search_tools((args.get("q") or "").strip(), top_k)
    refs = [{"type": "tool_reference", "tool_name": promoted_name(r.path)}
            for r in results if r.kind == "function"]
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": refs}


def run_agent_deferred(sift, client: Any, model: str, message: str, *,
                       keep: tuple[str, ...] = (), max_tokens: int = 1024,
                       max_steps: int = 12, verbose: bool = False,
                       extra: dict | None = None) -> str:
    """Tool-use loop over the deferred catalogue: SIFT answers ``search_tools``
    calls with tool references; discovered tools are executed by path."""
    import json as _json

    tools = deferred_tools(sift, keep=keep)
    messages: list[dict] = [{"role": "user", "content": message}]

    for _ in range(max_steps):
        resp = client.messages.create(
            model=model, system=sift.system_prompt, tools=tools,
            max_tokens=max_tokens, messages=messages, **(extra or {}))
        messages.append({"role": "assistant", "content": resp.content})

        if getattr(resp, "stop_reason", None) != "tool_use":
            return _text_of(resp.content)

        results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if block.name == "search_tools":
                out = tool_search_result(sift, block.id, block.input or {})
            else:
                path = block.name.replace("__", ".")
                body = sift.dispatch("execute_tool", {"path": path, "params": block.input or {}})
                out = {"type": "tool_result", "tool_use_id": block.id, "content": body}
            if verbose:
                print(f"  ↳ {block.name}({block.input}) = {_json.dumps(out)[:160]}")
            results.append(out)
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"reached max_steps={max_steps} without a final answer")


def run_agent(sift, client: Any, model: str, message: str, *,
              max_tokens: int = 1024, max_steps: int = 12, verbose: bool = False,
              extra: dict | None = None) -> str:
    """Drive a tool-use loop against the native Anthropic Messages API.

    ``client`` is duck-typed: it just needs ``messages.create(...)``.
    """
    tools = anthropic_tools(sift)
    messages: list[dict] = [{"role": "user", "content": message}]

    for _ in range(max_steps):
        resp = client.messages.create(
            model=model, system=sift.system_prompt, tools=tools,
            max_tokens=max_tokens, messages=messages, **(extra or {}))

        messages.append({"role": "assistant", "content": resp.content})

        if getattr(resp, "stop_reason", None) != "tool_use":
            return _text_of(resp.content)

        results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            out = sift.dispatch(block.name, block.input)
            if verbose:
                print(f"  ↳ {block.name}({block.input}) = {out[:160]}")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"reached max_steps={max_steps} without a final answer")
