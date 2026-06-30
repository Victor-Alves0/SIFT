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
    """The 3 meta-tools in Anthropic's tool format."""
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
