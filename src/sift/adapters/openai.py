"""OpenAI / OpenRouter (function-calling) adapter.

``sift.openai_tools()`` already returns the tool specs. This module adds a small
driver that runs a full agent loop against any OpenAI-compatible client (the
official ``openai`` SDK, OpenRouter via the same SDK, Azure, etc.). The client is
duck-typed: it just needs ``client.chat.completions.create(...)``.
"""
from __future__ import annotations

from typing import Any


def run_agent(sift, client: Any, model: str, message: str, *,
              max_steps: int = 12, verbose: bool = False,
              extra_body: dict | None = None) -> str:
    """Drive a tool-calling loop until the model returns a final answer.

    ``extra_body`` is forwarded to ``chat.completions.create`` — use it for
    provider extras like prompt caching or ``{"reasoning": {"effort": "low"}}``
    to keep tool-routing turns cheap.
    """
    messages: list[dict] = [
        {"role": "system", "content": sift.system_prompt},
        {"role": "user", "content": message},
    ]
    tools = sift.openai_tools()

    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools, extra_body=extra_body or {})
        msg = resp.choices[0].message

        assistant: dict = {"role": "assistant", "content": msg.content or ""}
        if getattr(msg, "tool_calls", None):
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant)

        if not getattr(msg, "tool_calls", None):
            return msg.content or ""

        for tc in msg.tool_calls:
            result = sift.dispatch(tc.function.name, tc.function.arguments)
            if verbose:
                print(f"  ↳ {tc.function.name}({tc.function.arguments}) = {result[:200]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result,
            })

    raise RuntimeError(f"reached max_steps={max_steps} without a final answer")


def run_agent_responses(sift, client: Any, model: str, message: str, *,
                        max_steps: int = 12, verbose: bool = False,
                        extra: dict | None = None) -> str:
    """Same loop over the OpenAI **Responses API** (the successor to chat
    completions). ``client`` just needs ``responses.create(...)``.
    """
    # Responses uses a flat tool shape (no "function" wrapper)
    tools = [{"type": "function", "name": f["function"]["name"],
              "description": f["function"]["description"],
              "parameters": f["function"]["parameters"]}
             for f in sift.openai_tools()]
    items: list[dict] = [{"role": "user", "content": message}]

    for _ in range(max_steps):
        resp = client.responses.create(model=model, instructions=sift.system_prompt,
                                       input=items, tools=tools, **(extra or {}))
        calls = [o for o in resp.output if getattr(o, "type", None) == "function_call"]
        if not calls:
            return getattr(resp, "output_text", "") or ""
        for call in calls:
            items.append({"type": "function_call", "call_id": call.call_id,
                          "name": call.name, "arguments": call.arguments})
            result = sift.dispatch(call.name, call.arguments)
            if verbose:
                print(f"  ↳ {call.name}({call.arguments}) = {result[:160]}")
            items.append({"type": "function_call_output", "call_id": call.call_id,
                          "output": result})

    raise RuntimeError(f"reached max_steps={max_steps} without a final answer")
