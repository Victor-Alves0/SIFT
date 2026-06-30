"""Prompted (text-based) tool calling — for models WITHOUT native function calling.

The core ``sift.dispatch(name, args)`` is format-agnostic, so any text model can
drive SIFT if we (a) tell it a JSON protocol in the prompt and (b) parse its plain
text back. This adapter does both, looping search -> execute -> answer.

``generate`` is the only thing you supply: a callable ``generate(prompt: str) ->
str``. That wraps ANYTHING — a HuggingFace pipeline, llama.cpp, Ollama /generate,
a base model — so SIFT reaches models the native adapters can't.

For weak models, pair this with constrained decoding (see ``sift.constrain``) and
prefer :func:`single_decision`.
"""
from __future__ import annotations

import json
import re
from typing import Callable

PROMPTED_SYSTEM = """You solve the user's task using tools, by emitting JSON.

You have 3 tools:
- search_tools     args: {"q": "<what you need>"}   -> returns matching tool paths WITH their schema
- get_tool_schema  args: {"path": "<'' | category | category.service>"}  -> browse (rarely needed)
- execute_tool     args: {"path": "<category.service.function>", "params": {...}}  -> run a tool

PROTOCOL — every reply MUST be exactly ONE JSON object and nothing else:
  to use a tool:        {"tool": "<name>", "args": {...}}
  to answer the user:   {"answer": "<your reply>"}

Always start with search_tools, then execute_tool with a full path from the results,
then give {"answer": ...}. Tool results are provided back to you as JSON."""


def _extract_json(text: str) -> dict | None:
    """Best-effort: pull the first JSON object out of a model's text reply."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def run_agent(sift, generate: Callable[[str], str], message: str, *,
              max_steps: int = 10, verbose: bool = False) -> str:
    """Drive a text model through the tool loop. ``generate(prompt)->str``."""
    transcript = f"{PROMPTED_SYSTEM}\n\nUser: {message}\n"

    for _ in range(max_steps):
        reply = generate(transcript + "Assistant:")
        obj = _extract_json(reply)

        if obj is None:
            transcript += (f"Assistant: {reply.strip()[:200]}\n"
                           'System: Invalid. Reply with ONE JSON object: '
                           '{"tool": ...} or {"answer": ...}.\n')
            continue

        if "answer" in obj:
            return str(obj["answer"])

        if "tool" in obj:
            result = sift.dispatch(obj["tool"], obj.get("args") or {})
            if verbose:
                print(f"  ↳ {obj['tool']}({obj.get('args')}) = {result[:160]}")
            transcript += (f'Assistant: {json.dumps(obj, ensure_ascii=False)}\n'
                           f"Tool result: {result}\n")
            continue

        transcript += ('System: JSON must contain "tool" or "answer".\n')

    raise RuntimeError(f"reached max_steps={max_steps} without an answer")


def single_decision(sift, generate: Callable[[str], str], query: str, *,
                    top_k: int = 3) -> dict:
    """One-shot path for very weak models: search server-side, then ask the model
    for a SINGLE decision (which tool + args). Returns {path, args, result}."""
    candidates = sift.gateway.search_compact(query, top_k)
    prompt = (
        f"{candidates}\n\n"
        f"User wants: {query}\n"
        'Reply with ONLY JSON: {"path": "<one path above>", "args": {<parameters>}}'
    )
    obj = _extract_json(generate(prompt)) or {}
    path = obj.get("path")
    args = obj.get("args") or {}
    try:
        result = sift.execute_tool(path, args) if path else {"error": "no path chosen"}
    except Exception as exc:
        result = {"error": str(exc)}
    return {"path": path, "args": args, "result": result}
