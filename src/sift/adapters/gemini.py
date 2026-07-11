"""Native Google Gemini adapter (google-genai SDK).

Gemini also speaks the OpenAI-compatible endpoint, but its native SDK uses
``function_declarations`` tools, ``function_call`` parts and ``function_response``
replies. This adapter bridges SIFT to that shape with plain dicts, duck-typing
the client (``client.models.generate_content(...)``) so it is testable offline.

    from google import genai
    from sift.adapters.gemini import run_agent
    run_agent(sift, genai.Client(), "gemini-2.0-flash", "what's my last email?")

Requires the ``gemini`` extra:  pip install "sift-tools[gemini]"
"""
from __future__ import annotations

from typing import Any


def gemini_tools(sift) -> list[dict]:
    """The meta-tools (plus any pinned tools) as Gemini function declarations."""
    decls = []
    for spec in sift.openai_tools():
        fn = spec["function"]
        decls.append({"name": fn["name"], "description": fn["description"],
                      "parameters": fn["parameters"]})
    return [{"function_declarations": decls}]


def _function_calls(resp: Any) -> list:
    cand = resp.candidates[0]
    parts = getattr(cand.content, "parts", None) or []
    return [p.function_call for p in parts if getattr(p, "function_call", None)]


def run_agent(sift, client: Any, model: str, message: str, *,
              max_steps: int = 12, verbose: bool = False,
              extra_config: dict | None = None) -> str:
    """Drive a tool-use loop against the native Gemini API."""
    config = {"tools": gemini_tools(sift), "system_instruction": sift.system_prompt,
              **(extra_config or {})}
    contents: list = [{"role": "user", "parts": [{"text": message}]}]

    for _ in range(max_steps):
        resp = client.models.generate_content(model=model, contents=contents, config=config)
        calls = _function_calls(resp)
        if not calls:
            return getattr(resp, "text", "") or ""

        contents.append(resp.candidates[0].content)   # the model's tool-call turn
        replies = []
        for call in calls:
            result = sift.dispatch(call.name, dict(call.args or {}))
            if verbose:
                print(f"  ↳ {call.name}({dict(call.args or {})}) = {result[:160]}")
            replies.append({"function_response": {"name": call.name,
                                                  "response": {"result": result}}})
        contents.append({"role": "user", "parts": replies})

    raise RuntimeError(f"reached max_steps={max_steps} without a final answer")
