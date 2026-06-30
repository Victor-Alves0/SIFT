"""Canonical definitions of the 3 meta-tools and the system prompt.

Shared by every adapter (OpenAI, LangChain, MCP) so the contract stays identical
no matter how SIFT is wired into an agent.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You discover tools dynamically — you never see the full catalogue.

Tools:
1. search_tools(q)            — find tools by need. Returns matches ALREADY WITH their schema, one per line (TOON): path|desc|param:type:req[:default]|r:fields[|risk]  (req: n=required, o=optional; "risk"=high-impact).
2. get_tool_schema(path)      — only to browse the hierarchy ("" lists categories, then a category, then a service). Rarely needed.
3. execute_tool(path, params) — run a function by its full path with parameters.

Flow: call search_tools, pick the best matching path, then call execute_tool DIRECTLY — the schema is already in the search result, so do NOT call get_tool_schema for it. "risk" actions: proceed only if the user authorised it. Results come pre-filtered; don't invent fields. Answer the user concisely when done."""


def tool_specs() -> list[dict]:
    """OpenAI/OpenRouter-style function-calling specs for the 3 meta-tools."""
    def s(desc: str) -> dict:
        return {"type": "string", "description": desc}

    return [
        {
            "type": "function",
            "function": {
                "name": "search_tools",
                "description": "Find tools by need. Returns the best matches WITH their schema inline — execute directly, no get_tool_schema needed.",
                "parameters": {
                    "type": "object",
                    "properties": {"q": s("the need, e.g. 'read last email'")},
                    "required": ["q"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_tool_schema",
                "description": "Browse the hierarchy (categories/services). Usually unnecessary — search_tools already returns schemas.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": s("'' lists categories, then 'cat', then 'cat.service'")},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_tool",
                "description": "Execute a function by full path; returns the filtered result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": s("full function path, e.g. 'google_workspace.gmail.read'"),
                        "params": {"type": "object", "description": "function parameters per its schema"},
                    },
                    "required": ["path"],
                },
            },
        },
    ]


META_TOOL_NAMES = ("search_tools", "get_tool_schema", "execute_tool")
