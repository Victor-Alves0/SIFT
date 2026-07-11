"""Canonical definitions of the meta-tools and the system prompt.

Shared by every adapter (OpenAI, LangChain, MCP) so the contract stays identical
no matter how SIFT is wired into an agent.

The surface is TWO tools: ``search_tools`` (discovery — schema returned inline, and
it also browses the hierarchy via ``path``) and ``execute_tool``. ``get_tool_schema``
is kept as a deprecated alias in ``dispatch`` for back-compat, but is no longer
part of the advertised surface.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You discover tools dynamically — you never see the full catalogue.

Tools:
1. search_tools(...)  — find tools by need. Returns the best matches ALREADY WITH their schema, one per line (TOON): path|desc|param:type:req[:default]|r:fields[|risk]  (req: n=required, o=optional; "risk"=high-impact). Three ways to call it:
   • BEST — an active tool request: state your intent as two fields, domain (the platform / permission area, e.g. "email", "calendar") and action (the operation + target, e.g. "read the latest message"). This aligns better with the tools than a raw query and is the preferred form.
   • simple — search_tools(q="read my last email") with a single natural-language need.
   • browse — search_tools(path="") lists categories, then path="category", then path="category.service".
2. execute_tool(path, params)  — run a function by its full path with parameters.

Flow: call search_tools (prefer domain+action), pick the best matching path, then call execute_tool DIRECTLY — the schema is already in the search result. If the matches don't fit, refine domain/action and search again. Write search queries in the language of the tool descriptions (usually English) — translate the user's need if it is phrased in another language. "risk" actions: proceed only if the user authorised it. Results come pre-filtered; don't invent fields. Answer the user concisely when done."""


def tool_specs() -> list[dict]:
    """OpenAI/OpenRouter-style function-calling specs for the 2 meta-tools."""
    def s(desc: str) -> dict:
        return {"type": "string", "description": desc}

    return [
        {
            "type": "function",
            "function": {
                "name": "search_tools",
                "description": ("Find tools by need — returns the best matches WITH their schema "
                                "inline, so you can execute directly. Preferred: an active tool "
                                "request via domain + action (a model-authored intent aligns "
                                "better with tools than a raw query). Or pass q for a simple "
                                "search, or path to browse the hierarchy (empty path lists "
                                "categories)."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain": s("active request: the platform / permission area, e.g. 'email', 'calendar', 'crm'"),
                        "action": s("active request: the operation + target, e.g. 'read the latest message'"),
                        "q": s("simple search: the need in natural language, e.g. 'read last email'"),
                        "path": s("browse a level instead — '', 'category', or 'category.service'"),
                    },
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


META_TOOL_NAMES = ("search_tools", "execute_tool")
