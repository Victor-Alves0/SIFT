"""LangChain adapter — exposes the 2 meta-tools as LangChain ``StructuredTool``s.

    from sift import Sift
    sift = Sift(); ...; sift.build_index()
    tools = sift.langchain_tools()           # plug into any LangChain agent

Requires the ``langchain`` extra:  pip install "sift-tools[langchain]"
"""
from __future__ import annotations


def langchain_tools(sift) -> list:
    from langchain_core.tools import StructuredTool

    def search_tools(q: str = "", path: str = "", domain: str = "", action: str = "") -> str:
        """Find tools by need — matches come back with their schema inline.
        Preferred: an active request via domain (platform/area) + action
        (operation + target). Or q for a simple search, or path to browse the
        hierarchy ('' lists categories, then 'category', then 'category.service')."""
        return sift.dispatch("search_tools",
                             {"q": q, "path": path, "domain": domain, "action": action})

    def execute_tool(path: str, params: dict | None = None) -> str:
        """Execute a function (full path category.service.function) and return the filtered result."""
        return sift.dispatch("execute_tool", {"path": path, "params": params or {}})

    return [
        StructuredTool.from_function(search_tools, name="search_tools"),
        StructuredTool.from_function(execute_tool, name="execute_tool"),
    ]
