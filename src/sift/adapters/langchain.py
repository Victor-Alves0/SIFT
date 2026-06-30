"""LangChain adapter — exposes the 3 meta-tools as LangChain ``StructuredTool``s.

    from sift import Sift
    sift = Sift(); ...; sift.build_index()
    tools = sift.langchain_tools()           # plug into any LangChain agent

Requires the ``langchain`` extra:  pip install "sift-tools[langchain]"
"""
from __future__ import annotations


def langchain_tools(sift) -> list:
    from langchain_core.tools import StructuredTool

    def search_tools(q: str) -> str:
        """Discover tools by natural language; returns candidate paths with scores."""
        return sift.dispatch("search_tools", {"q": q})

    def get_tool_schema(path: str) -> str:
        """Compact (TOON) schema of a hierarchy level. Empty path lists categories."""
        return sift.dispatch("get_tool_schema", {"path": path})

    def execute_tool(path: str, params: dict | None = None) -> str:
        """Execute a function (full path category.service.function) and return the filtered result."""
        return sift.dispatch("execute_tool", {"path": path, "params": params or {}})

    return [
        StructuredTool.from_function(search_tools, name="search_tools"),
        StructuredTool.from_function(get_tool_schema, name="get_tool_schema"),
        StructuredTool.from_function(execute_tool, name="execute_tool"),
    ]
