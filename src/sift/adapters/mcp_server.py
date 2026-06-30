"""MCP server adapter — expose SIFT as a Model Context Protocol server.

Any MCP client (Claude Desktop, IDEs, etc.) then sees just the 3 meta-tools and
discovers your whole catalogue through them.

    from sift import Sift
    sift = Sift(); ...; sift.build_index()
    sift.serve_mcp()                 # runs a stdio MCP server

Requires the ``mcp`` extra:  pip install "sift-tools[mcp]"
"""
from __future__ import annotations


def build_mcp_server(sift, name: str = "sift"):
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(name)

    @server.tool()
    def search_tools(q: str) -> str:
        """Discover tools by natural language; returns candidate paths with scores."""
        return sift.dispatch("search_tools", {"q": q})

    @server.tool()
    def get_tool_schema(path: str) -> str:
        """Compact (TOON) schema of a hierarchy level. Empty path lists categories."""
        return sift.dispatch("get_tool_schema", {"path": path})

    @server.tool()
    def execute_tool(path: str, params: dict | None = None) -> str:
        """Execute a function (full path category.service.function); returns the filtered result."""
        return sift.dispatch("execute_tool", {"path": path, "params": params or {}})

    return server
