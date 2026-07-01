"""MCP server adapter — expose SIFT as a Model Context Protocol server.

Any MCP client (Claude Desktop, IDEs, etc.) then sees just the 2 meta-tools and
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
    def search_tools(domain: str = "", action: str = "", q: str = "", path: str = "") -> str:
        """Find tools by need — matches come back with their schema inline. Preferred: an
        active tool request via domain (platform/permission area) + action (operation +
        target). Or pass q for a simple search, or path to browse (empty path lists
        categories)."""
        return sift.dispatch("search_tools", {"domain": domain, "action": action,
                                              "q": q, "path": path})

    @server.tool()
    def execute_tool(path: str, params: dict | None = None) -> str:
        """Execute a function (full path category.service.function); returns the filtered result."""
        return sift.dispatch("execute_tool", {"path": path, "params": params or {}})

    return server
