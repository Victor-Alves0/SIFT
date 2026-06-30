"""Importers: turn existing MCP servers and OpenAPI specs into hierarchy nodes,
so an existing ecosystem of tools becomes searchable through SIFT's meta-tools.
"""
from ._common import compress_params
from .mcp_proxy import StdioMcpProxy, connect_mcp_stdio

__all__ = ["compress_params", "StdioMcpProxy", "connect_mcp_stdio"]
