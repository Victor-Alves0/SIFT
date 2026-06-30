"""Live MCP execution proxy.

An MCP connection is async and stateful, but SIFT's executor interface is a plain
sync ``executor(tool_name, params) -> dict``. ``StdioMcpProxy`` bridges the two:
it launches a stdio MCP server, keeps the session open on a background event loop,
and turns each call into a synchronous ``call_tool``. So imported MCP tools become
runnable out of the box:

    from sift import Sift
    from sift.importers.mcp_proxy import connect_mcp_stdio

    sift = Sift()
    proxy = connect_mcp_stdio(sift, "npx", ["-y", "@modelcontextprotocol/server-github"],
                              category="integrations", service="github")
    sift.build_index()
    ...
    proxy.close()   # or use it as a context manager

Requires the ``mcp`` extra:  pip install "sift-tools[mcp]"
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any


def _content_to_dict(result: Any) -> dict:
    """Normalise an MCP CallToolResult into a plain dict."""
    is_error = bool(getattr(result, "isError", False))

    texts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
    joined = "\n".join(texts)

    if joined:
        try:
            data = json.loads(joined)
        except (json.JSONDecodeError, ValueError):
            return {"content": joined, "isError": is_error}
        if isinstance(data, dict):
            if is_error:
                data.setdefault("isError", True)
            return data
        return {"result": data, "isError": is_error}

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    return {"isError": is_error}


class StdioMcpProxy:
    """Holds a stdio MCP session open and exposes a sync executor."""

    def __init__(self, command: str, args: list[str] | None = None,
                 env: dict | None = None, *, timeout: float = 60.0) -> None:
        self.command = command
        self.args = list(args or [])
        self.env = env
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._session = None
        self._tools: list = []
        self._stop: asyncio.Event | None = None
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._error:
            raise self._error

    def _thread_main(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:  # pragma: no cover - surfaced via _error
            self._error = exc
            self._ready.set()

    async def _serve(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stop = asyncio.Event()
        try:
            params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._tools = (await session.list_tools()).tools
                    self._ready.set()
                    await self._stop.wait()
        except Exception as exc:
            self._error = exc
            self._ready.set()

    def list_tools(self) -> list:
        return self._tools

    def call(self, tool_name: str, params: dict | None = None) -> dict:
        if self._session is None:
            raise RuntimeError("MCP session is not ready")
        fut = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(tool_name, params or {}), self._loop)
        return _content_to_dict(fut.result(self.timeout))

    # executor interface used by register_listing: executor(tool_name, params) -> dict
    __call__ = call

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=5)

    def __enter__(self) -> "StdioMcpProxy":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def connect_mcp_stdio(target, command: str, args: list[str] | None = None, *,
                      category: str, service: str, env: dict | None = None,
                      timeout: float = 60.0) -> StdioMcpProxy:
    """Connect to a stdio MCP server, register its tools INTO ``target`` (a Sift or
    Registry) with live execution bound, and return the proxy (close it when done)."""
    from .mcp import register_listing

    proxy = StdioMcpProxy(command, args, env, timeout=timeout)
    register_listing(target, proxy.list_tools(), category=category, service=service,
                     executor=proxy)
    return proxy
