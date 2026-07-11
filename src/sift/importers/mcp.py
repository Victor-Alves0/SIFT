"""Import an MCP server's tools into the SIFT hierarchy.

Two layers:
* ``tools_from_listing`` — pure converter over an already-fetched tool listing
  (testable, no network).
* ``import_mcp_stdio`` — connects to a stdio MCP server, lists its tools and
  registers them. Requires the ``mcp`` extra.
"""
from __future__ import annotations

from typing import Callable, Iterable

from ..registry import Registry, ToolDef
from ._common import compress_params, looks_destructive


def _attr(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def tools_from_listing(listing: Iterable, *, category: str, service: str,
                       sanitize: bool = True) -> list[ToolDef]:
    """Convert MCP tool descriptors (dicts or objects) into ToolDefs.

    Each descriptor needs ``name``, ``description`` and ``inputSchema``.
    ``sanitize`` (default on) scrubs third-party descriptions before they enter
    your index and the model's context — control chars stripped, whitespace
    collapsed (multi-line "instructions" flatten), length capped. Hygiene, not a
    guarantee: see docs/security.md.
    """
    from ._common import sanitize_text
    defs: list[ToolDef] = []
    for t in listing:
        name = _attr(t, "name")
        if not name:
            continue
        desc = _attr(t, "description") or ""
        desc = sanitize_text(desc, max_len=200) if sanitize else desc[:200]
        schema = _attr(t, "inputSchema") or _attr(t, "input_schema") or {}
        params = compress_params(schema)
        if sanitize:
            for p in params.values():
                p["desc"] = sanitize_text(p["desc"], max_len=150)
        defs.append(
            ToolDef(
                path=f"{category}.{service}.{name}",
                description=desc,
                params=params,
                returns=[],  # MCP tools don't declare a response whitelist
                risk=looks_destructive(name),
            )
        )
    return defs


def _registry_of(target) -> Registry:
    return target.registry if hasattr(target, "registry") else target


def _bind_executor(executor: Callable[[str, dict], dict], tool_name: str):
    def _call(**kwargs) -> dict:
        return executor(tool_name, kwargs)
    return _call


def register_listing(target, listing: Iterable, *, category: str, service: str,
                     executor: Callable[[str, dict], dict] | None = None) -> int:
    """Register a fetched listing into a Sift/Registry. Returns count added.

    Pass ``executor(tool_name, params) -> dict`` to make the imported tools
    runnable (e.g. a proxy that performs the live MCP call). Without it they are
    discoverable/inspectable but ``execute_tool`` will report no executor bound.
    """
    reg = _registry_of(target)
    defs = tools_from_listing(listing, category=category, service=service)
    for d in defs:
        if executor is not None:
            d.fn = _bind_executor(executor, d.path.rsplit(".", 1)[-1])
        reg.add(d)
    return len(defs)


async def import_mcp_stdio(target, command: str, args: list[str] | None = None, *,
                           category: str, service: str,
                           executor: Callable[[str, dict], dict] | None = None) -> int:
    """Connect to a stdio MCP server, list its tools and register them.

    Pass ``executor`` (see :func:`register_listing`) to also bind live execution.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=args or [])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = (await session.list_tools()).tools
    return register_listing(target, listing, category=category, service=service, executor=executor)
