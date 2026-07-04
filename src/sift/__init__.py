"""SIFT — Search · Inspect · Filter · Trigger.

Hierarchical, search-first tool discovery for LLM agents. Give the model 2
meta-tools instead of a 30k-token catalogue; it discovers the rest by navigating.

Quickstart::

    from sift import Sift

    sift = Sift()

    @sift.tool("google_workspace.gmail.read",
               description="Read emails from the inbox",
               params={"q": "string:o:is:unread:search query", "m": "number:o:10:max"},
               returns=["id", "subject", "from", "snippet", "date"])
    def gmail_read(q="is:unread", m=10):
        return {"id": "1", "subject": "Hi", "from": "a@b.c", "snippet": "...",
                "date": "2026-06-30", "body": "filtered out"}

    sift.build_index()

    # discovery — schema comes back inline, so you can execute directly
    sift.search_request(domain="email", action="read the latest message")  # active request
    sift.search_tools("read my last email")                     # or a simple query
    sift.execute_tool("google_workspace.gmail.read", {"m": 1})  # run + filter

    # Plug into any stack:
    sift.openai_tools()       # function-calling specs
    sift.langchain_tools()    # LangChain BaseTool list
    sift.serve_mcp()          # expose as an MCP server
"""
from __future__ import annotations

import json
import logging
from typing import Callable

from .gateway import Gateway, SearchResult
from .metatools import META_TOOL_NAMES, SYSTEM_PROMPT, tool_specs
from .registry import Registry, ToolDef

__all__ = ["Sift", "Registry", "ToolDef", "SearchResult", "SYSTEM_PROMPT", "tool_specs"]
__version__ = "0.5.0"

_log = logging.getLogger("sift")


class Sift:
    """The public facade: register tools, build the index, expose meta-tools.

    Notable knobs:

    - ``index_cache``: file path persisting document vectors across process
      restarts (cold start drops from tens of seconds to ~ms on big catalogues).
    - ``max_result_chars``: cap on any tool result string sent to the model
      (default 100k chars; ``None``/0 disables). Truncation appends a marker
      telling the model the result was cut and how to trim the tool.
    - ``observer``: ``callable(event: str, data: dict)`` receiving ``search`` /
      ``execute`` / ``run_code`` events with timing — hook tracing/metrics here.

    Thread-safety: register tools and ``build_index()`` first, then serve; after
    the build, discovery/execution are read-only on SIFT's side and safe to call
    concurrently (your tool functions' own thread-safety is on you).
    """

    def __init__(self, *, registry: Registry | None = None, embedder=None,
                 model_name: str | None = None, retrieval: str = "hybrid",
                 reranker=None, min_score: float = 0.0, sandbox=None,
                 index_cache: str | None = None, max_result_chars: int | None = 100_000,
                 observer: Callable[[str, dict], None] | None = None) -> None:
        self.registry = registry or Registry()
        self._embedder = embedder
        self._model_name = model_name
        self._retrieval = retrieval
        self._reranker = reranker
        self._min_score = min_score
        self._sandbox = sandbox  # code-mode backend (default: in-process)
        self._index_cache = index_cache
        self._max_result_chars = max_result_chars
        self._observer = observer
        self._pinned: list[str] = []   # hot tools kept always-visible (no search)
        self._gateway: Gateway | None = None

    # ---------------------------------------------------------- observability
    def _emit(self, event: str, data: dict) -> None:
        _log.debug("%s %s", event, data)
        if self._observer is not None:
            try:
                self._observer(event, data)
            except Exception:   # observers must never break the tool loop
                _log.exception("sift observer raised")

    def _cap(self, text: str) -> str:
        limit = self._max_result_chars
        if limit and len(text) > limit:
            dropped = len(text) - limit
            return (text[:limit]
                    + f'... [truncated {dropped} chars — trim this tool with '
                      f'set_response(returns=[...]) or transform=]')
        return text

    # ----------------------------------------------------------- registration
    def tool(self, path: str, *, description: str, params: dict | None = None,
             returns: list[str] | None = None, risk: bool = False,
             transform: Callable | None = None, examples: list[str] | None = None,
             replace: bool = False) -> Callable:
        """Decorator: register a function as a tool at ``path``.

        ``examples`` are optional "how a user asks for this" phrasings — they are
        indexed for retrieval and improve discovery on ambiguous verbs.
        """
        def deco(fn: Callable[..., dict]) -> Callable[..., dict]:
            self.registry.add(ToolDef(path, description, params or {}, returns or [],
                                      risk, fn, transform, examples or []), replace=replace)
            return fn
        return deco

    def add_tool(self, path: str, fn: Callable[..., dict], *, description: str,
                 params: dict | None = None, returns: list[str] | None = None,
                 risk: bool = False, transform: Callable | None = None,
                 examples: list[str] | None = None, replace: bool = False) -> "Sift":
        self.registry.add(ToolDef(path, description, params or {}, returns or [],
                                  risk, fn, transform, examples or []), replace=replace)
        return self

    def describe(self, node_path: str, description: str) -> "Sift":
        self.registry.describe(node_path, description)
        self._drop_schema_cache()
        return self

    def set_response(self, path: str, *, returns: list[str] | None = None,
                     transform: Callable | None = None) -> "Sift":
        """Trim what a tool returns to the model — a field whitelist and/or a
        reshaping transform. Works on imported (MCP/OpenAPI) tools too."""
        self.registry.set_response(path, returns=returns, transform=transform)
        self._drop_schema_cache()  # a cached TOON line would show the old whitelist
        return self

    def _drop_schema_cache(self) -> None:
        if self._gateway is not None:
            self._gateway.invalidate_schema_cache()

    def pin(self, *paths: str) -> "Sift":
        """Keep these tools ALWAYS visible as first-class function specs, so the
        model calls them directly — no ``search_tools`` round-trip. Meant for a
        few hot, small-schema tools asked often (the "keep your 3–5 most-used
        tools loaded" pattern). Everything else stays discovery-only.

        Pinned tools appear in ``openai_tools``/``anthropic_tools`` named with the
        dotted path's ``.`` → ``__`` (LLM tool-name rules), and ``dispatch``
        routes those names straight to execution.
        """
        for p in paths:
            self.registry.tool(p)  # validate it exists (KeyError otherwise)
            if p not in self._pinned:
                self._pinned.append(p)
        return self

    def scope(self, *, allow: list[str] | None = None, deny: list[str] | None = None,
              allow_risky: bool = True):
        """A scoped view that only sees/runs tools matching the allow/deny globs
        (an `allowedTools` per model/session). Reuses the built index.
        ``allow_risky=False`` additionally blocks every tool flagged ``risk``."""
        from .scope import SiftScope
        return SiftScope(self, allow=allow, deny=deny, allow_risky=allow_risky)

    # ----------------------------------------------------------------- index
    def build_index(self) -> "Sift":
        if self._embedder is None and self._retrieval != "bm25":
            from .embeddings import FastEmbedder
            self._embedder = FastEmbedder(self._model_name)
        self._gateway = Gateway(self.registry, self._embedder, retrieval=self._retrieval,
                                reranker=self._reranker, min_score=self._min_score)
        self._gateway.build_index(cache=self._index_cache)
        return self

    @property
    def gateway(self) -> Gateway:
        if self._gateway is None:
            raise RuntimeError("call build_index() before using the gateway")
        return self._gateway

    # ------------------------------------------------------------ meta-tools
    def search_tools(self, q: str, top_k: int = 5) -> list[SearchResult]:
        return self.gateway.search_tools(q, top_k)

    def search_request(self, domain: str, action: str, top_k: int = 3) -> list[SearchResult]:
        """Active tool request: route a structured intent (``domain`` + ``action``)
        instead of a raw query — sharper alignment, higher accuracy at scale."""
        return self.gateway.search_request(domain, action, top_k)

    def get_tool_schema(self, path: str) -> str:
        return self.gateway.get_tool_schema(path)

    def execute_tool(self, path: str, params: dict | None = None) -> dict:
        return self.gateway.execute_tool(path, params)

    async def aexecute_tool(self, path: str, params: dict | None = None) -> dict:
        """Async execution — required for ``async def`` tools; fine for sync ones."""
        return await self.gateway.aexecute_tool(path, params)

    async def adispatch(self, name: str, arguments: dict | str) -> str:
        """Async twin of ``dispatch``: awaits async tools; ``run_code`` is moved
        to a worker thread (the sandbox may block on a subprocess)."""
        import asyncio
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        try:
            if name == "execute_tool":
                res = await self.aexecute_tool(args["path"], args.get("params") or {})
                self._emit("execute", {"path": args.get("path"), "ok": True, "ms": None})
                return self._cap(json.dumps(res, ensure_ascii=False, default=str))
            if "__" in name:  # pinned/promoted tool called directly by its flat name
                path = name.replace("__", ".")
                res = await self.aexecute_tool(path, args)
                self._emit("execute", {"path": path, "ok": True, "ms": None})
                return self._cap(json.dumps(res, ensure_ascii=False, default=str))
            if name == "run_code":
                return self._cap(await asyncio.to_thread(self.run_code, args["code"]))
        except Exception as exc:
            self._emit("execute", {"path": args.get("path"), "ok": False, "error": str(exc), "ms": None})
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        return self.dispatch(name, args)  # search/browse are cheap and sync-safe

    def session(self, *, max_promoted: int = 10):
        """A per-conversation view that REMEMBERS discovered tools and promotes
        them to real function specs on later turns (no re-searching). See
        :mod:`sift.session`."""
        from .session import SiftSession
        return SiftSession(self, max_promoted=max_promoted)

    def dispatch(self, name: str, arguments: dict | str) -> str:
        """Run a meta-tool call by name; returns a string (TOON or JSON).

        Handy as the single entry point when wiring SIFT into an LLM loop.
        Handles the 2 meta-tools plus ``run_code`` (code mode). Results are
        capped at ``max_result_chars``; errors come back as ``{"error": ...}``.
        """
        import time
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        t0 = time.perf_counter()
        try:
            if name == "search_tools":
                top_k = int(args.get("top_k", 3))
                domain = (args.get("domain") or "").strip()
                action = (args.get("action") or "").strip()
                q = (args.get("q") or "").strip()
                if domain or action:  # active tool request — structured intent
                    out = self.gateway.search_request_compact(domain, action, top_k)
                elif q:               # semantic discovery — TOON with schema inline
                    out = self.gateway.search_compact(q, top_k)
                else:                 # browse a level (falls back to search on a bad guess)
                    out = self.gateway.browse(args.get("path", "") or "", top_k)
                self._emit("search", {"q": q, "domain": domain, "action": action,
                                      "ms": round((time.perf_counter() - t0) * 1000, 1)})
                return out
            if name == "run_code":
                out = self._cap(self.run_code(args["code"]))
                self._emit("run_code", {"ok": '"error"' not in out[:12],
                                        "ms": round((time.perf_counter() - t0) * 1000, 1)})
                return out
            if name == "get_tool_schema":  # deprecated alias — folded into search_tools
                return self.gateway.browse(args.get("path", "") or "", int(args.get("top_k", 3)))
            if name == "execute_tool":
                res = self.execute_tool(args["path"], args.get("params") or {})
                self._emit("execute", {"path": args.get("path"), "ok": True,
                                       "ms": round((time.perf_counter() - t0) * 1000, 1)})
                return self._cap(json.dumps(res, ensure_ascii=False, default=str))
            if "__" in name:  # a pinned/promoted tool called directly by its flat name
                path = name.replace("__", ".")
                res = self.execute_tool(path, args)   # args ARE the params here
                self._emit("execute", {"path": path, "ok": True,
                                       "ms": round((time.perf_counter() - t0) * 1000, 1)})
                return self._cap(json.dumps(res, ensure_ascii=False, default=str))
            return json.dumps({"error": f"unknown meta-tool {name!r}"})
        except Exception as exc:  # surfaced back to the model as a tool result
            self._emit("execute" if name == "execute_tool" else name,
                       {"path": args.get("path"), "ok": False, "error": str(exc),
                        "ms": round((time.perf_counter() - t0) * 1000, 1)})
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    # --------------------------------------------------------------- adapters
    def openai_tools(self) -> list[dict]:
        """OpenAI/OpenRouter function-calling specs: the 2 meta-tools, plus one
        first-class spec per pinned tool (callable without a search)."""
        specs = tool_specs()
        if self._pinned:
            from .session import function_spec
            specs += [function_spec(self.registry.tool(p)) for p in self._pinned]
        return specs

    @property
    def system_prompt(self) -> str:
        if not self._pinned:
            return SYSTEM_PROMPT
        from .session import promoted_name
        names = ", ".join(promoted_name(p) for p in self._pinned)
        return (SYSTEM_PROMPT + "\n\nSome tools are already available to call DIRECTLY "
                f"(no search needed): {names}.")

    # --- code mode (orchestrate many tools in one turn) ---
    def code_tools(self) -> list[dict]:
        from .codemode import code_tool_specs
        return code_tool_specs()

    @property
    def code_system_prompt(self) -> str:
        from .codemode import CODE_SYSTEM_PROMPT
        return CODE_SYSTEM_PROMPT

    def run_code(self, code: str) -> str:
        from .codemode import run_code
        return run_code(self, code, sandbox=self._sandbox)

    @property
    def meta_tool_names(self) -> tuple[str, ...]:
        return META_TOOL_NAMES

    def langchain_tools(self) -> list:
        from .adapters.langchain import langchain_tools
        return langchain_tools(self)

    def anthropic_tools(self) -> list[dict]:
        """The 2 meta-tools in the native Anthropic (Messages API) tool format."""
        from .adapters.anthropic import anthropic_tools
        return anthropic_tools(self)

    # --- for models without native tool calling (prompted / constrained) ---
    def tool_call_schema(self) -> dict:
        """JSON Schema for a prompted step — feed to a structured-output decoder."""
        from .constrain import tool_call_json_schema
        return tool_call_json_schema()

    def json_gbnf(self) -> str:
        """GBNF grammar (llama.cpp) constraining output to valid JSON."""
        from .constrain import json_gbnf
        return json_gbnf()

    def mcp_server(self, name: str = "sift"):
        from .adapters.mcp_server import build_mcp_server
        return build_mcp_server(self, name=name)

    def serve_http(self, *, host: str = "127.0.0.1", port: int = 8000, scope=None) -> None:
        """Run an OpenAPI HTTP server exposing the 2 meta-tools (OpenWebUI tool
        server, REST clients). Requires the ``server`` extra."""
        from .http_server import serve_http
        serve_http(self, host=host, port=port, scope=scope)

    def serve_mcp(self, name: str = "sift", transport: str = "stdio") -> None:
        """Run SIFT as an MCP server exposing the 2 meta-tools.

        transport: "stdio" (default; Claude Desktop, local clients) or "sse" /
        "streamable-http" (remote clients, OpenWebUI). HTTP host/port are taken
        from the MCP server settings / env.
        """
        server = self.mcp_server(name)
        server.run() if transport == "stdio" else server.run(transport=transport)
