"""Per-model / per-session tool scoping (an `allowedTools` for SIFT).

Build the full catalogue once; then hand each model a *scoped view* that only
sees and can run a subset — exactly the OpenWebUI "pick tools for this model"
pattern. The shared embedding index is reused (no rebuild per scope).

    sift = Sift(); ...; sift.build_index()
    view = sift.scope(allow=["google_workspace.gmail.*", "web.search.*"],
                      deny=["*.delete", "*.send"])
    view.dispatch("search_tools", {"q": "read my last email"})   # only allowed tools
    view.execute_tool("crm.contacts.delete", {})                 # PermissionError

Patterns are globs over the dotted path (``fnmatch``): ``"google_workspace.*"``,
``"*.gmail.read"``, exact paths, etc. ``allow=None`` means everything; ``deny``
always wins.
"""
from __future__ import annotations

import json
from fnmatch import fnmatch


class SiftScope:
    def __init__(self, sift, *, allow: list[str] | None = None,
                 deny: list[str] | None = None, allow_risky: bool = True,
                 pin: list[str] | None = None) -> None:
        self._sift = sift
        self._allow = list(allow) if allow is not None else None
        self._deny = list(deny or [])
        self._allow_risky = allow_risky
        self.meta: dict = {}   # free-form integrator metadata (yours to use)
        # per-scope pins: hot tools always-visible for THIS model/session only
        self._pinned: list[str] = []
        for p in pin or []:
            sift.registry.tool(p)          # must exist
            if not self.allowed(p):        # a denied pin is a config error — fail loudly
                raise ValueError(f"pinned tool {p!r} is denied by this scope's allow/deny")
            self._pinned.append(p)

    def allowed(self, path: str) -> bool:
        if not path:
            return False
        if self._allow is not None and not any(fnmatch(path, p) for p in self._allow):
            return False
        if any(fnmatch(path, p) for p in self._deny):
            return False
        if not self._allow_risky and path in self._sift.registry.risky_paths():
            return False
        return True

    # ---- scoped meta-tools ----
    def search_tools(self, q: str, top_k: int = 3):
        return self._sift.gateway.search_tools(q, top_k, predicate=self.allowed)

    def search_compact(self, q: str, top_k: int = 3) -> str:
        return self._sift.gateway.search_compact(q, top_k, predicate=self.allowed)

    def search_request(self, domain: str, action: str, top_k: int = 3):
        return self._sift.gateway.search_request(domain, action, top_k, predicate=self.allowed)

    def search_request_compact(self, domain: str, action: str, top_k: int = 3) -> str:
        return self._sift.gateway.search_request_compact(domain, action, top_k, predicate=self.allowed)

    def get_tool_schema(self, path: str, top_k: int = 3) -> str:
        # scoped browse: denied tools are not even visible (no schema disclosure);
        # a bad guess falls back to a scoped search
        return self._sift.gateway.browse(path, top_k, predicate=self.allowed)

    def execute_tool(self, path: str, params: dict | None = None):
        if not self.allowed(path):
            raise PermissionError(f"tool {path!r} is not allowed in this scope")
        return self._sift.execute_tool(path, params)

    async def aexecute_tool(self, path: str, params: dict | None = None):
        if not self.allowed(path):
            raise PermissionError(f"tool {path!r} is not allowed in this scope")
        return await self._sift.aexecute_tool(path, params)

    async def adispatch(self, name: str, arguments: dict | str) -> str:
        """Async twin of ``dispatch`` — same scoping, async tools awaited."""
        import asyncio

        from . import PATH_HINT, _error_json, _exc_message
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        try:
            if name == "execute_tool":
                path = (args.get("path") or "").strip()
                if not path:
                    return _error_json("execute_tool requires a 'path' argument", PATH_HINT)
                if not self.allowed(path):
                    return _error_json(f"tool {path!r} is not allowed in this scope")
                return await self._sift.adispatch("execute_tool", args)
            if "__" in name:  # pinned/promoted flat name — enforce scope
                path = name.replace("__", ".")
                if not self.allowed(path):
                    return _error_json(f"tool {path!r} is not allowed in this scope")
                return await self._sift.adispatch(name, args)
            if name == "run_code":
                return self._sift._cap(await asyncio.to_thread(self.run_code, args["code"]))
        except Exception as exc:  # noqa: BLE001 - surfaced to the model
            return _error_json(_exc_message(exc),
                               PATH_HINT if isinstance(exc, KeyError) else None)
        return self.dispatch(name, args)  # search/browse are cheap and sync-safe

    def run_code(self, code: str) -> str:
        # call()/search() inside the snippet route through THIS scope, so allow/deny
        # is enforced even in code mode; uses the parent Sift's sandbox backend.
        from .codemode import run_code
        return run_code(self, code, sandbox=self._sift._sandbox)

    def dispatch(self, name: str, arguments: dict | str) -> str:
        from . import PATH_HINT, _error_json, _exc_message
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        try:
            if name == "search_tools":
                top_k = int(args.get("top_k", 3))
                domain = (args.get("domain") or "").strip()
                action = (args.get("action") or "").strip()
                if domain or action:
                    return self.search_request_compact(domain, action, top_k)
                q = (args.get("q") or "").strip()
                if q:
                    return self.search_compact(q, top_k)
                return self.get_tool_schema(args.get("path", "") or "", top_k)  # scoped browse
            if name == "execute_tool":
                path = (args.get("path") or "").strip()
                if not path:   # missing argument ≠ permission problem — don't mislead
                    return _error_json("execute_tool requires a 'path' argument", PATH_HINT)
                if not self.allowed(path):
                    return _error_json(f"tool {path!r} is not allowed in this scope")
                return self._sift.dispatch("execute_tool", args)
            if name == "get_tool_schema":  # deprecated alias — scoped browse too
                return self.get_tool_schema(args.get("path", "") or "", int(args.get("top_k", 3)))
            if name == "run_code":
                return self._sift._cap(self.run_code(args["code"]))  # same cap as Sift.dispatch
            if "__" in name:  # pinned/promoted tool called directly — enforce scope
                path = name.replace("__", ".")
                if not self.allowed(path):
                    return _error_json(f"tool {path!r} is not allowed in this scope")
                return self._sift.dispatch(name, args)
            return json.dumps({"error": f"unknown meta-tool {name!r}"})
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool result
            return _error_json(_exc_message(exc),
                               PATH_HINT if isinstance(exc, KeyError) else None)

    # ---- pass-throughs for wiring into a model ----
    def _pin_visible(self, specs: list[dict], name_key) -> list[dict]:
        """Drop pinned-tool specs the scope denies (their name is the __ path)."""
        out = []
        for spec in specs:
            name = name_key(spec)
            if "__" in name and not self.allowed(name.replace("__", ".")):
                continue
            out.append(spec)
        return out

    def openai_tools(self) -> list[dict]:
        from .session import function_spec
        specs = self._pin_visible(self._sift.openai_tools(),
                                  lambda s: s["function"]["name"])
        have = {s["function"]["name"] for s in specs}
        for p in self._pinned:   # this scope's own pins, on top of the parent's
            spec = function_spec(self._sift.registry.tool(p))
            if spec["function"]["name"] not in have:
                specs.append(spec)
        return specs

    def anthropic_tools(self) -> list[dict]:
        # convert from THIS scope's openai surface so per-scope pins are included
        from .adapters.anthropic import anthropic_tools
        return anthropic_tools(self)

    def code_tools(self) -> list[dict]:
        return self._sift.code_tools()

    @property
    def system_prompt(self) -> str:
        # direct-call hint covers the parent's pins visible here + this scope's own
        visible = [p for p in self._sift._pinned if self.allowed(p)] + self._pinned
        return self._sift._prompt_for(visible)

    @property
    def code_system_prompt(self) -> str:
        return self._sift.code_system_prompt
