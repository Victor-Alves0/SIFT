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
                 deny: list[str] | None = None, allow_risky: bool = True) -> None:
        self._sift = sift
        self._allow = list(allow) if allow is not None else None
        self._deny = list(deny or [])
        self._allow_risky = allow_risky

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

    def get_tool_schema(self, path: str) -> str:
        # scoped browse: denied tools are not even visible (no schema disclosure)
        return self._sift.gateway.get_tool_schema(path, predicate=self.allowed)

    def execute_tool(self, path: str, params: dict | None = None):
        if not self.allowed(path):
            raise PermissionError(f"tool {path!r} is not allowed in this scope")
        return self._sift.execute_tool(path, params)

    def run_code(self, code: str) -> str:
        # call()/search() inside the snippet route through THIS scope, so allow/deny
        # is enforced even in code mode; uses the parent Sift's sandbox backend.
        from .codemode import run_code
        return run_code(self, code, sandbox=self._sift._sandbox)

    def dispatch(self, name: str, arguments: dict | str) -> str:
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
                return self.get_tool_schema(args.get("path", "") or "")  # scoped browse
            if name == "execute_tool":
                if not self.allowed(args.get("path", "")):
                    return json.dumps({"error": f"tool {args.get('path')!r} is not allowed in this scope"})
                return self._sift.dispatch("execute_tool", args)
            if name == "get_tool_schema":  # deprecated alias — scoped browse too
                return self.get_tool_schema(args.get("path", "") or "")
            if name == "run_code":
                return self.run_code(args["code"])
            return json.dumps({"error": f"unknown meta-tool {name!r}"})
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool result
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    # ---- pass-throughs for wiring into a model ----
    def openai_tools(self) -> list[dict]:
        return self._sift.openai_tools()

    def anthropic_tools(self) -> list[dict]:
        return self._sift.anthropic_tools()

    def code_tools(self) -> list[dict]:
        return self._sift.code_tools()

    @property
    def system_prompt(self) -> str:
        return self._sift.system_prompt

    @property
    def code_system_prompt(self) -> str:
        return self._sift.code_system_prompt
