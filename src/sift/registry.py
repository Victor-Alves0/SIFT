"""Hierarchical tool registry.

The hierarchy has three levels encoded as a dotted path:

    category.service.function     e.g. "google_workspace.gmail.read"

Each function carries a compact schema:

    params: {name: "<type>:<req>:<default>:<description>"}
        req = "n" (required) or "o" (optional)
    returns: response whitelist — only these fields survive execute_tool
    risk:    high-impact action (send, delete) flag

Categories and services may have optional human descriptions; when missing they
are synthesised from their children so search still works.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass
class Param:
    name: str
    type: str
    required: bool
    default: str
    desc: str


def parse_param(name: str, compact: str) -> Param:
    """Decode the compact form ``"<type>:<req>:<default>:<desc>"`` into a Param.

    Convenient for the common case. Note that in this string form the ``default``
    field cannot contain ``:`` (the description, being last, can). For
    colon-bearing defaults use the structured dict form instead::

        params={"q": {"type": "string", "default": "is:unread", "desc": "query"}}
    """
    parts = compact.split(":", 3)
    parts += [""] * (4 - len(parts))
    return Param(name=name, type=parts[0], required=parts[1] == "n", default=parts[2], desc=parts[3])


def to_param(name: str, spec: Any) -> Param:
    """Normalise a param spec (compact string, dict, or Param) into a Param.

    Accepted forms::

        "string:n::search query"                       # compact
        {"type": "string", "required": True, "desc": "search query"}
        {"type": "string", "default": "is:unread", "desc": "query"}  # ':' ok here
    """
    if isinstance(spec, Param):
        return spec
    if isinstance(spec, str):
        return parse_param(name, spec)
    if isinstance(spec, dict):
        return Param(
            name=name,
            type=str(spec.get("type", "string")),
            required=bool(spec.get("required", False)),
            default="" if spec.get("default") is None else str(spec.get("default")),
            desc=str(spec.get("desc", spec.get("description", ""))),
        )
    raise TypeError(f"param {name!r}: unsupported spec {type(spec).__name__}")


def param_dict(p: Param) -> dict:
    """Serialisable view of a Param (for adapters / JSON schema)."""
    return {"type": p.type, "required": p.required, "default": p.default, "desc": p.desc}


def _fn_schema(tool: "ToolDef") -> dict:
    return {
        "d": tool.description,
        "p": {name: param_dict(p) for name, p in tool.params.items()},
        "r": tool.returns,
        "risk": tool.risk,
    }


@dataclass
class ToolDef:
    path: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    returns: list[str] = field(default_factory=list)
    risk: bool = False
    fn: Callable[..., dict] | None = None
    transform: Callable[[Any], Any] | None = None  # post-process the raw result

    def __post_init__(self) -> None:
        # store params canonically as Param objects regardless of input form
        self.params = {name: to_param(name, spec) for name, spec in self.params.items()}

    @property
    def parts(self) -> tuple[str, str, str]:
        c, s, f = self.path.split(".", 2)
        return c, s, f


@dataclass
class SearchEntry:
    path: str
    kind: str  # "service" | "function"
    description: str
    text: str  # rich text used to build the embedding


class Registry:
    """Holds tools keyed by dotted path, plus optional node descriptions."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}
        self._node_desc: dict[str, str] = {}  # "cat" or "cat.svc" -> description

    # ------------------------------------------------------------------ build
    def add(self, tool: ToolDef) -> None:
        if tool.path.count(".") != 2:
            raise ValueError(
                f"tool path must be 'category.service.function', got {tool.path!r}"
            )
        self._tools[tool.path] = tool

    def describe(self, node_path: str, description: str) -> None:
        """Attach a description to a category ('cat') or service ('cat.svc')."""
        self._node_desc[node_path] = description

    def bind(self, path: str, fn: Callable[..., dict]) -> None:
        """Attach an executor callable to an already-registered tool."""
        if path not in self._tools:
            raise KeyError(f"unknown tool {path!r}")
        self._tools[path].fn = fn

    def set_response(self, path: str, *, returns: list[str] | None = None,
                     transform: Callable[[Any], Any] | None = None) -> None:
        """Configure how a tool's result is trimmed before it reaches the model.

        ``returns`` is a top-level field whitelist; ``transform`` is a callable
        that reshapes the raw result (e.g. extract just message ids from a verbose
        MCP response). Both can be set on imported tools too — fewer tokens/call.
        """
        tool = self.tool(path)
        if returns is not None:
            tool.returns = returns
        if transform is not None:
            tool.transform = transform

    # ------------------------------------------------------------------ load
    @classmethod
    def from_json(cls, path: str | Path) -> "Registry":
        """Load the nested JSON format (same shape as the Go reference)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        reg = cls()
        for cat_name, cat in data.items():
            if "d" in cat:
                reg.describe(cat_name, cat["d"])
            for svc_name, svc in cat.get("services", {}).items():
                if "d" in svc:
                    reg.describe(f"{cat_name}.{svc_name}", svc["d"])
                for fn_name, fn in svc.get("fns", {}).items():
                    reg.add(
                        ToolDef(
                            path=f"{cat_name}.{svc_name}.{fn_name}",
                            description=fn.get("d", ""),
                            params=fn.get("p", {}) or {},
                            returns=fn.get("r", []) or [],
                            risk=bool(fn.get("risk", False)),
                        )
                    )
        return reg

    # ------------------------------------------------------------- navigation
    def tool(self, path: str) -> ToolDef:
        if path not in self._tools:
            raise KeyError(f"unknown tool {path!r}")
        return self._tools[path]

    def tools(self) -> Iterable[ToolDef]:
        return self._tools.values()

    def categories(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for t in self._tools.values():
            c = t.parts[0]
            out.setdefault(c, self._desc(c))
        return out

    def services(self, category: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for t in self._tools.values():
            c, s, _ = t.parts
            if c == category:
                out.setdefault(s, self._desc(f"{c}.{s}"))
        if not out:
            raise KeyError(f"unknown category {category!r}")
        return out

    def functions(self, service_path: str) -> dict[str, ToolDef]:
        out: dict[str, ToolDef] = {}
        for t in self._tools.values():
            c, s, f = t.parts
            if f"{c}.{s}" == service_path:
                out[f] = t
        if not out:
            raise KeyError(f"unknown service {service_path!r}")
        return out

    def risky_paths(self) -> set[str]:
        return {p for p, t in self._tools.items() if t.risk}

    # --------------------------------------------------------------- indexing
    def search_entries(self) -> list[SearchEntry]:
        """Flatten to searchable nodes (services and functions), stable order."""
        entries: list[SearchEntry] = []
        for cat in sorted(self.categories()):
            cat_desc = self._desc(cat)
            for svc in sorted(self.services(cat)):
                svc_path = f"{cat}.{svc}"
                svc_desc = self._desc(svc_path)
                entries.append(
                    SearchEntry(svc_path, "service", svc_desc, f"{cat} {svc}: {svc_desc}. {cat_desc}")
                )
                for fn_name, tool in sorted(self.functions(svc_path).items()):
                    entries.append(
                        SearchEntry(
                            tool.path,
                            "function",
                            tool.description,
                            f"{cat} {svc} {fn_name}: {tool.description}",
                        )
                    )
        return entries

    # ----------------------------------------------------------------- schema
    def schema(self, path: str) -> dict:
        """Structured (JSON-friendly) view of a level — used by adapters/HTTP."""
        path = path.strip(". ")
        if path == "":
            return {"level": "categories", "items": self.categories()}
        depth = path.count(".")
        if depth == 0:
            return {"level": "services", "path": path, "d": self._desc(path),
                    "services": self.services(path)}
        if depth == 1:
            fns = {name: _fn_schema(t) for name, t in self.functions(path).items()}
            return {"level": "functions", "path": path, "d": self._desc(path), "fns": fns}
        return {"level": "function", "path": path, "schema": _fn_schema(self.tool(path))}

    # ----------------------------------------------------------------- helpers
    def _desc(self, node_path: str) -> str:
        if node_path in self._node_desc:
            return self._node_desc[node_path]
        # synthesise from children when not explicitly described
        if node_path.count(".") == 0:  # category
            kids = [t.description for t in self._tools.values() if t.parts[0] == node_path]
        else:  # service
            kids = [t.description for t in self._tools.values()
                    if f"{t.parts[0]}.{t.parts[1]}" == node_path]
        return "; ".join(dict.fromkeys(kids))[:200]
