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

    ``req`` is ``n`` or ``r`` (required) / ``o`` or empty (optional) — an unknown
    flag raises instead of silently meaning "optional" (a typo like ``x`` used to
    turn a required param optional without a trace).

    Convenient for the common case. Note that in this string form the ``default``
    field cannot contain ``:`` (the description, being last, can). For
    colon-bearing defaults use the structured dict form instead::

        params={"q": {"type": "string", "default": "is:unread", "desc": "query"}}
    """
    parts = compact.split(":", 3)
    parts += [""] * (4 - len(parts))
    req = parts[1].lower()
    if req not in ("n", "r", "o", ""):
        raise ValueError(f"param {name!r}: req flag must be 'n'/'r' (required) or "
                         f"'o' (optional), got {parts[1]!r}")
    return Param(name=name, type=parts[0], required=req in ("n", "r"),
                 default=parts[2], desc=parts[3])


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


_ANNOTATION_TYPES = {int: "integer", float: "number", bool: "boolean",
                     str: "string", list: "array", dict: "object"}


def derive_params(fn) -> dict[str, Param]:
    """Derive a param spec from a function's signature (used when ``params=`` is
    omitted at registration). Without this, ``def add(a, b)`` registered bare
    would be discoverable but fail on EVERY call — only declared params are
    bound. Types come from annotations (``a: int`` → integer), requiredness from
    the absence of a default."""
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):   # builtins / C callables
        return {}
    out: dict[str, Param] = {}
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue   # *args/**kwargs carry no schema
        typ = _ANNOTATION_TYPES.get(p.annotation, "string")
        required = p.default is inspect.Parameter.empty
        default = "" if required or p.default is None else str(p.default)
        out[name] = Param(name=name, type=typ, required=required, default=default, desc="")
    return out


def _fn_schema(tool: "ToolDef") -> dict:
    return {
        "d": tool.description,
        "p": {name: param_dict(p) for name, p in tool.params.items()},
        "r": tool.returns,
        "risk": tool.risk,
    }


_JSON_TYPES = {"string": "string", "number": "number", "integer": "integer", "int": "integer",
               "float": "number", "boolean": "boolean", "bool": "boolean", "array": "array",
               "list": "array", "object": "object", "dict": "object"}


def input_schema_for(tool: "ToolDef") -> dict:
    """A standard JSON Schema for the tool's parameters — what OpenAI-style
    function specs and Anthropic ``input_schema`` expect."""
    props: dict = {}
    required: list[str] = []
    for name, p in tool.params.items():
        prop: dict = {"type": _JSON_TYPES.get(p.type.lower(), "string")}
        if p.desc:
            prop["description"] = p.desc
        if p.default != "":
            prop["default"] = p.default
        props[name] = prop
        if p.required:
            required.append(name)
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


@dataclass
class ToolDef:
    path: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    returns: list[str] = field(default_factory=list)
    risk: bool = False
    fn: Callable[..., dict] | None = None
    transform: Callable[[Any], Any] | None = None  # post-process the raw result
    examples: list[str] = field(default_factory=list)  # "how a user asks" — indexed

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
    text: str      # rich text for the EMBEDDING side (desc + params + examples)
    lex: str = ""  # lean text for the BM25 side (path + desc only) — extra tokens
                   # would only dilute tf/length normalisation for exact matching

    def __post_init__(self) -> None:
        if not self.lex:
            self.lex = self.text


class Registry:
    """Holds tools keyed by dotted path, plus optional node descriptions."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}
        self._node_desc: dict[str, str] = {}  # "cat" or "cat.svc" -> description

    # ------------------------------------------------------------------ build
    def add(self, tool: ToolDef, *, replace: bool = False) -> None:
        if tool.path.count(".") != 2:
            raise ValueError(
                f"tool path must be 'category.service.function', got {tool.path!r}"
            )
        # silent overwrites are a footgun (two imported MCP servers with a
        # same-named tool would shadow each other without a trace)
        if not replace and tool.path in self._tools:
            raise ValueError(
                f"tool {tool.path!r} is already registered — pass replace=True to overwrite"
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
            cat_desc = self._node_desc.get(cat, "")   # only an EXPLICIT category desc
            for svc in sorted(self.services(cat)):
                svc_path = f"{cat}.{svc}"
                svc_desc = self._desc(svc_path)
                # NB: don't append a synthesised category desc here — it duplicates
                # the service's own text (inflating BM25 tf) and leaks sibling
                # services' wording into this entry
                text = f"{cat} {svc}: {svc_desc}"
                if cat_desc:
                    text += f". {cat_desc}"
                entries.append(SearchEntry(svc_path, "service", svc_desc, text))
                for fn_name, tool in sorted(self.functions(svc_path).items()):
                    lex = f"{cat} {svc} {fn_name}: {tool.description}"
                    # the embedding side also gets param names/descriptions and
                    # example phrasings — context dense retrieval can use, but
                    # which would dilute BM25's exact-term matching
                    param_text = " ".join(
                        f"{p.name} {p.desc}".strip() for p in tool.params.values()
                    ).strip()
                    text = lex
                    if param_text:
                        text += f" ({param_text})"
                    if tool.examples:   # "how a user asks" phrasings — strong signal
                        text += " e.g. " + "; ".join(tool.examples)
                    entries.append(SearchEntry(tool.path, "function", tool.description,
                                               text, lex=lex))
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
