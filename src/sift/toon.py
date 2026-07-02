"""TOON (Token-Optimized Object Notation) codec.

Collapses each tool to a single line, cutting most of the JSON-Schema token
overhead. Function line layout::

    path|description|param:type:req[:default]|...|r:f1,f2[|risk]

where ``req`` is ``n`` (required) or ``o`` (optional). Example::

    google_workspace.gmail.read|Read emails|q:string:o:is:unread|m:number:o:10|r:id,subject,from
"""
from __future__ import annotations

from .registry import Registry, ToolDef


def _clean(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ").strip()


def _default_token(default: str) -> str:
    # quote defaults that contain the segment delimiter so they stay unambiguous
    if ":" in default or "|" in default or " " in default:
        return "'" + default.replace("|", "/") + "'"
    return default


def encode_function(tool: ToolDef) -> str:
    parts = [tool.path, _clean(tool.description)]
    for name in sorted(tool.params):
        p = tool.params[name]
        req = "n" if p.required else "o"
        seg = f"{name}:{p.type}:{req}"
        if p.default:
            seg += f":{_default_token(p.default)}"
        parts.append(seg)
    if tool.returns:
        parts.append("r:" + ",".join(tool.returns))
    if tool.risk:
        parts.append("risk")
    return "|".join(parts)


def encode_service(reg: Registry, service_path: str, *, predicate=None) -> str:
    fns = reg.functions(service_path)
    if predicate is not None:
        fns = {name: t for name, t in fns.items() if predicate(t.path)}
    return "\n".join(encode_function(fns[name]) for name in sorted(fns))


def encode_category(reg: Registry, category: str, *, predicate=None) -> str:
    svcs = reg.services(category)
    if predicate is not None:  # keep only services with at least one visible tool
        visible = {t.parts[1] for t in reg.tools()
                   if t.parts[0] == category and predicate(t.path)}
        svcs = {name: desc for name, desc in svcs.items() if name in visible}
    return "\n".join(f"{category}.{name}|{_clean(desc)}" for name, desc in sorted(svcs.items()))


def encode_categories(reg: Registry, *, predicate=None) -> str:
    cats = reg.categories()
    if predicate is not None:  # keep only categories with at least one visible tool
        visible = {t.parts[0] for t in reg.tools() if predicate(t.path)}
        cats = {name: desc for name, desc in cats.items() if name in visible}
    return "\n".join(f"{name}|{_clean(desc)}" for name, desc in sorted(cats.items()))


def estimate_tokens(text: str) -> int:
    """Offline ~4-chars/token approximation, for format comparisons only."""
    return (len(text) + 3) // 4 if text else 0
