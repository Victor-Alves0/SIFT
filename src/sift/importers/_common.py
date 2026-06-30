"""Shared helpers for importers: JSON-Schema -> SIFT compact param format."""
from __future__ import annotations

_DESTRUCTIVE = ("delete", "remove", "drop", "send", "purge", "destroy", "revoke")


def _compact_type(json_type) -> str:
    if isinstance(json_type, list):
        json_type = next((t for t in json_type if t != "null"), "string")
    return "number" if json_type in ("number", "integer") else "string"


def _clean(text) -> str:
    return str(text or "").replace("\n", " ").strip()


def compress_params(input_schema: dict | None) -> dict[str, dict]:
    """Convert a JSON-Schema ``object`` into SIFT's structured param dicts.

    Returns ``{name: {"type", "required", "default", "desc"}}`` — the structured
    form, so defaults containing ``:`` (e.g. ``is:unread``) survive intact.
    """
    schema = input_schema or {}
    props = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    out: dict[str, dict] = {}
    for name, prop in props.items():
        prop = prop or {}
        out[name] = {
            "type": _compact_type(prop.get("type", "string")),
            "required": name in required,
            "default": "" if prop.get("default") is None else str(prop.get("default")),
            "desc": _clean(prop.get("description", "")),
        }
    return out


def looks_destructive(name: str) -> bool:
    n = name.lower()
    return any(v in n for v in _DESTRUCTIVE)
