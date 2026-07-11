"""Shared helpers for importers: JSON-Schema -> SIFT compact param format."""
from __future__ import annotations

import re

_DESTRUCTIVE = ("delete", "remove", "drop", "send", "purge", "destroy", "revoke")

_KNOWN_TYPES = {"string", "number", "integer", "boolean", "array", "object"}


def _compact_type(json_type) -> str:
    if isinstance(json_type, list):
        json_type = next((t for t in json_type if t != "null"), "string")
    return json_type if json_type in _KNOWN_TYPES else "string"


def _clean(text) -> str:
    return str(text or "").replace("\n", " ").strip()


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_WS = re.compile(r"\s+")


def sanitize_text(text, *, max_len: int = 300) -> str:
    """Neutralise imported third-party text before it enters the index and the
    model's context: strip control characters, collapse whitespace (multi-line
    'instructions' flatten into one visible line), cap the length.

    This is injection HYGIENE, not a guarantee — a malicious MCP server's tool
    descriptions still reach the model as (visible) text. Review what you import;
    see docs/security.md.
    """
    s = _CONTROL.sub("", str(text or ""))
    s = _WS.sub(" ", s).strip()
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def compress_params(input_schema: dict | None) -> dict[str, dict]:
    """Convert a JSON-Schema ``object`` into SIFT's structured param dicts.

    Returns ``{name: {"type", "required", "default", "desc"}}`` — the structured
    form, so defaults containing ``:`` (e.g. ``is:unread``) survive intact.
    """
    schema = input_schema if isinstance(input_schema, dict) else {}
    props = schema.get("properties", {})
    if not isinstance(props, dict):   # real-world MCP schemas are messy
        return {}
    required_raw = schema.get("required", [])
    required = set(required_raw) if isinstance(required_raw, (list, set, tuple)) else set()
    out: dict[str, dict] = {}
    for name, prop in props.items():
        if not isinstance(prop, dict):   # e.g. "properties": {"q": "the query"}
            prop = {"description": str(prop)} if prop else {}
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
