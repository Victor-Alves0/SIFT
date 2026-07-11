"""Import an OpenAPI 3.x spec into the SIFT hierarchy.

Each operation (path + method) becomes a function. The service defaults to the
operation's first tag, so a single API fans out into a tidy sub-tree.
"""
from __future__ import annotations

import re
from typing import Callable, Iterator

from ..registry import Registry, ToolDef
from ._common import _compact_type

_HTTP_METHODS = ("get", "post", "put", "delete", "patch")
_WRITE_METHODS = ("post", "put", "delete", "patch")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "op"


def _params_from_operation(op: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in op.get("parameters", []) or []:
        name = p.get("name")
        if not name:
            continue
        schema = p.get("schema", {}) or {}
        from ._common import sanitize_text
        out[name] = {
            "type": _compact_type(schema.get("type", "string")),
            "required": bool(p.get("required")),
            "default": "" if schema.get("default") is None else str(schema.get("default")),
            "desc": sanitize_text(p.get("description", ""), max_len=150),
        }
    if "requestBody" in op:
        out["body"] = {
            "type": "string",
            "required": bool(op["requestBody"].get("required")),
            "default": "",
            "desc": "JSON request body",
        }
    return out


def _iter_operations(spec: dict, category: str, service: str | None) -> Iterator[tuple[ToolDef, str, str]]:
    for route, methods in (spec.get("paths", {}) or {}).items():
        for method, op in (methods or {}).items():
            if method.lower() not in _HTTP_METHODS or not isinstance(op, dict):
                continue
            tags = op.get("tags") or []
            svc = service or (_slug(tags[0]) if tags else "api")
            op_id = op.get("operationId") or _slug(f"{method}_{route}")
            from ._common import sanitize_text
            desc = sanitize_text(op.get("summary") or op.get("description") or op_id,
                                 max_len=200)
            td = ToolDef(
                path=f"{category}.{svc}.{_slug(op_id)}",
                description=desc,
                params=_params_from_operation(op),
                returns=[],
                risk=method.lower() in _WRITE_METHODS,
            )
            yield td, method.upper(), route


def tools_from_openapi(spec: dict, *, category: str, service: str | None = None) -> list[ToolDef]:
    """Convert an OpenAPI spec dict into ToolDefs (discovery only)."""
    return [td for td, _, _ in _iter_operations(spec, category, service)]


def _registry_of(target) -> Registry:
    return target.registry if hasattr(target, "registry") else target


def _bind_request(request: Callable[[str, str, dict], dict], method: str, route: str):
    def _call(**kwargs) -> dict:
        return request(method, route, kwargs)
    return _call


def register_openapi(target, spec: dict, *, category: str, service: str | None = None,
                     request: Callable[[str, str, dict], dict] | None = None) -> int:
    """Register every operation of an OpenAPI spec. Returns count added.

    Pass ``request(method, route, params) -> dict`` to make the imported
    operations runnable (see :func:`httpx_request` for a ready-made HTTP one).
    """
    reg = _registry_of(target)
    n = 0
    for td, method, route in _iter_operations(spec, category, service):
        if request is not None:
            td.fn = _bind_request(request, method, route)
        reg.add(td)
        n += 1
    return n


def httpx_request(base_url: str, client=None) -> Callable[[str, str, dict], dict]:
    """A ready-made ``request`` executor that calls a live HTTP API via httpx.

    Path params ({id}) are substituted from params; the rest become query string
    (GET) or JSON body (writes). Requires the ``openapi`` extra (httpx).
    """
    import httpx

    cl = client or httpx.Client(timeout=30)

    def _request(method: str, route: str, params: dict) -> dict:
        params = dict(params)
        body = params.pop("body", None)
        path = route
        for key in list(params):
            token = "{" + key + "}"
            if token in path:
                path = path.replace(token, str(params.pop(key)))
        url = base_url.rstrip("/") + path
        if method == "GET":
            resp = cl.request(method, url, params=params)
        else:
            resp = cl.request(method, url, json=body if body is not None else params)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"status": resp.status_code, "text": resp.text}

    return _request
