"""Subprocess sandbox child entrypoint.

Reads one config line ({"code", "max_lines"}) on stdin, runs the snippet via the
shared ``sandbox.execute`` policy, and proxies each tool call back to the parent
over stdio (the parent holds the real tools). The snippet's own stdout is captured
by ``execute``; only the IPC JSON lines go to the real stdout.
"""
from __future__ import annotations

import json
import sys

from sift.sandbox import execute

_OUT = sys.stdout   # captured before execute() redirects the snippet's stdout
_IN = sys.stdin


def _rpc(op: str, **kw):
    _OUT.write(json.dumps({"op": op, **kw}) + "\n")
    _OUT.flush()
    resp = json.loads(_IN.readline())
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "tool proxy error"))
    return resp["value"]


def _call(_path: str, /, **params):
    return _rpc("call", path=_path, params=params)


def _search(_q: str, /, top_k: int = 5):
    return _rpc("search", q=_q, top_k=top_k)


def _schema(_path: str, /):
    return _rpc("schema", path=_path)


def main() -> None:
    cfg = json.loads(_IN.readline())
    result = execute(cfg["code"], _call, _search, _schema, max_lines=cfg.get("max_lines", 200_000))
    _OUT.write(json.dumps({"op": "done", "result": result}) + "\n")
    _OUT.flush()


if __name__ == "__main__":
    main()
