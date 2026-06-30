"""OpenAPI HTTP server — expose SIFT's 3 meta-tools over REST.

Tool hubs like OpenWebUI consume an OpenAPI tool server: point them at this app's
URL and they turn each operation into a tool the model can call. So the model
gets exactly search_tools / get_tool_schema / execute_tool, and discovers your
whole catalogue through them.

    from sift import Sift
    sift = Sift(); ...; sift.build_index()
    sift.serve_http(host="0.0.0.0", port=8000)     # OpenAPI at /openapi.json, docs at /docs

Optional bearer auth: set env ``SIFT_API_KEY`` and send ``Authorization: Bearer <key>``.
Pass a ``scope`` (from ``sift.scope(...)``) to expose only a subset of tools.
Requires the server extra:  pip install "sift-tools[server]"
"""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from . import __version__


class SearchReq(BaseModel):
    q: str
    top_k: int = 3


class SchemaReq(BaseModel):
    path: str = ""


class ExecReq(BaseModel):
    path: str
    params: dict | None = None


class Result(BaseModel):
    result: str


def build_app(sift, *, scope=None, title: str = "SIFT Tool Server") -> FastAPI:
    """Build a FastAPI app exposing the 3 meta-tools. Returns the ASGI app."""
    target = scope if scope is not None else sift
    api_key = os.getenv("SIFT_API_KEY")

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    app = FastAPI(
        title=title,
        version=__version__,
        description=("Hierarchical, search-first tool discovery. Use search_tools to find a "
                     "tool (its schema comes inline), then execute_tool with the path."),
    )

    @app.get("/health", summary="Liveness probe")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/search_tools", response_model=Result, dependencies=[Depends(require_auth)],
              summary="Find tools by need; returns the best matches WITH their schema inline.")
    def search_tools(req: SearchReq) -> Result:
        return Result(result=target.dispatch("search_tools", {"q": req.q, "top_k": req.top_k}))

    @app.post("/get_tool_schema", response_model=Result, dependencies=[Depends(require_auth)],
              summary="Browse the hierarchy (empty path lists categories).")
    def get_tool_schema(req: SchemaReq) -> Result:
        return Result(result=target.dispatch("get_tool_schema", {"path": req.path}))

    @app.post("/execute_tool", response_model=Result, dependencies=[Depends(require_auth)],
              summary="Execute a tool by full path; returns the filtered result.")
    def execute_tool(req: ExecReq) -> Result:
        return Result(result=target.dispatch("execute_tool",
                                             {"path": req.path, "params": req.params or {}}))

    return app


def serve_http(sift, *, host: str = "127.0.0.1", port: int = 8000, scope=None) -> None:
    import uvicorn

    uvicorn.run(build_app(sift, scope=scope), host=host, port=port)
