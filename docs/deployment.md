# Deployment — run SIFT as a server

Run SIFT as a standalone server so a hub (OpenWebUI, an IDE, Claude Desktop, a
REST client) connects to *it*, and you wire your tools/MCPs/OpenAPI into SIFT —
one hub for everything. The client only ever sees the 2 meta-tools.

Two server flavours: **MCP** and **OpenAPI HTTP**.

## MCP server

Expose SIFT as a Model Context Protocol server. Any MCP client then discovers your
whole catalogue through `search_tools` / `execute_tool`.

```python
sift.serve_mcp()                       # stdio (Claude Desktop, local clients)
sift.serve_mcp(transport="sse")        # HTTP/SSE (remote clients, OpenWebUI)
```

Or from the example script:

```bash
python examples/serve_mcp.py           # stdio
python examples/serve_mcp.py sse       # HTTP/SSE
```

Requires the `[mcp]` extra. Build your catalogue (define `@sift.tool`s and/or run
importers) and `build_index()` **before** calling `serve_mcp`.

## OpenAPI HTTP server

Tool hubs like OpenWebUI consume an OpenAPI tool server. `serve_http` runs a
FastAPI app exposing `POST /search_tools` and `POST /execute_tool`, with
`/openapi.json` and interactive docs at `/docs`.

```python
sift.serve_http(host="0.0.0.0", port=8000)
```

```bash
python examples/serve_http.py          # OpenAPI at /openapi.json, docs at /docs
```

Requires the `[server]` extra (`fastapi` + `uvicorn`).

### Authentication

Set `SIFT_API_KEY` in the environment to require a bearer token on every request:

```bash
export SIFT_API_KEY=secret
# clients must send:  Authorization: Bearer secret
```

Unauthenticated requests get `401`. `/health` stays open as a liveness probe.

### Scoped servers

Pass a [scope](scoping.md) so a given server exposes only a subset — e.g. a
read-only endpoint:

```python
readonly = sift.scope(deny=["*.delete", "*.send"])
sift.serve_http(scope=readonly, port=8001)
```

You can run several servers off one `Sift`, each with a different scope.

## Docker

The repo ships a `Dockerfile` for the OpenAPI server:

```bash
docker build -t sift-server .
docker run -p 8000:8000 -e SIFT_API_KEY=secret sift-server
```

Customise `examples/serve_http.py` with your own `@sift.tool`s and importers, then
rebuild. (For fully untrusted [code mode](code-mode.md), the container is also a
good place to add the OS-level isolation that the subprocess sandbox doesn't
provide on its own.)

## Connecting OpenWebUI

- **OpenAPI:** Tools → *OpenAPI tool server* → point it at your server URL.
- **MCP:** use OpenWebUI's MCP support, or bridge the stdio server via `mcpo`.

Either way the model sees just the 2 meta-tools and discovers the catalogue
through them — so adding tools server-side needs no client change.
