# Importing existing ecosystems

You don't have to hand-write every tool. Importers turn **OpenAPI specs** and
**MCP servers** into hierarchy nodes, so an existing ecosystem becomes searchable
through SIFT's meta-tools. Each imported operation/tool becomes a
`category.service.function` node.

Two levels of import:

- **register** the tools for *discovery* (searchable, schema visible);
- **bind an executor** so they also *run* through `execute_tool`.

## OpenAPI

```python
from sift.importers.openapi import register_openapi, httpx_request

# discovery only — populate the hierarchy from a spec dict
register_openapi(sift, spec, category="acme")

# runnable — bind an HTTP executor so execute_tool actually calls the API
register_openapi(sift, spec, category="acme",
                 request=httpx_request("https://api.acme.com"))
sift.build_index()
```

`httpx_request(base_url, client=None)` returns a `request(method, route, params)`
callable that performs the HTTP call (requires the `[openapi]` extra). Supply your
own callable to add auth headers, retries, etc. `tools_from_openapi(spec, ...)`
returns the `ToolDef`s without registering, if you want to inspect first.

## MCP servers

### From a static listing

If you already have a tool listing (e.g. from an MCP `list_tools` response):

```python
from sift.importers.mcp import register_listing

register_listing(sift, listing, category="integrations", service="github",
                 executor=lambda name, params: my_mcp_proxy(name, params))
```

`tools_from_listing(listing, ...)` gives the `ToolDef`s without registering.

### Live MCP over stdio (register + bind in one call)

The cleanest path for a real MCP server: `connect_mcp_stdio` launches the server,
registers its tools, **and** binds execution (keeping the session open) in one go:

```python
from sift.importers import connect_mcp_stdio

proxy = connect_mcp_stdio(
    sift, "npx", ["-y", "@modelcontextprotocol/server-github"],
    category="integrations", service="github")
sift.build_index()

# ... imported GitHub MCP tools now discover AND run out of the box ...

proxy.close()   # shut the server down when finished
```

There's also `import_mcp_stdio(...)` (async) for registration inside an existing
event loop, and `StdioMcpProxy` if you want to manage the connection yourself.

## Make imported tools cheap

Imported tools often return verbose payloads. Trim them **after import** with
`set_response` — the same projection you'd put on a native tool:

```python
sift.set_response("integrations.github.list_issues",
                  transform=lambda r: {"issues": [i["title"] for i in r["items"]]},
                  returns=["issues"])
```

This is the main lever for keeping a large imported catalogue token-efficient. See
[Executing & filtering](executing-and-filtering.md#response-projection).

## Import + scope + deploy

A common shape: import several ecosystems into one `Sift`, then expose scoped
views per model or per server — one hub for everything.

```python
register_openapi(sift, acme_spec, category="acme", request=httpx_request(ACME_URL))
connect_mcp_stdio(sift, "npx", ["-y", "@modelcontextprotocol/server-github"],
                  category="integrations", service="github")
sift.build_index()

readonly = sift.scope(deny=["*.delete", "*.send", "*.create"])
sift.serve_http(scope=readonly)     # see Deployment
```
