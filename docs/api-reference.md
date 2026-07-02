# API reference

A condensed reference for the public surface. Source of truth is the docstrings in
[`src/sift`](../src/sift).

## `Sift`

```python
Sift(*, registry=None, embedder=None, model_name=None, retrieval="hybrid",
     reranker=None, min_score=0.0, sandbox=None)
```

| Param | Default | Meaning |
|---|---|---|
| `registry` | new `Registry` | pre-populated registry (e.g. `Registry.from_json`) |
| `embedder` | local fastembed | any object with `embed(texts) -> list[vector]` |
| `model_name` | `bge-small-en-v1.5` | fastembed model (or `SIFT_EMBED_MODEL` env) |
| `retrieval` | `"hybrid"` | `"hybrid"` · `"embedding"` · `"bm25"` |
| `reranker` | `None` | object with `rerank(query, docs) -> list[float]` |
| `min_score` | `0.0` | relevance floor; below it, discovery returns nothing |
| `sandbox` | `InProcessSandbox` | code-mode backend |

### Registration

| Method | Returns | Notes |
|---|---|---|
| `@sift.tool(path, *, description, params=None, returns=None, risk=False, transform=None)` | decorator | register a function as a tool |
| `add_tool(path, fn, *, description, params=None, returns=None, risk=False, transform=None)` | `Sift` | register an existing function (chainable) |
| `describe(node_path, description)` | `Sift` | set a category/service description |
| `set_response(path, *, returns=None, transform=None)` | `Sift` | (re)configure projection on any tool |
| `scope(*, allow=None, deny=None, allow_risky=True)` | `SiftScope` | a scoped view |
| `build_index()` | `Sift` | **build the search index (call once, after registering)** |

### Discovery & execution

| Method | Returns | Notes |
|---|---|---|
| `search_tools(q, top_k=5)` | `list[SearchResult]` | simple query search |
| `search_request(domain, action, top_k=3)` | `list[SearchResult]` | active tool request (two-stage routing) |
| `get_tool_schema(path)` | `str` (TOON) | browse a level (`""` → categories) |
| `execute_tool(path, params=None)` | `dict` | run + project; args coerced to declared types (`integer`/`number`/`boolean`/`array`/`object`) |
| `dispatch(name, arguments)` | `str` | run any meta-tool call; errors returned as `{"error": ...}` |

`dispatch` handles `search_tools` (active request `domain`/`action` → query `q` →
browse `path`), `execute_tool`, `run_code`, and the deprecated `get_tool_schema`
alias.

### Adapters & specs

| Member | Returns | Extra |
|---|---|---|
| `openai_tools()` | `list[dict]` | function-calling specs (2 tools) |
| `system_prompt` | `str` | instruction block |
| `anthropic_tools()` | `list[dict]` | `[anthropic]` |
| `langchain_tools()` | `list[StructuredTool]` | `[langchain]` |
| `meta_tool_names` | `tuple[str, ...]` | `("search_tools", "execute_tool")` |

### Code mode

| Member | Returns | Notes |
|---|---|---|
| `code_tools()` | `list[dict]` | code-mode specs (`search_tools` + `run_code`) |
| `code_system_prompt` | `str` | code-mode instructions |
| `run_code(code)` | `str` (JSON) | execute a snippet in the sandbox |

### Constrained decoding

| Member | Returns |
|---|---|
| `tool_call_schema()` | JSON Schema for one prompted step |
| `json_gbnf()` | GBNF grammar (llama.cpp) |

### Servers

| Method | Notes |
|---|---|
| `mcp_server(name="sift")` | build the MCP server object; `[mcp]` |
| `serve_mcp(name="sift", transport="stdio")` | run it; `transport` = `"stdio"` / `"sse"` |
| `serve_http(*, host="127.0.0.1", port=8000, scope=None)` | OpenAPI HTTP server; `[server]` |

## `SearchResult`

```python
@dataclass
class SearchResult:
    path: str          # full dotted path
    kind: str          # "function" | "service"
    description: str
    score: float
```

## `SiftScope`

Returned by `Sift.scope(...)`. Mirrors the facade — `search_tools`,
`search_request`, `search_compact`, `search_request_compact`, `execute_tool`,
`run_code`, `dispatch`, `openai_tools`, `anthropic_tools`, `langchain_tools`,
`system_prompt`, `code_system_prompt` — with `allow`/`deny`/`allow_risky` enforced
on both discovery and execution. `allowed(path) -> bool` tests a single path. See
[Scoping](scoping.md).

## `Registry`

Holds tools by dotted path. Highlights:

| Member | Notes |
|---|---|
| `Registry.from_json(path)` | load the nested `category → services → fns` JSON |
| `add(ToolDef)` | register (path must have exactly two dots) |
| `bind(path, fn)` | attach an executor to an already-registered tool |
| `set_response(path, *, returns=None, transform=None)` | configure projection |
| `describe(node_path, description)` | category/service description |
| `categories()` / `services(cat)` / `functions(svc_path)` | navigation |
| `tool(path) -> ToolDef` | fetch one (`KeyError` if unknown) |
| `risky_paths() -> set[str]` | all `risk=True` paths |
| `search_entries() -> list[SearchEntry]` | flatten for indexing |
| `schema(path) -> dict` | structured (JSON) view of a level |

`ToolDef(path, description, params, returns, risk, fn, transform)` — params are
normalised to `Param` objects. `Param(name, type, required, default, desc)`.

## Sandbox backends (`sift.sandbox`)

| Class | Purpose |
|---|---|
| `InProcessSandbox(max_lines=200_000)` | fast in-process guard (default) |
| `SubprocessSandbox(timeout=10, max_lines=200_000, cpu_seconds=10, memory_mb=512)` | isolated process + rlimits + watchdog |

Both expose `run(code, call, search, schema) -> str`. See [Code mode](code-mode.md).

## Embedders & rerankers

| Class | Notes |
|---|---|
| `sift.embeddings.FastEmbedder(model_name=None)` | local ONNX embeddings |
| `sift.embeddings.cosine(a, b)` | cosine similarity helper |
| `sift.rerank.FastEmbedReranker(model_name=...)` | local cross-encoder reranker |

Any duck-typed `embed(texts)->list[vector]` / `rerank(query, docs)->list[float]`
object works in their place.

## Importers (`sift.importers`)

| Function | Module | Notes |
|---|---|---|
| `register_openapi(target, spec, *, category, service=None, request=None)` | `openapi` | register (and optionally bind) OpenAPI ops |
| `httpx_request(base_url, client=None)` | `openapi` | an HTTP executor; `[openapi]` |
| `tools_from_openapi(spec, *, category, service=None)` | `openapi` | ToolDefs without registering |
| `register_listing(target, listing, *, category, service, executor=None)` | `mcp` | register from a tool listing |
| `import_mcp_stdio(target, command, args=None, *, ...)` | `mcp` | async register over stdio |
| `connect_mcp_stdio(target, command, args=None, *, category, service)` | `mcp_proxy` | launch + register + bind; returns a proxy |
| `StdioMcpProxy` | `mcp_proxy` | manage a live MCP session yourself |

See [Importing ecosystems](importing.md).
