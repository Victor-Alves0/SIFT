# API reference

A condensed reference for the public surface. Source of truth is the docstrings in
[`src/sift`](../src/sift).

## `Sift`

```python
Sift(*, registry=None, embedder=None, model_name=None, retrieval="hybrid",
     reranker=None, min_score=0.0, sandbox=None, index_cache=None,
     max_result_chars=100_000, observer=None, on_risky=None)
```

| Param | Default | Meaning |
|---|---|---|
| `registry` | new `Registry` | pre-populated registry (e.g. `Registry.from_json`) |
| `embedder` | local fastembed | `embed(texts) -> list[vector]`; optional `embed_query` for asymmetric models |
| `model_name` | `bge-small-en-v1.5` | fastembed model (or `SIFT_EMBED_MODEL` env) |
| `retrieval` | `"hybrid"` | `"hybrid"` · `"embedding"` · `"bm25"` |
| `reranker` | `None` | object with `rerank(query, docs) -> list[float]` |
| `min_score` | `0.0` | relevance floor — below it, discovery says "no tool fits". `0.0` = always return the top-k. Calibrate with `quality.suggest_min_score()` |
| `sandbox` | `InProcessSandbox` | code-mode backend |
| `index_cache` | `None` | file path persisting vectors across restarts (auto-invalidated) |
| `max_result_chars` | `100_000` | cap on results sent to the model (`None` disables) |
| `observer` | `None` | `callable(event, data)` — search/execute/run_code telemetry |
| `on_risky` | `None` | `callable(path, args) -> bool` — confirm hook before any `risk=True` execution |
| `on_result` | `None` | `callable(path, result) -> result` — global post-filter (e.g. injection scrub) |

### Registration

| Method | Returns | Notes |
|---|---|---|
| `@sift.tool(path, *, description, params=None, returns=None, risk=False, transform=None, examples=None, replace=False, cacheable=False, cache_ttl=60, timeout=None)` | decorator | register a function as a tool |
| `add_tool(path, fn, *, description, params=None, returns=None, risk=False, transform=None)` | `Sift` | register an existing function (chainable) |
| `describe(node_path, description)` | `Sift` | set a category/service description |
| `set_response(path, *, returns=None, transform=None)` | `Sift` | (re)configure projection on any tool |
| `scope(*, allow=None, deny=None, allow_risky=True, pin=None)` | `SiftScope` | a scoped view (with per-scope pins) |
| `pin(*paths)` | `Sift` | keep hot tools always-visible as first-class specs (no search round-trip) |
| `build_index()` | `Sift` | **build the search index (call once, after registering)** |

### Discovery & execution

| Method | Returns | Notes |
|---|---|---|
| `search_tools(q, top_k=5)` | `list[SearchResult]` | simple query search |
| `search_request(domain, action, top_k=3)` | `list[SearchResult]` | active tool request (two-stage routing) |
| `get_tool_schema(path)` | `str` (TOON) | browse a level (`""` → categories) |
| `execute_tool(path, params=None)` | `dict` | run + project; args coerced to declared types (`integer`/`number`/`boolean`/`array`/`object`) |
| `aexecute_tool(path, params=None)` | `dict` (async) | awaits `async def` tools natively |
| `dispatch(name, arguments)` | `str` | run any meta-tool call; errors returned as `{"error": ...}`; capped |
| `adispatch(name, arguments)` | `str` (async) | async twin; `run_code` runs on a worker thread |
| `session(max_promoted=10)` | `SiftSession` | per-conversation discovered-tool memory |

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

## `SiftSession` (`sift.session`)

Per-conversation tool memory (see [Providers → Session memory](providers.md#session-memory-any-provider)).
`tools()` returns meta-tools + promoted specs; `dispatch` records discoveries and
routes promoted names (`path` with `.` → `__`) straight to execution;
`discovered` lists remembered paths. Wraps a `Sift` or a `SiftScope`.

## Anthropic tool search adapter (`sift.adapters.anthropic`)

| Function | Purpose |
|---|---|
| `deferred_tools(sift, *, keep=())` | catalogue as `defer_loading` tools + SIFT's search tool |
| `tool_search_result(sift, tool_use_id, args, *, top_k=5)` | `tool_result` with `tool_reference` blocks |
| `run_agent_deferred(sift, client, model, message, *, keep=())` | full loop over the deferred catalogue |
| `run_agent(sift, client, model, message)` | classic 2-meta-tool loop |

`sift.adapters.openai` additionally provides `run_agent_responses` (Responses API);
`sift.adapters.gemini` provides `gemini_tools` / `run_agent` (native Gemini SDK).

## Quality toolkit (`sift.quality`)

| Function | Purpose |
|---|---|
| `lint(sift, *, dup_threshold=0.92, ...)` | static catalogue checks → `LintReport` (errors/warnings/format) |
| `selftest(sift, *, top_k=5)` | each tool findable by its own description/examples → failures |
| `GapTracker()` | observer: `gaps()` (searches matching nothing) + `suggest_pins()` |

## Observability (`sift.otel`)

`otel_observer(tracer=None)` builds an observer that emits an OpenTelemetry span
per `search`/`execute`/`run_code` event (`[otel]` extra).

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
`aexecute_tool`, `run_code`, `dispatch`, `adispatch`, `openai_tools`,
`anthropic_tools`, `langchain_tools`, `system_prompt`, `code_system_prompt` —
with `allow`/`deny`/`allow_risky` enforced on both discovery and execution.
Per-scope `pin=` adds always-visible tools for this view only; `meta` is a public
dict for integrator metadata; `allowed(path) -> bool` tests a single path. See
[Scoping](scoping.md).

## `Registry`

Holds tools by dotted path. Highlights:

| Member | Notes |
|---|---|
| `Registry.from_json(path)` | load the nested `category → services → fns` JSON |
| `add(ToolDef, *, replace=False)` | register (two-dot path; duplicate raises unless `replace=True`) |
| `input_schema_for(tool)` | JSON Schema for a tool's params (module function) |
| `derive_params(fn)` | param spec from a function signature (module function; used when `params=` is omitted) |
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
