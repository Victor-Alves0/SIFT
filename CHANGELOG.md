# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semver.

## [0.3.0] — 2026-07-02

Hardening release: a full type system at the LLM→tool boundary, scoped browsing,
and published benchmarks. Driven by an external code review — every confirmed
bug below was reproduced before fixing.

### Fixed
- **Type coercion no longer breaks int-expecting tools**: `number` keeps integral
  values as `int` (slicing/pagination work), and a dedicated `integer` type is
  supported.
- **Booleans are coerced**: `"false"`/`"0"`/`"no"`/`"off"` → `False` (a string
  `"false"` was truthy before — dangerous on `risk` tools). `array`/`object`
  params parse JSON strings.
- **Explicit `""` is a real value**: only an absent/`None` argument counts as
  missing, so a model can override a non-empty default with an empty string.
- **Test suite no longer fails collection without the `server` extra**
  (`pytest.importorskip("fastapi")`).
- **Code-mode line budget only counts the snippet's own lines**: frames from real
  tool implementations are neither counted against the budget nor traced (a heavy
  but legitimate tool could exhaust the snippet's budget before).
- **TOON schema cache is invalidated** on `set_response`/`describe` — no more
  stale schema lines showing an old `returns` whitelist.

### Security
- **Scoped browsing**: `search_tools(path=...)` on a `SiftScope` now filters what
  it lists — denied tools' schemas are not disclosed, and categories/services with
  no visible tools are omitted (previously browse was unscoped by design; only
  execution was blocked). The deprecated `get_tool_schema` alias is scoped too.
- HTTP server auth uses `secrets.compare_digest` (constant-time comparison).
- Sandbox: `class` definitions now raise a clear policy error (previously a
  cryptic `NameError: __build_class__`).

### Added
- Richer index text: parameter names/descriptions are embedded alongside the tool
  description, improving retrieval.
- `benchmarks/ab_active_request.py` — reproducible raw-query vs active-request
  A/B (top-1 64% → 100% on a collision catalogue); benchmark numbers (SIFT vs
  flat: up to 8.4× cheaper at 250 tools) published in the README.
- Documented the `min_score` scale difference between `search_tools` and
  `search_request`.

## [0.2.0] — 2026-07-01

### Changed
- **Two meta-tools instead of three.** `get_tool_schema` is folded into
  `search_tools` (matches already come back with their schema inline; browse the
  hierarchy via `search_tools(path=...)`). Smaller surface, one fewer decision
  per turn, lower idle cost (~480 → ~430 tokens). `get_tool_schema` stays as a
  back-compat alias in `dispatch` and as a facade/gateway method.

### Added
- **Active tool request** (`search_request(domain, action)` / the `domain` +
  `action` fields on `search_tools`): a structured, model-authored intent that
  aligns better with tool docs than a raw query. Routed in two stages (service on
  `domain`, function on `action`) and fused with MCP-Zero's
  `(s_server·s_tool)·max(s_server, s_tool)` — over SIFT's **hybrid** signals
  (local embeddings + BM25), not dense-only. Enforced through scopes too.
- **Pluggable code-mode sandbox** (`Sift(sandbox=...)`): `InProcessSandbox`
  (default) and `SubprocessSandbox` — isolated process, tool calls proxied to the
  parent, wall-clock watchdog, and CPU/memory rlimits (Unix).

### Fixed
- LangChain adapter now exposes the 2-tool surface (was still exporting the
  removed `get_tool_schema` tool) and its `search_tools` supports query, browse,
  and the active request.

## [0.1.0] — 2026

Initial release (published to PyPI as `sift-tools`).

### Core
- Hierarchical tool registry (category → service → function) with TOON schema codec.
- Meta-tools with merged search+inspect (schema returned inline) so the model
  executes directly.
- Hybrid retrieval (embeddings + BM25 + RRF), optional cross-encoder reranker,
  relevance floor (`min_score`) with an explicit "no matching tools" reply.
- Response projection: per-tool field whitelist (`returns`) and/or `transform`,
  configurable on imported tools too.
- Per-model scoping (`sift.scope(allow=, deny=, allow_risky=)`) — an `allowedTools`.
- Code mode (`run_code`) to orchestrate many tools in one turn, in a hardened
  in-process sandbox (AST policy + line budget); scope-aware.

### Integrations
- Adapters: OpenAI-compatible, native Anthropic, LangChain, MCP server, and a
  prompted (text) adapter for models without native tool calling.
- Constrained-decoding helpers (`tool_call_schema`, `json_gbnf`).
- Importers: OpenAPI and MCP (with a live `StdioMcpProxy` executor).
- Servers: MCP (`serve_mcp`, stdio/SSE) and OpenAPI HTTP (`serve_http`) + Docker.

### Tooling
- Evaluation: filter-level metrics, token report, BFCL-style accuracy, and a
  SIFT-vs-flat agent benchmark.
- CI (lint + tests on 3.10–3.12) and PyPI trusted-publishing workflow.
