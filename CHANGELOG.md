# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semver.

## [0.2.0] — unreleased

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
