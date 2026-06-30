# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semver.

## [0.1.0] — unreleased

Initial release.

### Core
- Hierarchical tool registry (category → service → function) with TOON schema codec.
- Three meta-tools (`search_tools`, `get_tool_schema`, `execute_tool`); merged
  search+inspect (schema returned inline) so the model executes directly.
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
