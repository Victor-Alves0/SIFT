# Architecture

How the pieces fit. SIFT is a small, dependency-light library; the whole flow is
easy to hold in your head.

## Components

```
src/sift/
  registry.py     the hierarchy: tools keyed by category.service.function + navigation
  toon.py         TOON codec — one-line schema encoding
  embeddings.py   local fastembed backend (+ the Embedder protocol, cosine)
  retrieval.py    BM25 + Reciprocal Rank Fusion (the lexical/hybrid half)
  rerank.py       optional cross-encoder reranker
  gateway.py      the 2 meta-tools: search (query/active/browse) + execute + filter
  metatools.py    canonical tool specs + system prompt (shared by every adapter)
  scope.py        per-model allow/deny views (SiftScope)
  codemode.py     run_code: orchestrate many tools in one snippet
  sandbox.py      pluggable code-mode backends (InProcessSandbox, SubprocessSandbox)
  constrain.py    JSON Schema / GBNF for constrained decoders
  http_server.py  OpenAPI HTTP tool server (serve_http)
  __init__.py     the Sift facade that ties it together
  adapters/       openai · anthropic · langchain · mcp_server · prompted
  importers/      openapi · mcp · mcp_proxy (live MCP execution)
  bench.py · agentbench.py · evalsuite.py   evaluation tooling
```

The **`Sift` facade** ([`__init__.py`](../src/sift/__init__.py)) is the only class
most users touch. It owns a `Registry` and, after `build_index()`, a `Gateway`.

## A turn, end to end

```
                      ┌─────────────── your process ───────────────┐
   model              │  Sift facade → Gateway → Registry (tools)   │
     │  search_tools  │        │            │                       │
     ├───────────────▶│  dispatch ──▶ search_request / search_compact
     │   (TOON back)  │        │            │  hybrid retrieval + TOON
     │◀───────────────┤        │            │
     │  execute_tool  │        │            │
     ├───────────────▶│  dispatch ──▶ execute_tool ─▶ your fn ─▶ transform ─▶ returns
     │   (JSON back)  │        │
     │◀───────────────┤
     └────────────────┘
```

1. **Discovery.** The model calls `search_tools`. The gateway runs retrieval and
   returns the top functions rendered as TOON *with schema inline*
   (`search_compact` for a query, `search_request_compact` for a `domain`+`action`
   request). No separate "inspect" round-trip.
2. **Execution.** The model calls `execute_tool(path, params)`. The gateway
   coerces args, calls your function, applies `transform`, then the `returns`
   whitelist, and returns JSON.
3. **`dispatch`** is the seam: it maps a tool-call name+args to the right gateway
   method and always returns a string. Every adapter is a thin wrapper over it.

## The index

`build_index()` calls `registry.search_entries()` to flatten the hierarchy into
`SearchEntry` rows — one per **service** and one per **function**, each with a
rich `text` blob (path terms + descriptions) used for retrieval. Then:

- `retrieval in {hybrid, embedding}` → embed every entry's text (vectors cached).
- `retrieval in {hybrid, bm25}` → build a BM25 index over the same texts.

At query time: embedding cosine and/or BM25 scores are computed and, for
`hybrid`, fused with **Reciprocal Rank Fusion** (no score normalisation needed).
The **active request** path instead computes per-entry relevance for `domain` and
`action` separately and combines them multiplicatively per function (see
[Discovery](discovery.md)). An optional reranker re-scores the shortlist.

## TOON

**TOON** (Token-Optimized Object Notation) is SIFT's schema encoding: one line per
tool, so a search result carries full schemas in a fraction of JSON's tokens.

```
google_workspace.gmail.read|Read emails from the inbox|q:string:o|m:number:o:10|r:id,subject,from,snippet,date
path ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│ description ┄┄┄┄┄┄┄┄┄┄┄┄┄│ param:type:req[:default] │ r:returns[|risk]
```

`req` is `n` (required) or `o` (optional); a trailing `|risk` marks a high-impact
tool. Colon-bearing defaults are quoted (`q:string:o:'is:unread'`). The codec also
renders category and service listings for browse. This inline-schema design is
what lets discovery and inspection merge into one meta-tool.

## Design principles

- **Fixed prompt overhead.** The model always sees 2 tools + a ~200-token prompt,
  independent of catalogue size. Cost scales with *use*, not with *inventory*.
- **Discovery ≠ inventory.** Tools are found by search/navigation, never dumped.
- **The owner controls the surface.** `returns`/`transform` (what a tool exposes),
  `scope` (which tools a model may use), `risk` (what needs confirmation).
- **Provider-agnostic core.** No model calls inside SIFT; `dispatch` is the seam.
- **Local-first.** Embeddings run on-device via fastembed; `bm25` needs no model
  at all.
