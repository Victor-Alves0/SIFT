# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semver.

## [0.5.0] — 2026-07-04

Two safe, opt-in latency optimizations for the "cheap even with few tools"
case — where the cost is a heavy schema and the round-trips, not the tool count.
Neither touches the quality path (a tool with required params still gets its
schema before the model fills them).

### Added
- **Pinned tools** (`sift.pin("utils.time.now", ...)`): keep a few hot,
  small-schema tools always visible as first-class function specs, so the model
  calls them **directly — no `search_tools` round-trip**. Everything else stays
  discovery-only. Pinned tools appear in `openai_tools`/`anthropic_tools` named
  by their `.` → `__` path; `dispatch`/`adispatch` route those names straight to
  execution, and scopes hide/deny them like any other tool. This is the "keep
  your 3–5 most-used tools loaded" pattern, made first-class.

### Changed
- **Browse now falls back to search on a bad guess.** When the model calls
  `search_tools(path="datetime")` and there is no such category/service, SIFT
  treats the guess as a query instead of returning `unknown category` — saving
  the wasted round-trip the error used to cost. Valid paths still list the level.

Modeled cost on a zero-context "what's today's date?" (chars≈tokens/4): the
image's 4-inference trace (bad browse guess → search → execute → answer) →
**3 inferences with browse-fallback (~−26%)** → **2 inferences when the time
tool is pinned (~−44%)**. Honest note: a tool whose parameters carry meaning
(e.g. the timezone here) must NOT be auto-executed at search time — that would
silently return the wrong answer. Pinning keeps the model's parameter decision
intact while removing only the discovery round-trip.

## [0.4.1] — 2026-07-02

Performance/robustness patch on the subprocess sandbox (from an external
report — verified before fixing; the "fastembed/onnxruntime in the child" part
of the report was inaccurate since those are lazy, but the package-import
overhead was real).

### Fixed
- **Sandbox child no longer imports the ``sift`` package.** It is launched as a
  plain script and loads ``sandbox.py`` standalone, so ``gateway`` →
  ``embeddings`` → numpy stay out of the child. Measured: child boot 0.43s →
  0.22s, ``run_code`` round-trip 0.36s → 0.17s (~2×) — and the process running
  untrusted code carries a smaller surface. Regression-tested (the child must
  not have ``sift``/``numpy`` in ``sys.modules``).
- ``dispatch("run_code")`` through a **scope** now respects ``max_result_chars``
  like the ``Sift`` path (it bypassed the cap).
- A child that dies at boot returns the JSON error with its stderr tail instead
  of raising ``BrokenPipeError`` at the caller; the stderr drain thread is
  joined before composing the error (no race on the tail).

## [0.4.0] — 2026-07-02

Production-readiness release: index persistence, result caps, observability,
async, session memory — and SIFT as a custom search backend for Anthropic's
native tool search (`defer_loading`).

### Added
- **Index persistence** (`Sift(index_cache="path.npz")`): document vectors are
  cached with a content+model hash; warm start loads instead of re-embedding
  (measured ~10× on 300 tools: 4.4s → 0.46s; the gap grows with catalogue size).
- **Result cap** (`Sift(max_result_chars=100_000)`, on by default): tool results
  and code-mode output sent to the model are truncated with a marker telling the
  model how the owner can trim the tool (`set_response`). A 1 MB result no
  longer lands in the context unannounced.
- **Observability** (`Sift(observer=fn)`): `search` / `execute` / `run_code`
  events with timing and error info; plus stdlib `logging` under the `"sift"`
  logger. Observer exceptions never break the tool loop.
- **Async surface**: `aexecute_tool` / `adispatch`; `async def` tools are awaited
  natively (calling one through the sync path raises a helpful `TypeError`).
- **Session memory** (`sift.session()` / `SiftSession`): discovered tools are
  remembered per conversation and *promoted* to first-class function specs on
  later turns (the `tool_reference`-expansion pattern) — no re-searching. Works
  over scopes; promoted execution stays allow/deny-enforced.
- **Anthropic native tool search integration**
  (`adapters.anthropic.deferred_tools` / `tool_search_result` /
  `run_agent_deferred`): the whole catalogue as `defer_loading: true` tools with
  SIFT as the custom client-side search tool answering with `tool_reference`
  blocks — hybrid semantic retrieval + active tool request where the built-in
  variants offer regex/BM25.
- **OpenAI Responses API driver** (`adapters.openai.run_agent_responses`).
- **`examples=`** on `@tool`/`add_tool`: "how a user asks" phrasings, indexed on
  the dense side for better discovery of ambiguous verbs.
- **`replace=` on registration** — duplicate paths now raise instead of silently
  shadowing (two imported MCP servers with a same-named tool used to overwrite
  each other without a trace).
- `py.typed` (PEP 561) — type checkers now see SIFT's hints. CI covers 3.13.

### Changed
- **Retrieval quality**: BM25 gained a light stemmer ("emails"~"email",
  "deleted"~"delete"); all-zero BM25 ties now return *no* results instead of an
  arbitrary tool; BM25 matches against lean path+description text while
  embeddings get the enriched text (params + examples) — each signal plays to
  its strength. Service entries no longer duplicate/leak sibling descriptions.
- **`min_score` is now one scale across modes** (max embedding cosine when an
  embedder exists) — a threshold tuned once applies to both `search_tools` and
  `search_request`.
- Query-side embeddings use the embedder's `embed_query` when available
  (E5-style asymmetric models; a no-op for the default bge model).
- A/B re-measured on the agent-facing view (functions only): raw query 79% vs
  active request 100% top-1.

### Security
- **SubprocessSandbox no longer inherits the parent environment** — the child
  gets a minimal allowlist (PATH etc.), so API keys never reach the process
  running untrusted code. Child stderr is now captured and surfaced (tail) when
  the sandbox dies unexpectedly, instead of being discarded.

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
