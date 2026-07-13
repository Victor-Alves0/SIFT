# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semver.

## [0.8.2] — 2026-07-13

An integrator caught a regression 0.8.0 introduced — and finding it proved the 0.8.1
benchmark had been measuring the wrong thing. **0.8.1's headline was wrong and is
retracted.**

### Fixed
- **Code mode stopped batching.** 0.8.0 added `execute_tool` to the code-mode surface
  (rightly: it stopped the model writing Python for single calls, 29 snippets → 2). The
  side effect, reported from production and then reproduced: on a **fan-out** — "open
  each of these 4 emails" — the model preferred one `execute_tool` per item and
  abandoned `run_code` entirely (**0 snippets in 3/3 runs, ~76% more tokens**). Turns
  don't suffer, because parallel tool calls run them in one round-trip — so nothing
  looks broken; but every payload lands in the conversation permanently.
  `CODE_SYSTEM_PROMPT` had the rule ("run_code for 2+ calls") but too abstract to fire
  at the moment of choice. It now names the situation the model can actually see —
  *"the same tool once per item in a list → ONE run_code with a loop; NEVER one
  execute_tool per item"* — and the `execute_tool` spec says the same at the point of
  use. Measured: **snippets 0/3 → 3/3, tokens −23%** on the fan-out task.

### Changed
- **`benchmarks/RESULTS.md`: the code-mode verdict is corrected.** 0.8.1 concluded
  "classic tool calling beat code mode on every metric". That benchmark had no genuine
  fan-out case: its `mail.gmail.search` returned the message **bodies**, so `read(id)`
  was redundant and the model rightly skipped it. With search returning headers only —
  like every real mail API — the picture is:

  | task shape | classic | code mode | |
  |---|--:|--:|---|
  | single (5) | **3,438** | 3,438 | tie |
  | composite, light payloads (5) | **5,721** | 6,222 | classic |
  | **fan-out (1)** — read N items, keep a little | 8,373 | **4,146** | **code mode, 2.0×** |
  | all (11) | **4,696** | 4,768 | tie |

  So: code mode is **not** a general win, and **not** a loss either — it wins exactly
  one shape, and wins it by 2×. Turns are not where it wins (parallel tool calls
  already collapse a fan-out); **payload** is. README and `docs/code-mode.md` now say
  that instead of 0.8.1's "code mode lost".

## [0.8.1] — 2026-07-13

Benchmarked code mode for the first time — and it lost. Two library bugs surfaced
while doing it.

### Added
- **`benchmarks/codemode_bench.py`** — code mode vs classic tool calling, live model,
  single-call and composite tasks. Result: **classic won on every metric** (3.1 vs 3.3
  turns, 6.5k vs 8.4k effective tokens, both 100%). Measured mechanism: function
  calling emits **parallel tool calls**, so a 6-lookup fan-out already collapses into
  one turn without a sandbox — which is most of what code mode was for. Published as a
  negative result; `README` and `docs/code-mode.md` now lead with it instead of the
  "collapses multi-turn overhead" pitch. It *did* validate 0.8.0: giving code mode an
  `execute_tool` cut snippets written from **19 to 2**, and 5 of those 19 (26%) hit a
  sandbox failure mode 0.8.0 fixed.
  The harness gates itself on `quality.selftest()` — the first run was invalid because
  synthetic distractors shadowed the gold tools, and `selftest` named it exactly.

### Fixed
- **The observer never saw tools called from inside a snippet.** `execute` events were
  emitted from `dispatch` only, so code-mode traffic showed up as one fat `run_code`
  with none of the tools it ran — and `GapTracker.suggest_pins()` under-counted every
  tool used from a snippet. Both `execute_tool` and `call()` inside the sandbox now
  funnel through `Sift.execute_tool`, which is where the event is emitted.
- **Code mode got a weaker `search_tools` than classic mode.** `code_tool_specs()`
  defined its own reduced spec taking only `q`, so the model could not issue an active
  tool request (`domain` + `action`) — measurably the sharper route (99.5% vs 96.2%
  top-1 on the MCP-Zero dataset). It now reuses `metatools.tool_specs()` unchanged.
- `lint()` warns when `min_score` is 0 — otherwise nobody discovers they need
  `suggest_min_score()`, and discovery silently keeps returning its best guess forever.

## [0.8.0] — 2026-07-13

Nothing in SIFT may succeed hollowly. In tool calling every wasted round-trip
re-sends the entire context, so a result the model cannot learn from — an empty
`{"stdout": ""}`, a top-k of irrelevant tools — does not cost a call, it costs a
turn. Both were fixed, and code mode was reviewed against the current frontier.

### Added
- **`execute_tool` is now part of the code-mode surface.** Code mode used to expose
  only `search_tools` + `run_code`, forcing the model to write Python even for a
  single call — paying sandbox overhead and a real parse-failure rate to do what one
  JSON call does. Code mode's win is COMPOSITE work (many calls, control flow,
  filtering a large result); for one call, direct execution is cheaper and cannot
  fail to compile.
- **`CODE_SYSTEM_PROMPT` now tells the model to keep `output` small** — filter,
  slice and aggregate *inside* the snippet. Intermediate values stay in the sandbox
  for free, while everything in `output` is re-sent with the conversation on every
  later turn. This is the pattern behind code execution's headline savings, and it
  was the one SIFT never asked for.
- **REPL semantics in `run_code`**: a bare expression on the last line is promoted
  to `output`, like a notebook. Models routinely end a snippet with the value they
  mean to return instead of assigning it; that intent is honoured rather than
  charged a round-trip. An explicit `output = …` still wins, and a trailing
  `print(...)` is untouched.
- **`sift.sandbox.SANDBOX_RULES`** — the sandbox's limits as prompt text,
  **generated from the policy constants themselves**, and now part of
  `CODE_SYSTEM_PROMPT`. The rules cannot drift from what is enforced.
- **`sift.quality.suggest_min_score()`** — calibrate the relevance floor from the
  catalogue instead of guessing it. Scores every tool's own description/examples
  (positives) against needs the catalogue cannot serve (`negatives=`) and proposes
  the value between them; says so honestly when the two do not separate (a catalogue
  problem no floor can fix) and when no negatives were given (a recall ceiling, not
  a calibration).

### Fixed
- **Hollow success in `run_code`**: a snippet that assigned nothing and printed
  nothing returned `{"stdout": ""}` — an empty *success* the model could not learn
  from, so it guessed, retried, or fell back to `search_tools` (thousands of tokens,
  measured in the field). It now returns an actionable `error` + `hint`.
  Crucially, "empty" is still distinguished from "failed": a snippet that *did* set
  `output` to an empty value returns `{"output": null}`, and a no-result error
  carries a `ran` field naming how many tool calls already executed — so a retry
  never silently re-sends an email.
- **Sandbox policy was enforced but never communicated**: `CODE_SYSTEM_PROMPT`
  told the model to assign `output` but never mentioned that imports are rejected,
  so the model discovered the rule by burning a turn on `SandboxError`. Policy
  violations now return the rules as a `hint`, and the prompt states them up front.
- **Discovery's no-match message was effectively dead code.** It only fired when
  `min_score > 0`, and `min_score` defaults to 0.0 — so search always handed back
  its top-k however irrelevant, teaching the model that a tool always exists. The
  floor is now calibratable (`suggest_min_score`), and the message tells the model
  what to do instead: answer directly, and do **not** re-search with a synonym
  (which would burn another full context for the same answer).
- Docs: the subprocess backend has been launched as a script (not `-m
  sift._sandbox_child`) since 0.4.1; `docs/code-mode.md` still described the old
  form.

## [0.7.0] — 2026-07-11

Evidence + quality release: SIFT evaluated on the public MCP-Zero dataset, an
executable catalog-quality toolkit, and a hardened execution layer.

### Added
- **Independent-dataset benchmark** (`benchmarks/mcpzero_needle.py`): needle-in-
  a-haystack over the public MCP-tools catalogue from the MCP-Zero paper (308
  servers / 2,797 tools) — the first SIFT evaluation on a dataset we didn't
  construct. See benchmarks/RESULTS.md for numbers (local bge-small embeddings,
  no API).
- **Catalog quality toolkit** (`sift.quality`): `lint()` (missing/short/long
  descriptions, undocumented params, near-duplicate tools via the built vectors,
  fragmented categories), `selftest()` (every tool must be findable with its own
  description/examples — failures name who beat it), and `GapTracker` (observer:
  `gaps()` = searches that matched nothing; `suggest_pins()` = hot tools worth
  pinning).
- **Result cache** (`@tool(cacheable=True, cache_ttl=60)`): opt-in memoization
  of idempotent reads per (path, params).
- **Per-tool timeout** (`@tool(timeout=10)`): the caller gets a clean
  `TimeoutError` and moves on (honest semantics documented — Python threads
  can't be killed; async tools get real cancellation).
- **Incremental rebuild**: `build_index()` after adding tools re-embeds only
  new/changed texts instead of the whole catalogue.
- **Schema-in-error**: a parameter error now carries the tool's TOON line
  (`"schema": ...`) so the model fixes the call in one retry.
- **`on_result` hook** (`Sift(on_result=fn)`): global post-filter over every
  tool result after projection — the place for prompt-injection scrubbing.
- **Importer sanitization** (default on): third-party MCP/OpenAPI descriptions
  are scrubbed (control chars, collapsed whitespace, length cap) before entering
  the index; `sanitize=False` opts out. `compress_params` now survives the
  malformed schemas found in real MCP catalogues, and maps
  integer/boolean/array/object types faithfully (was flattened to number/string).
- **Native Gemini adapter** (`adapters.gemini`, `[gemini]` extra) and an
  **OpenTelemetry observer bridge** (`sift.otel.otel_observer`, `[otel]` extra).
- **Docs**: security model (`docs/security.md` — injection honestly, container/
  seccomp recipe), catalog quality guide, and a Google Workspace cookbook (the
  50k-token MCP case end to end).
- Search observer events now include ``hits`` (feeds `GapTracker`).

## [0.6.0] — 2026-07-11

DX/robustness release driven by field feedback from an integrator building a
production app on 0.5.0 — every reported bug was reproduced before fixing.

### Fixed
- **Bare registration is callable, not a trap.** A tool registered without
  ``params=`` used to be discoverable but fail on *every* call (only declared
  params are bound). The spec is now **derived from the function signature**
  (annotations → types, defaults → optional): ``def add(a: int, b: int)``
  just works. Explicit ``params=`` still wins.
- **Typed params fail loudly, not plausibly.** An unparseable value for a
  declared type used to pass through raw (``'x' * 4 == 'xxxx'`` — garbage that
  propagates hallucination); it now raises a clean, named error the model can
  retry from (``parameter 'a': expected an integer, got 'x'``).
- **Req-flag typos no longer silently mean "optional"**: ``r`` is accepted as
  required (users assume it); any other unknown flag raises at registration.
- **Errors point at the fix, not the wrong direction.** A missing ``path``
  argument is named as such (was: ``tool None is not allowed in this scope`` — a
  misclassified permission error), and path errors carry a ``hint`` telling the
  model to ``search_tools`` for the correct path — weak models read a bare
  "not allowed" and give up. ``KeyError`` messages are no longer double-quoted.
- **Code-mode watchdog no longer kills long-running tools.** The subprocess
  timeout now budgets the *sandboxed snippet* only — the clock pauses while the
  parent runs a proxied (trusted) tool, so a deep-search tool slower than the
  timeout completes instead of having its result discarded.

### Added
- **Per-scope pins**: ``sift.scope(allow=..., pin=[...])`` — per-model hot tools
  with no shared mutable state on the parent (kills the ``_pinned[:] = ...`` /
  ``clear()`` dance). A pin denied by the scope's own rules raises.
- **Async on scopes**: ``SiftScope.adispatch`` / ``aexecute_tool`` with the same
  allow/deny enforcement — no more ``run_in_threadpool`` around every call.
- **``on_risky`` hook** (``Sift(on_risky=fn)``): human-in-the-loop confirmation
  for ``risk=True`` tools, called with the prepared args right before execution;
  ``False``/raise blocks. The confirm-before-send pattern, standardized.
- **``meta`` dict** on ``Sift`` and ``SiftScope`` — a sanctioned home for
  integrator metadata (no more private-attribute hacks).
- **Language guidance in the system prompt**: models are told to write search
  queries in the language of the tool descriptions (translating the user's
  phrasing), and the docs cover multilingual embedding models
  (``paraphrase-multilingual-*``, ``multilingual-e5-large``, ``jina-v3``).

### Docs
- README reframed: SIFT is the **HOW** of tool use (exposure, discovery,
  execution, governance); the **WHEN** stays with the model, driven by your
  instructions and tool descriptions. Benchmarks now read as **schema payload**,
  not tool count — what actually scales cost (a single heavy MCP ≈ 50k tokens).

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
