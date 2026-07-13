# SIFT benchmarks

## Code mode vs classic tool calling

**A negative result about our own feature — published because we ran the experiment.**

Code mode (the model writes a snippet that orchestrates several tools) is sold across
the industry as a way to collapse multi-turn overhead. We finally measured it, on a
100-tool catalogue, 10 tasks (5 answerable with a single call, 5 composite), live model
(`deepseek-v4-flash`, reasoning low, prompt caching on):

| condition | success | turns | raw tok | eff tok |
|---|--:|--:|--:|--:|
| **classic (search + execute)** | **100%** | **3.1** | **7,146** | **6,455** |
| code 0.7 (as shipped: run_code only) | 100% | 3.9 | 10,208 | 7,490 |
| code 0.8 (as shipped: + execute_tool) | 100% | 3.3 | 9,107 | 8,370 |

**Classic tool calling won — including on the composite tasks.** The mechanism is
measured, not guessed: modern function calling emits **parallel tool calls**. On the
N+1 task ("for every organizer of today's events, check the CRM"), the classic
condition did this:

```
turn 0: 2 tool calls   search_tools ×2
turn 1: 1 tool call    execute_tool(cal.google.list)
turn 2: 6 tool calls   execute_tool(crm.contacts.get) ×6      <- one turn, six lookups
turn 3: final answer
```

Six lookups in **one round-trip**, no sandbox, no Python. Collapsing a fan-out into a
single turn is exactly what code mode exists to do — and function calling already does
it. Code mode's remaining, real case is what parallel calls *cannot* do: **filter a
large result before it reaches the context**, genuine control flow, and calls whose
arguments depend on an earlier call's output. That case is narrower than the hype.

### What the benchmark DID validate: the 0.8.0 changes

Between the two code-mode versions the deltas are clear:

| | code 0.7 | code 0.8 |
|---|--:|--:|
| turns | 3.9 | **3.3** |
| raw tokens | 10,208 | **9,107** |
| **snippets the model wrote** | **19** | **2** |
| ...of which hit a sandbox failure mode | **5 (26%)** | **0** |

Giving code mode an `execute_tool` (0.8.0) stopped the model writing Python for
single-call requests: 19 snippets → 2. And of 0.7's 19 snippets, **5 (26%) ended in a
bare expression, tried an `import`, or produced nothing** — each of which, before
0.8.0, returned a hollow `{"stdout": ""}` or an error with no guidance, i.e. **a
wasted round-trip that re-sends the entire conversation**.

Honest caveats: one run, 10 tasks, one model — directional, not definitive. The
composite tasks may still under-represent the case where filtering in the sandbox pays
(the email tasks cost code mode *more*, which we have not fully explained). And the
catalogue is ours.

A note on how we got here: the first run of this benchmark was **invalid** — the
synthetic distractors shadowed the gold tools (`calendar.events.list` outranked the
gold `cal.google.list`), so the model burned turns on a distractor that returned
`{"ok": true}` forever. `sift.quality.selftest()` names that failure exactly, so
`build_sift()` now **gates the benchmark on it**: if a gold tool can't be found by its
own description, the run aborts instead of printing a number that measures our
catalogue instead of code mode.

Reproduce: `python benchmarks/codemode_bench.py`

## Independent dataset: MCP-tools needle (from the MCP-Zero paper)

The first SIFT evaluation on a catalogue we did NOT construct: the public
MCP-tools dataset released with [MCP-Zero](https://arxiv.org/abs/2506.01056)
(**308 servers / 2,797 tools**, the full catalogue in one index). For every tool,
discovery must surface it ("needle") from all 2,797 — query-only uses the tool's
description; the active request adds the server as ``domain``:

| condition | top-1 | top-5 |
|---|---:|---:|
| query-only (tool description) | 96.2% | 99.9% |
| **active request (domain + action)** | **99.5%** | 99.9% |

- **Local embeddings, no API**: default `bge-small-en-v1.5` (384-dim, on-device)
  + hybrid BM25 — where MCP-Zero's routing uses OpenAI `text-embedding-3-large`
  (3072-dim, paid API). 5,594 searches in 354s ≈ **63 ms/search** on CPU.
- **Context economics at this scale**: the flat schema payload of the catalogue
  is ~115k tokens per request; SIFT's fixed surface is ~0.7k (**~170× smaller**).
- Honest caveat: this is a *self-retrieval* needle test (the request is the
  tool's own documentation — mirroring the paper's premise that model-authored
  requests align with tool docs). It is not a replication of their
  LLM-in-the-loop runs.

Reproduce: `python benchmarks/mcpzero_needle.py <dataset.json>` (download
instructions in the script header).

## SIFT vs the flat-catalogue baseline

**Question:** what does hierarchical, search-first discovery (SIFT, 2 meta-tools)
buy you over the approach most tool/MCP setups use today — every tool dumped into
the model's context as function-calling specs?

**Setup**
- Model: `deepseek/deepseek-v4-flash` (via OpenRouter), `reasoning: low`, prompt caching on
- SIFT discovery: hybrid retrieval (embeddings + BM25 + RRF)
- 12 self-contained tasks; catalogue padded with distractors to 25 / 100 / 250 tools
- Success = the gold tool was actually executed during the trajectory
- `eff tok` = cost-weighted tokens (cached input discounted to 10%) — the real cost proxy

## Results

| catalog | condition | success | raw tok | eff tok | SIFT cheaper | wrong calls |
|--------:|-----------|--------:|--------:|--------:|-------------:|------------:|
|  25 | flat (market baseline) | 100% |  7,337 |  3,497 |  —    | 0.25 |
|  25 | **SIFT**               | 100% |  3,719 |  3,124 | 1.1×  | 0.00 |
| 100 | flat (market baseline) | 100% | 27,857 | 16,068 |  —    | 0.08 |
| 100 | **SIFT**               | 100% |  5,078 |  3,965 | 4.1×  | 0.00 |
| 250 | flat (market baseline) | 100% | 59,757 | 31,936 |  —    | 0.00 |
| 250 | **SIFT**               | 100% |  4,227 |  3,795 | 8.4×  | 0.00 |

## What it shows

1. **SIFT's cost is flat; flat's scales with the catalogue.** From 25→250 tools
   SIFT stays ~3–4k effective tokens; the flat baseline goes 3.5k → 16k → 32k. At
   250 tools SIFT is **8.4× cheaper** — and that's *after* caching already halved
   the flat cost (59.7k raw → 31.9k effective).
2. **SIFT keeps tool-call accuracy.** Zero wrong-tool calls at every size; the flat
   baseline drifts (0.25 at 25 tools).
3. **Same success.** With self-contained tasks both reach 100% — so the win is pure
   cost + accuracy, not a success trade-off.
4. **Tail risk.** One flat task at 250 tools blew up to **152,231 tokens** (the model
   thrashing in a huge context); SIFT on the same task used 5,710.

## Active tool request A/B (raw query vs domain+action)

Top-1 routing accuracy on a 17-tool multi-domain catalogue with deliberate verb
collisions (read/list/send/delete across gmail, calendar, drive, slack, crm,
jira), hybrid retrieval, 14 cases. Measured on the **agent-facing view**
(functions only — what `dispatch` actually renders; services are navigation
nodes). Numbers from v0.4.0's retrieval (stemmed BM25, lean-lex/rich-dense
field split):

| discovery form | top-1 |
|---|---:|
| query-only — `search_tools(raw_user_query)` | 11/14 = **79%** |
| active request — `search_request(domain, action)` | 14/14 = **100%** |

The structured request fixes the ambiguous verbs ("remove that ticket" hits
`jira.issues.create` instead of `.delete` on query-only; "open that doc" lands on
the wrong domain's `read`). Directionally consistent with the MCP-Zero paper
(query-only plateaus at ~65–72% there). Caveat: small, author-constructed
catalogue — directional evidence, not an independent benchmark.

Reproduce (offline): `python benchmarks/ab_active_request.py`

## Directional comparison to ToolMenuBench (market reference)

| approach | success | tokens |
|----------|--------:|-------:|
| All tools flat (standard MCP) | 32.1% | 56,062 |
| CMTF (SIFT-style minimal filtering) | 85.7% | 1,125 |

Same direction, larger scale in the paper (250 tools, broader distractor types and
a stricter, multi-turn success metric). Our self-contained single-tool tasks make
both conditions succeed, isolating the cost/accuracy gap.

## Optimizations in this run

Merge Search+Inspect (search returns TOON schema inline → execute directly),
trimmed discovery output, hybrid retrieval, lean system prompt, prompt caching +
low reasoning (effective-token accounting). See the project README.

## Reproduce

```bash
python benchmarks/run_benchmark.py            # sizes 25,100,250
SIFT_BENCH_SIZES=25 SIFT_BENCH_TASKS=4 python benchmarks/run_benchmark.py  # quick
```

## Out of scope

tau-bench (stateful multi-turn customer-service env with a backing DB) is an
external harness, not a metric computable from a tool schema. BFCL-style
single-shot function-call accuracy is available via `sift.evalsuite.bfcl_style`.
