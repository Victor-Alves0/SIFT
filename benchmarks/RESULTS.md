# SIFT benchmark — SIFT vs the flat-catalogue baseline

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
jira), hybrid retrieval, 14 cases:

| discovery form | top-1 |
|---|---:|
| query-only — `search_tools(raw_user_query)` | 9/14 = **64%** |
| active request — `search_request(domain, action)` | 14/14 = **100%** |

The structured request fixed 5 ambiguous cases (e.g. "what's on my agenda" landed
on the calendar *service* instead of `calendar.list`; "remove that ticket" hit
`jira.issues.create` instead of `.delete`). Matches the MCP-Zero paper's finding
that query-only retrieval plateaus at ~65–72%. Caveat: small, author-constructed
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
