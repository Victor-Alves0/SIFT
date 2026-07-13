# SIFT benchmarks

## Code mode vs classic tool calling

**Code mode is not a general win. It has exactly one shape it wins, and it wins that
one by 2×.** 100-tool catalogue, live model (`deepseek-v4-flash`, reasoning low, prompt
caching on), 11 tasks. Effective tokens (cached input discounted) — the cost proxy:

| task shape | classic | code 0.8 | verdict |
|---|--:|--:|---|
| **single** (5) — one call answers it | **3,438** | 3,438 | tie |
| **composite** (5) — a few calls, light payloads | **5,721** | 6,222 | classic |
| **fan-out** (1) — read N items, keep a little | 8,373 | **4,146** | **code mode, 2.0×** |
| **all** (11) | **4,696** | 4,768 | tie (1.5%) |

100% success in every condition. So: **pick code mode by the shape of the work, not by
default** — and if none of your work is fan-out shaped, classic tool calling is
strictly simpler and no more expensive.

### Why classic is so hard to beat: parallel tool calls

Code mode's industry pitch is "collapse N round-trips into one". Measured, that pitch is
mostly already yours for free — modern function calling emits **parallel tool calls**:

```
turn 2: 6 tool calls   execute_tool(crm.contacts.get) ×6      <- one turn, six lookups
```

Six CRM lookups in one round-trip, no sandbox, no Python. **Turns are not where code
mode wins.**

### Where it does win: payload, not round-trips

On the fan-out task ("open my 4 most recent emails, tell me which mention QUARTERLY"),
classic and code mode take the **same 4 turns** — but classic emits 4 parallel
`execute_tool` reads, so **4 full email bodies land in the conversation and stay
there**. A snippet loops the same 4 ids inside the sandbox and returns one line.
Same turns, half the tokens (8,373 → 4,146 effective).

That is the whole case for code mode, and it is a real one: **it is the only way to
keep a large intermediate result out of the context.** Parallel calls cannot do it;
neither can anything else in the stack.

### The 0.8.2 correction: the model has to *choose* it

0.8.0 added `execute_tool` to the code-mode surface (rightly — it stopped the model
writing Python for single calls: **29 snippets → 2**, and 0.7 wasted a turn on ~5% of
its snippets, which is what 0.8.0 fixed). But an integrator reported the side effect,
and the benchmark reproduced it: **on the fan-out task the model then preferred
`execute_tool` and abandoned batching entirely — 0 snippets in 3/3 runs**, at ~76% more
tokens.

The rule was in `CODE_SYSTEM_PROMPT` ("run_code for 2+ calls") but too abstract to fire
at the moment of choice. 0.8.2 states it as the situation the model can actually see —
*"the same tool once per item in a list → ONE run_code with a loop; never one
execute_tool per item"*:

| code 0.8, fan-out task, n=3 | snippets used | raw tokens |
|---|--:|--:|
| before (0.8.1 prompt) | **0 / 3** | ~10,960 |
| after (0.8.2 prompt) | **3 / 3** | **~8,470 (−23%)** |

### Honest caveats

One model, one catalogue (ours), few tasks — directional. And this benchmark has now
been **wrong twice**, both times because the catalogue lied:

1. First run: synthetic distractors *shadowed* the gold tools (`calendar.events.list`
   outranked the gold `cal.google.list`), so the model burned turns on a distractor that
   returned `{"ok": true}` forever. `sift.quality.selftest()` named that exactly, so the
   harness now **gates on it** and aborts rather than print a number that measures our
   catalogue instead of code mode.
2. Second run: `mail.gmail.search` returned the message **bodies**, so `read(id)` was
   redundant and there was no genuine fan-out case at all — which is how 0.8.1 shipped
   the headline "classic wins on everything". It doesn't. Search now returns headers
   only, like every real mail API.

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
