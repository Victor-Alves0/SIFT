# SIFT — Search · Inspect · Filter · Trigger

[![CI](https://github.com/Victor-Alves0/SIFT/actions/workflows/ci.yml/badge.svg)](https://github.com/Victor-Alves0/SIFT/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sift-tools.svg)](https://pypi.org/project/sift-tools/)
[![Python](https://img.shields.io/pypi/pyversions/sift-tools.svg)](https://pypi.org/project/sift-tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Hierarchical, search-first tool discovery for LLM agents.** Give the model
**2 meta-tools** instead of a 30k-token catalogue — it discovers the rest by
navigating. Drop-in for OpenAI function-calling, LangChain, or MCP.

```bash
pip install sift-tools
```

Repo: [github.com/Victor-Alves0/SIFT](https://github.com/Victor-Alves0/SIFT) · PyPI: [sift-tools](https://pypi.org/project/sift-tools/) · 📖 Full docs: [`docs/`](docs/README.md)

```python
from sift import Sift

sift = Sift()

@sift.tool("google_workspace.gmail.read",
           description="Read emails from the inbox",
           params={"q": "string:o:is:unread:search query", "m": "number:o:10:max"},
           returns=["id", "subject", "from", "snippet", "date"])
def gmail_read(q="is:unread", m=10):
    ...  # call the real Gmail API
    return {"id": "1", "subject": "Hi", "from": "a@b.c", "snippet": "...",
            "date": "2026-06-30", "body": "filtered out by the whitelist"}

sift.build_index()

# discovery — the matching schema comes back inline, so you execute directly
sift.search_request(domain="email", action="read the latest message")  # active request
sift.search_tools("read my last email")                     # …or a simple query
sift.execute_tool("google_workspace.gmail.read", {"m": 1})  # → run + filter
```

## Why

The model never sees the whole catalogue — only 2 tools. It discovers what it
needs by walking **category → service → function**. The system prompt stays a
fixed ~200 tokens whether your tool schemas total 1k or 100k tokens. Adding a
tool is one decorator. Schemas are returned in **TOON** (one line per tool), and
responses are **filtered** to a per-tool whitelist.

> **What actually costs you is schema *size*, not tool *count*.** A single
> Google Workspace MCP can inject ~50k tokens of schemas — "few tools" is not
> "cheap". SIFT's surface is constant because the model pulls one TOON line per
> tool it actually needs, regardless of how heavy the full schemas are.

### What SIFT decides — and what it deliberately doesn't

SIFT is the **HOW** of tool use: how tools are exposed (2 meta-tools), found
(hybrid retrieval + active request), described (TOON), executed (typed args,
projection, caps) and governed (scopes, `risk`, `on_risky`). The **WHEN** — 
whether the model reaches for a tool at all, and which need maps to which tool —
remains the model's decision, driven by **your system instructions and your tool
descriptions**. Good descriptions and `examples=` improve discovery; clear
instructions decide when the model searches. SIFT gives that decision good
plumbing; it doesn't make it for the model.

```
search_tools(...)          → discovery WITH schema inline (query, active         [Search + Inspect]
                             request, or hierarchy browse — local embeddings)
execute_tool(path, params) → run + response filtering                            [Trigger + Filter]
```

`search_tools` merges discovery and inspection: a match already carries its TOON
schema, so the model calls `execute_tool` next — no separate "inspect" round-trip.

### Active tool request (sharper discovery at scale)

Beyond a raw query, the model can state a **structured intent** — `domain` (the
platform / permission area) + `action` (the operation + target). A model-authored
request aligns better with the tool docs than a user's raw phrasing, which lifts
routing accuracy when the catalogue is large (the [MCP-Zero](https://arxiv.org/abs/2506.01056)
finding). SIFT routes it in two stages (score the service on `domain`, the
function on `action`, fuse) over its **hybrid** signals — not dense-only:

```python
sift.search_request(domain="calendar", action="create an event tomorrow at 3pm")
```

## Benchmarks

**SIFT vs the flat catalogue** (what most tool/MCP setups do today — every schema
injected every turn). Agent-level runs, `deepseek/deepseek-v4-flash` via OpenRouter,
prompt caching on, 12 tasks, catalogue padded with distractors
([full method & results](benchmarks/RESULTS.md)). The first column is tool
*count*, but read it as **schema payload** — that's what a flat setup injects and
what actually scales the cost (~95 tok/tool in this catalogue; a heavy real-world
MCP reaches the same payload with far fewer tools):

| catalog (≈schema payload) | condition | success | eff. tokens | SIFT cheaper | wrong calls |
|--------:|-----------|--------:|------------:|-------------:|------------:|
|  25 (≈2.4k tok) | flat baseline | 100% |  3,497 | —    | 0.25 |
|  25 (≈2.4k tok) | **SIFT**      | 100% |  3,124 | 1.1× | 0.00 |
| 100 (≈9.5k tok) | flat baseline | 100% | 16,068 | —    | 0.08 |
| 100 (≈9.5k tok) | **SIFT**      | 100% |  3,965 | **4.1×** | 0.00 |
| 250 (≈24k tok) | flat baseline | 100% | 31,936 | —    | 0.00 |
| 250 (≈24k tok) | **SIFT**      | 100% |  3,795 | **8.4×** | 0.00 |

SIFT's cost stays ~flat as the catalogue grows; the flat baseline scales with it
(and one flat task at 250 tools blew up to 152k tokens — SIFT used 5.7k on the
same task). Zero wrong-tool calls at every size.

**Raw query vs active tool request** (top-1 routing accuracy on the agent-facing
view, 17-tool catalogue with deliberate verb collisions, hybrid retrieval):

| discovery form | top-1 |
|---|---:|
| `search_tools("cancel my 3pm")` — raw user query | 79% |
| `search_request(domain="calendar", action="cancel an event")` | **100%** |

Matches the MCP-Zero finding (query-only retrieval plateaus at ~65–72%). Honest
caveat: small, author-constructed catalogues — directional, not independent
benchmarks. Both are reproducible:

```bash
python benchmarks/run_benchmark.py        # SIFT vs flat (needs OPENROUTER_API_KEY)
python benchmarks/ab_active_request.py    # query vs active request (offline)
```

## Install

```bash
pip install sift-tools                 # core (local embeddings, no API key)
pip install "sift-tools[langchain]"    # + LangChain adapter
pip install "sift-tools[mcp]"          # + MCP server adapter
pip install "sift-tools[all,dev]"      # everything + test tooling
```

Embeddings run **locally** via `fastembed` (ONNX) — no embedding API key needed.
Swap in any embedder with an `embed(texts) -> list[vector]` method.

## Bring your own model (provider-agnostic)

The core is **LLM-agnostic** — it never calls a model itself. It hands you the
2 tool specs + a system prompt, and `sift.dispatch(name, args)` executes whatever
tool call your model emits. Wire it to any provider:

```python
# 1) OpenAI-compatible (OpenAI, OpenRouter, DeepSeek, Together, Groq, Mistral,
#    and LOCAL servers: Ollama / LM Studio / vLLM) — works out of the box
from openai import OpenAI
from sift.adapters.openai import run_agent

client = OpenAI()                                              # OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # Ollama, local
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)    # OpenRouter
run_agent(sift, client, "gpt-4o-mini", "what's my last email?")

# 2) Native Anthropic (Messages API)
import anthropic
from sift.adapters.anthropic import run_agent as run_claude
run_claude(sift, anthropic.Anthropic(), "claude-haiku-4.5", "what's my last email?")

# 3) LangChain (Anthropic, Gemini, Cohere, Bedrock, Ollama, ...)
agent_tools = sift.langchain_tools()        # plug into any LangChain agent

# 4) Expose SIFT itself as an MCP server (Claude Desktop, IDEs, ...)
sift.serve_mcp()

# 5) Any other SDK — the universal primitive:
specs  = sift.openai_tools()                # give your model the 2 tool specs
system = sift.system_prompt
answer = sift.dispatch(name, arguments)     # run a tool call -> string back
```

| Provider / path | How | Status |
|---|---|---|
| OpenAI-compatible (incl. local Ollama/vLLM) | `openai_tools()` + `dispatch()` / `adapters.openai.run_agent` | ✅ live-tested |
| Native Anthropic | `adapters.anthropic.run_agent` | ✅ unit + offline-tested |
| LangChain | `langchain_tools()` | ✅ live-tested |
| MCP clients | `serve_mcp()` | ✅ |
| **No native tool calling** (base/small models) | `adapters.prompted` | ✅ live-tested |

### Weak or no-tool-calling models (Llama 3B, base models, …)

`dispatch` is format-agnostic, so any text model can drive SIFT via a prompted
JSON protocol — no native function calling required:

```python
from sift.adapters.prompted import run_agent, single_decision

def generate(prompt: str) -> str:      # wrap ANY text model (HF, llama.cpp, Ollama)
    return my_model(prompt)

run_agent(sift, generate, "what's my last email?")     # text-protocol tool loop
single_decision(sift, generate, "read my last email")  # 1 decision, for the weakest models
```

For small local models, constrain the decoder so output is always parseable:

```python
sift.tool_call_schema()   # JSON Schema -> Outlines / LM Format Enforcer / vLLM guided_json
sift.json_gbnf()          # GBNF grammar -> llama.cpp
```

SIFT's tiny 2-tool surface actually *helps* weak models (less to get lost in).
Realistic floor is ~1–3B params; sub-1B models (OPT-350M) can be interfaced but
are too small to follow the format reliably.

## Import an existing ecosystem

```python
from sift.importers.openapi import register_openapi
from sift.importers.mcp import import_mcp_stdio, register_listing

register_openapi(sift, spec, category="acme")                    # OpenAPI 3.x
await import_mcp_stdio(sift, "npx", ["-y", "@modelcontextprotocol/server-github"],
                       category="integrations", service="github")  # MCP server
```

Each operation/tool becomes a node in the hierarchy — instantly searchable.

## Per-model scoping (`allowedTools`) & response projection

Built for hubs like OpenWebUI: build the catalogue **once**, then give each model
a scoped view of which tools it may see/run, and trim what each tool returns.

```python
# pick tools for this model (globs over the dotted path); reuses the built index
view = sift.scope(allow=["google_workspace.gmail.*", "web.search.*"],
                  deny=["*.delete", "*.send"])
view.dispatch("search_tools", {"q": "read my last email"})  # only allowed tools
view.execute_tool("crm.contacts.delete", {})                # PermissionError (deny wins)

# trim a verbose tool's result so each call costs fewer tokens (great for MCPs):
sift.set_response("google_workspace.gmail.query",
                  transform=lambda r: {"ids": [m["id"] for m in r["messages"]]})
sift.set_response("google_workspace.gmail.read", returns=["id", "subject", "from"])
```

**Idle cost:** when a tool isn't used (the user just says "hi"), SIFT adds only the
~430-token fixed surface (system prompt + 2 meta-tool specs) — **independent of
catalogue size**, and ~free across a conversation with prompt caching. A flat
catalogue instead injects *every* schema each turn (~2.4k tokens at 25 tools,
~95k at 1,000).

## Production knobs

```python
sift = Sift(
    index_cache="./sift-index.npz",   # persist vectors: warm start loads in ~ms
                                      # instead of re-embedding (10×+ on big catalogues)
    max_result_chars=100_000,         # cap any tool result sent to the model (default);
                                      # truncation tells the model how to trim the tool
    observer=lambda ev, data: print(ev, data),   # search/execute/run_code events
)                                                # with timing — wire tracing here
```

Async agents: `await sift.aexecute_tool(...)` / `await sift.adispatch(...)` —
`async def` tools are awaited natively. Thread-safety: register + `build_index()`
first, then serve; after the build, discovery/execution are read-only on SIFT's
side and safe to call concurrently.

## Cheap even with few (but heavy) tools — pin the hot ones

The cost of a flat catalogue isn't the *number* of tools, it's the *schema size*
(one Google Workspace MCP can be ~50k tokens). SIFT already fixes the bulk of
that — the model sees ~430 tokens and pulls the one-line TOON of just the tool it
needs. What's left is the discovery round-trip. For a handful of tools asked
often, skip discovery entirely by **pinning** them:

```python
sift.pin("utils.time.now", "google_workspace.gmail.read")   # a few hot tools
```

Pinned tools are always-visible first-class specs — the model calls them in **one
round-trip, no search** — while everything else stays discovery-only. Modeled on
a zero-context "what's today's date?": 4 inferences (the trace below) → 2 when the
time tool is pinned (~−44% tokens). And `search_tools(path="…")` now **falls back
to a search** when the model guesses a category that doesn't exist, instead of
wasting a round-trip on an error.

> Don't force it: a tool whose parameters carry meaning (a timezone, a query)
> must still get its schema before the model fills them — pinning removes only the
> discovery hop, never the model's parameter decision.

## Session memory (no re-searching)

Without memory, a model re-searches for the same tool every turn. A session
remembers what discovery surfaced and **promotes** those tools to first-class
function specs on later turns (the same pattern Anthropic's tool search uses to
expand `tool_reference` blocks):

```python
session = sift.session()
tools = session.tools()            # 2 meta-tools now; grows as tools are found
session.dispatch(name, args)       # records discoveries; promoted tools are
                                   # called DIRECTLY by name — no search round-trip
```

Works over a scope too (`SiftSession(view)`) — promoted execution stays
allow/deny-enforced.

## Claude-native tool search (defer_loading) with SIFT retrieval

Anthropic's tool search tool ships regex/BM25 variants — and an official hook for
[custom search backends](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool#custom-tool-search-implementation).
SIFT plugs into that slot: the whole catalogue goes up as `defer_loading: true`
tools, discovery runs through SIFT's **hybrid retrieval + active tool request**,
and the API expands the returned `tool_reference` blocks natively:

```python
from sift.adapters.anthropic import run_agent_deferred
run_agent_deferred(sift, anthropic.Anthropic(), "claude-opus-4-8",
                   "what's my last email?", keep=("google_workspace.gmail.read",))
# or wire it yourself: deferred_tools(sift) + tool_search_result(sift, id, args)
```

## Hybrid retrieval & reranking

Discovery fuses **embeddings + BM25** with Reciprocal Rank Fusion (semantics +
exact terms), and an optional cross-encoder **reranker** sharpens the final order:

```python
sift = Sift(retrieval="hybrid")          # default; also "embedding" or "bm25"

from sift.rerank import FastEmbedReranker
sift = Sift(reranker=FastEmbedReranker())  # opt-in cross-encoder rerank
```

`retrieval="bm25"` needs no model download at all. Set a relevance floor so
discovery returns *nothing* (an explicit "no matching tools") instead of the
nearest-but-irrelevant tool when the catalogue doesn't cover the request:

```python
sift = Sift(min_score=0.3)   # cosine floor (tune per embedding model)
```

## Code mode (compose many tools in one turn)

Instead of one round-trip per tool, let the model write a snippet that
orchestrates tools in a single turn (collapses multi-turn overhead):

```python
tools  = sift.code_tools()          # search_tools + run_code
system = sift.code_system_prompt
# in the loop, run_code executes:  call(path, **params), search(q), schema(path)
sift.run_code("output = call('google_workspace.gmail.read', m=1)")
```

Execution goes through a **pluggable sandbox** backend:

```python
from sift.sandbox import InProcessSandbox, SubprocessSandbox

Sift(sandbox=InProcessSandbox())                 # default — fast, trusted catalogues
Sift(sandbox=SubprocessSandbox(timeout=10))      # isolated process for untrusted code
```

- **InProcessSandbox** (default): AST policy (no imports, dunders, `str.format`
  escapes, or dangerous names) + a line budget + restricted builtins. Fast; for
  catalogues you trust.
- **SubprocessSandbox**: runs the snippet in a **separate process** — tool calls
  are proxied back to the parent (the child can't touch your tools/memory), with a
  wall-clock watchdog and CPU/memory rlimits (Unix). A big step up, but **not a
  VM**: on its own it doesn't block network/filesystem syscalls. For fully
  untrusted input, wrap it in OS isolation (container / seccomp).

## Evaluate

```python
from sift.bench import Task, run_filter, token_report
print(token_report(sift.registry).format())     # TOON vs JSON token savings
print(run_filter(sift, tasks, top_k=3).format()) # filter-level metrics (no LLM cost)

from sift.evalsuite import Case, bfcl_style       # BFCL-style function-call accuracy
print(bfcl_style(call_model, sift.registry, cases).format())

from sift.agentbench import build_catalog, run_flat, run_sift  # SIFT vs flat baseline
```

Filter-level metrics (à la ToolMenuBench): gold next-tool exposure, no-visible-tool
rate, average visible tools, MRR, risky-tool exposure, unauthorized risky exposure.
(tau-bench's stateful environment is out of scope — it's an external harness.)

## Schema format

A param is either the **compact string** `"<type>:<req>:<default>:<description>"`
(`req` is `n` required / `o` optional) or the **structured dict** form when you
need a default containing `:` (e.g. a Gmail `is:unread` query):

```python
params={
    "m": "number:o:10:max results",                                  # compact
    "q": {"type": "string", "default": "is:unread", "desc": "query"},  # structured
}
```

`returns` is the response whitelist. `risk=True` flags high-impact actions
(send/delete) — surfaced as `|risk` in TOON so the agent can confirm first.

## Make imported tools runnable

Importers populate the hierarchy for discovery; bind an executor to also run them:

```python
from sift.importers.openapi import register_openapi, httpx_request
register_openapi(sift, spec, category="acme",
                 request=httpx_request("https://api.acme.com"))

from sift.importers.mcp import register_listing
register_listing(sift, listing, category="integrations", service="github",
                 executor=lambda name, params: my_mcp_proxy(name, params))
```

For a live MCP server, `connect_mcp_stdio` launches it, registers its tools AND
binds execution (keeps the session open) in one call:

```python
from sift.importers import connect_mcp_stdio
proxy = connect_mcp_stdio(sift, "npx", ["-y", "@modelcontextprotocol/server-github"],
                          category="integrations", service="github")
sift.build_index()
# ... imported MCP tools now run out of the box ...
proxy.close()
```

## Deploy as a server

Run SIFT as a standalone server so a hub (OpenWebUI, IDEs, …) connects to *it*,
and you wire tools/MCPs/OpenAPI into SIFT — one hub for everything.

```bash
# OpenAPI HTTP server (OpenWebUI "tool server", REST clients)
python examples/serve_http.py            # OpenAPI at /openapi.json, docs at /docs

# MCP server
python examples/serve_mcp.py             # stdio (Claude Desktop)
python examples/serve_mcp.py sse         # HTTP/SSE (remote)

# Docker (OpenAPI server)
docker build -t sift-server .
docker run -p 8000:8000 -e SIFT_API_KEY=secret sift-server
```

Set `SIFT_API_KEY` to require `Authorization: Bearer <key>`. Pass a `scope=` to
`build_app` / `serve_http` to expose only a subset of tools per server. Customize
`examples/serve_http.py` with your own `@sift.tool`s and importers.

> OpenWebUI: add the server URL under Tools → OpenAPI tool server. (For MCP,
> bridge via `mcpo` or OpenWebUI's MCP support.) The model then sees just the 2
> meta-tools and discovers your catalogue through them.

## Documentation

Full guides live in [`docs/`](docs/README.md):

- [Getting started](docs/getting-started.md) · [Building tools](docs/building-tools.md)
- [Discovery & retrieval](docs/discovery.md) · [Executing & filtering](docs/executing-and-filtering.md)
- [Providers](docs/providers.md) · [Scoping](docs/scoping.md) · [Code mode & sandbox](docs/code-mode.md)
- [Importing ecosystems](docs/importing.md) · [Deployment](docs/deployment.md)
- [Architecture](docs/architecture.md) · [API reference](docs/api-reference.md)

## Repo layout

```
src/sift/            the Python library (the product)
  registry.py        hierarchy + navigation
  toon.py            TOON codec
  embeddings.py      local fastembed backend
  retrieval.py       BM25 + RRF (hybrid search)
  rerank.py          optional cross-encoder reranker
  gateway.py         the 2 meta-tools + hybrid search + active request + filtering
  scope.py           per-model allow/deny tool scoping (allowedTools)
  metatools.py       canonical tool specs + system prompt
  codemode.py        run_code: orchestrate tools in one turn
  sandbox.py         pluggable code-mode backends (in-process / subprocess)
  constrain.py       JSON schema / GBNF for constrained decoders
  http_server.py     OpenAPI HTTP tool server (serve_http)
  adapters/          openai · anthropic · langchain · mcp_server · prompted
  importers/         mcp · openapi · mcp_proxy (live MCP execution)
  bench.py           filter-level metrics + token report
  agentbench.py      SIFT vs flat-catalogue benchmark
  evalsuite.py       BFCL-style function-call accuracy
docs/                full guides (getting-started, building-tools, …)
examples/            quickstart, live smokes, serve_http / serve_mcp
tests/               pytest suite (offline, deterministic)
.github/workflows/   CI (lint+test) and PyPI publish
Dockerfile           containerized OpenAPI server
```

## License

MIT.
