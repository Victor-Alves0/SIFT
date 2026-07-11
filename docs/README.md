# SIFT documentation

**Hierarchical, search-first tool discovery for LLM agents.** Instead of dumping
a 30k-token tool catalogue into every prompt, you give the model **2 meta-tools**
(`search_tools`, `execute_tool`) and it discovers what it needs by navigating a
`category → service → function` hierarchy. The fixed prompt overhead stays ~430
tokens whether you have 5 tools or 5,000.

```python
from sift import Sift

sift = Sift()

@sift.tool("google_workspace.gmail.read",
           description="Read emails from the inbox",
           params={"m": "number:o:10:max results"},
           returns=["id", "subject", "from", "snippet", "date"])
def gmail_read(m=10):
    ...  # call the real Gmail API
    return {"id": "1", "subject": "Hi", "from": "a@b.c", "snippet": "...",
            "date": "2026-06-30", "body": "dropped by the whitelist"}

sift.build_index()
sift.search_request(domain="email", action="read the latest message")
sift.execute_tool("google_workspace.gmail.read", {"m": 1})
```

## Guides

| Guide | What it covers |
|---|---|
| [Getting started](getting-started.md) | Install, register your first tool, run a discovery→execute loop |
| [Building tools](building-tools.md) | The `@tool` decorator, params (compact + structured), `returns`, `risk`, `transform`, naming, loading from JSON |
| [Discovery & retrieval](discovery.md) | `search_tools` (query · active request · browse), retrieval modes, reranker, relevance floor, how ranking works |
| [Executing & filtering](executing-and-filtering.md) | `execute_tool`, argument coercion, response projection, the `dispatch` primitive |
| [Providers](providers.md) | OpenAI-compatible, Anthropic, LangChain, models without native tool-calling, constrained decoding |
| [Scoping](scoping.md) | Per-model `allowedTools` via `scope(allow=, deny=, allow_risky=)` |
| [Code mode & sandbox](code-mode.md) | `run_code`, the pluggable sandbox backends, the security model |
| [Importing ecosystems](importing.md) | OpenAPI specs and MCP servers → hierarchy nodes, binding executors |
| [Deployment](deployment.md) | Run SIFT as an MCP server or an OpenAPI HTTP server; Docker; auth |
| [Catalog quality](quality.md) | `lint()`, retrieval self-test, gap tracking, pin suggestions |
| [Security](security.md) | The layer model, prompt injection honestly, sandbox isolation recipes |
| [Cookbook: Google Workspace](cookbook-google-workspace.md) | A heavy 50k-token MCP behind SIFT, end to end |
| [Architecture](architecture.md) | How the registry, gateway, TOON codec and index fit together |
| [API reference](api-reference.md) | The `Sift` facade and the key classes/functions |

## The mental model in one paragraph

You register tools onto a three-level hierarchy and call `build_index()` once.
That gives you two things to hand an LLM: a small **system prompt** and **2 tool
specs**. The model calls `search_tools` (with a query, a structured `domain +
action` request, or a browse `path`) and gets back the top matches *with their
schema inline* (in [TOON](architecture.md#toon), one line per tool). It then calls
`execute_tool(path, params)`; SIFT runs your function and **filters** the result
to a per-tool whitelist before it reaches the model. `sift.dispatch(name, args)`
is the single entry point that runs whichever meta-tool call the model emitted —
that's the seam every provider adapter is built on.
