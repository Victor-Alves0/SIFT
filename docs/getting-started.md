# Getting started

## Install

```bash
pip install sift-tools                 # core (local embeddings, no API key)
pip install "sift-tools[langchain]"    # + LangChain adapter
pip install "sift-tools[mcp]"          # + MCP server adapter
pip install "sift-tools[openai]"       # + OpenAI/OpenRouter driver
pip install "sift-tools[anthropic]"    # + native Anthropic driver
pip install "sift-tools[server]"       # + FastAPI/uvicorn (OpenAPI HTTP server)
pip install "sift-tools[all,dev]"      # everything + test tooling
```

Requires Python 3.10+. The only hard dependencies are `numpy` and `fastembed`;
everything else is an optional extra. Embeddings run **locally** via `fastembed`
(ONNX) — no embedding API key is needed. The model (`BAAI/bge-small-en-v1.5` by
default) downloads on first use.

## Your first tool

A tool is a normal Python function decorated with `@sift.tool`, placed at a
dotted path `category.service.function`:

```python
from sift import Sift

sift = Sift()

@sift.tool(
    "google_workspace.gmail.read",
    description="Read emails from the inbox, newest first",
    params={"m": "number:o:10:max results"},   # optional param, default 10
    returns=["id", "subject", "from", "snippet", "date"],  # response whitelist
)
def gmail_read(m=10):
    # a real implementation calls the Gmail API here
    return {"id": "msg_1", "subject": "Meeting tomorrow", "from": "joao@acme.com",
            "snippet": "Confirming our meeting.", "date": "2026-06-30",
            "body": "this field is dropped — not in `returns`"}

sift.build_index()   # build the search index ONCE, after registering tools
```

> **`build_index()` is required.** Discovery raises until you call it. Register
> all your tools first, then build. See [Building tools](building-tools.md) for
> the full `@tool` surface (params, `risk`, `transform`, …).

## Drive it manually

The three things an agent does — discover, then execute — map to two methods:

```python
# 1) discovery: a simple natural-language query …
for r in sift.search_tools("read my last email", top_k=3):
    print(r.score, r.path)     # → google_workspace.gmail.read

# … or a structured "active tool request" (usually more accurate at scale):
sift.search_request(domain="email", action="read the latest message")

# 2) execute by path; the result is filtered to the `returns` whitelist
sift.execute_tool("google_workspace.gmail.read", {"m": 1})
# → {"id": "msg_1", "subject": "...", "from": "...", "snippet": "...", "date": "..."}
#   (note: no "body")
```

To browse the hierarchy instead of searching, pass a `path`:

```python
sift.get_tool_schema("")                        # list categories
sift.get_tool_schema("google_workspace")        # list services in a category
sift.get_tool_schema("google_workspace.gmail")  # list functions in a service
```

## Drive it with an LLM

You rarely call the methods above by hand — you hand the specs to a model and let
it choose. The universal primitive is `dispatch`:

```python
specs  = sift.openai_tools()     # the 2 meta-tool function-calling specs
system = sift.system_prompt      # the ~200-token instruction block
# ... your model emits a tool call (name, arguments) ...
result = sift.dispatch(name, arguments)   # runs it, returns a string back
```

Most providers have a ready-made driver so you don't write the loop yourself:

```python
from openai import OpenAI
from sift.adapters.openai import run_agent

run_agent(sift, OpenAI(), "gpt-4o-mini", "what's my last email?")
```

See [Providers](providers.md) for OpenAI-compatible endpoints (incl. local
Ollama/vLLM), native Anthropic, LangChain, and models without native tool-calling.

## Where to go next

- [Building tools](building-tools.md) — everything about defining tools.
- [Discovery & retrieval](discovery.md) — get the routing to pick the right tool.
- [Deployment](deployment.md) — expose SIFT as an MCP or HTTP server.
