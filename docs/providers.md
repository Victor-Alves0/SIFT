# Providers — wire SIFT to any model

SIFT never calls a model itself. It hands you **2 tool specs** + a **system
prompt**, and `sift.dispatch(name, args)` runs whatever tool call the model
emits. That makes it provider-agnostic. This page shows each path.

| Path | How | Needs |
|---|---|---|
| OpenAI-compatible (OpenAI, OpenRouter, DeepSeek, Together, Groq, Mistral, **local** Ollama/LM Studio/vLLM) | `adapters.openai.run_agent` or `openai_tools()` + `dispatch` | `[openai]` |
| Native Anthropic (Messages API) | `adapters.anthropic.run_agent` | `[anthropic]` |
| LangChain (any LangChain-supported model) | `langchain_tools()` | `[langchain]` |
| MCP clients (Claude Desktop, IDEs) | `serve_mcp()` | `[mcp]` |
| **No native tool calling** (base/small models) | `adapters.prompted` | — |
| Anything else | `openai_tools()` + `system_prompt` + `dispatch` | — |

## OpenAI-compatible (incl. local models)

The `openai` SDK speaks to far more than OpenAI — point `base_url` anywhere:

```python
from openai import OpenAI
from sift.adapters.openai import run_agent

client = OpenAI()                                                         # OpenAI
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)     # OpenRouter
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")   # Ollama (local)

run_agent(sift, client, "gpt-4o-mini", "what's my last email?",
          extra_body={"reasoning": {"effort": "low"}})   # provider extras pass through
```

`run_agent` drives the full loop (search → execute → answer) until the model
returns a final message. The client is duck-typed — anything exposing
`chat.completions.create(...)` works, which is why a fake client can unit-test it.

## Native Anthropic

Anthropic's tool format differs (uses `input_schema`, system is a separate arg);
this adapter bridges it:

```python
import anthropic
from sift.adapters.anthropic import run_agent

run_agent(sift, anthropic.Anthropic(), "claude-haiku-4.5", "what's my last email?")
```

`sift.anthropic_tools()` gives the raw specs if you want to drive the loop yourself.

## LangChain

```python
tools = sift.langchain_tools()   # [search_tools, execute_tool] as StructuredTools
# plug `tools` into any LangChain agent / graph
```

The `search_tools` StructuredTool accepts `q`, `path`, `domain`, and `action`, so
LangChain agents get the full discovery surface (query, browse, active request).

## Models without native tool-calling

`dispatch` is text-based, so any model that can emit JSON can drive SIFT via a
prompted protocol — no function-calling API required. You supply one callable,
`generate(prompt) -> str`, wrapping *anything* (HuggingFace pipeline, llama.cpp,
Ollama `/generate`, a base model):

```python
from sift.adapters.prompted import run_agent, single_decision

def generate(prompt: str) -> str:
    return my_model(prompt)

run_agent(sift, generate, "what's my last email?")      # full text tool loop
single_decision(sift, generate, "read my last email")   # 1 decision, weakest models
```

`single_decision` searches server-side first, then asks the model for a single
`{path, args}` choice — the most robust path for very small models.

SIFT's tiny 2-tool surface actually *helps* weak models (less to get lost in).
Realistic floor is ~1–3B params; sub-1B models can be interfaced but rarely
follow the format reliably.

### Constrained decoding

For local decoders, force output to be parseable:

```python
sift.tool_call_schema()   # JSON Schema → Outlines / LM Format Enforcer / vLLM guided_json
sift.json_gbnf()          # GBNF grammar → llama.cpp
```

The JSON Schema constrains each step to `{"tool": ..., "args": {...}}` or
`{"answer": "..."}`; the GBNF constrains output to valid JSON. Pair either with
`adapters.prompted`.

## The universal primitive

Under every adapter is the same three-line contract — use it directly with any SDK:

```python
specs  = sift.openai_tools()          # 2 function-calling specs
system = sift.system_prompt           # instruction block
# ... model emits a tool call (name, arguments as dict or JSON string) ...
result = sift.dispatch(name, arguments)   # → string result to feed back
```
