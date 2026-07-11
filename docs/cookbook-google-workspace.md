# Cookbook: a heavy MCP (Google Workspace) behind SIFT

The motivating case: one Google Workspace MCP server can inject **~50k tokens of
schemas** into every request — with a dozen tools. This walks through putting it
behind SIFT end-to-end: import → trim → scope → pin → serve. Numbers are
approximate (≈4 chars/token); measure your own with `token_report`.

## 1. Import the MCP server

```python
from sift import Sift
from sift.importers import connect_mcp_stdio

sift = Sift(index_cache="./gw-index.npz")     # warm restarts: no re-embedding

proxy = connect_mcp_stdio(
    sift, "npx", ["-y", "@your/google-workspace-mcp"],   # your server + creds env
    category="google_workspace", service="workspace")
sift.build_index()
```

Every tool is now discoverable through the 2 meta-tools. Descriptions are
sanitized on import (see [security](security.md)). Idle cost per turn: **~430
tokens** instead of the full schema payload.

## 2. Trim the verbose responses

Workspace APIs return deeply nested payloads; the model needs a fraction:

```python
sift.set_response("google_workspace.workspace.gmail_search",
                  transform=lambda r: {"messages": [
                      {"id": m["id"], "subject": m.get("subject", ""),
                       "from": m.get("from", ""), "snippet": m.get("snippet", "")}
                      for m in r.get("messages", [])[:10]]})
sift.set_response("google_workspace.workspace.calendar_list_events",
                  returns=["events"])   # top-level whitelist is often enough
```

`Sift(max_result_chars=...)` (on by default) is the backstop for anything you
didn't trim.

## 3. Check the catalogue before shipping

```python
from sift import quality
print(quality.lint(sift).format())        # missing descs, near-duplicates, bloat
for f in quality.selftest(sift):          # every tool findable by its own phrasing?
    print(f.path, "beaten by", f.beaten_by)
```

## 4. Scope + pin per model

```python
readonly = sift.scope(
    deny=["*.send*", "*.delete*", "*.create*"],
    pin=["google_workspace.workspace.gmail_search"],   # the hot tool: 1 call, no search
)
assistant = sift.scope(allow_risky=True,
                       pin=["google_workspace.workspace.gmail_search",
                            "google_workspace.workspace.calendar_list_events"])
```

Add a confirm gate for the send/delete class:

```python
sift = Sift(on_risky=lambda path, args: my_ui.confirm(path, args), ...)
```

## 5. Serve it

```python
sift.serve_http(scope=readonly, port=8000)   # OpenWebUI tool server, REST
# or: sift.serve_mcp()                       # Claude Desktop and MCP clients
```

## What this buys you (approximate)

| | flat (schemas injected) | behind SIFT |
|---|---:|---:|
| idle turn ("oi") | ~50k tok | ~0.4k tok |
| turn using 1 tool | ~50k tok, 1 call | ~0.4k + 1 TOON line, 2 calls (1 if pinned) |
| adding a tool | every client pays more | fixed surface, index grows server-side |

The flat column is what the OpenWebUI + MCPO setup pays today. Track your real
production numbers with `Sift(observer=quality.GapTracker())` — it also tells
you which needs found no tool (`gaps()`) and what to pin next (`suggest_pins()`).
