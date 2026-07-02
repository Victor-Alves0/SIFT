# Scoping — per-model `allowedTools`

Build the catalogue **once**, then hand each model (or session, or user) a
**scoped view** that only sees and can run a subset. This is the OpenWebUI "pick
tools for this model" pattern — and the shared search index is reused, so a scope
is cheap (no rebuild).

```python
sift = Sift(); ...; sift.build_index()

view = sift.scope(
    allow=["google_workspace.gmail.*", "web.search.*"],  # globs over the dotted path
    deny=["*.delete", "*.send"],                          # deny always wins
    allow_risky=True,                                     # or False to block all risk=True
)
```

## How matching works

Patterns are globs (`fnmatch`) over the full dotted path:

- `"google_workspace.*"` — a whole category
- `"*.gmail.read"` — a specific function across categories
- `"crm.contacts.delete"` — one exact path
- `"*.delete"` — every delete function

Resolution order for a path:

1. `allow` — if set, the path must match at least one allow glob (unset ⇒ allow all).
2. `deny` — if the path matches any deny glob, it's rejected. **Deny wins.**
3. `allow_risky=False` — additionally rejects any tool flagged `risk=True`.

## It's enforced on both halves

Scoping filters **discovery** *and* guards **execution** — a model can neither see
nor run a tool outside its scope, in every mode:

```python
view.dispatch("search_tools", {"q": "delete a contact"})   # deny'd tools never appear
view.dispatch("search_tools", {"domain": "crm", "action": "remove a contact"})  # same
view.dispatch("search_tools", {"path": "crm.contacts"})    # browse: denied tools hidden too
view.execute_tool("crm.contacts.delete", {})               # → PermissionError
view.dispatch("execute_tool", {"path": "crm.contacts.delete"})  # → {"error": "...not allowed..."}
view.run_code("output = call('crm.contacts.delete')")      # allow/deny applies inside code too
```

Browsing the hierarchy is scoped as well: a denied function never appears in a
service listing, its direct path returns an error instead of a schema, and a
category/service whose tools are *all* denied is omitted from listings entirely —
denial means no disclosure, not just no execution. (One nuance: synthesized
category/service descriptions are built from all children at registration time,
so give sensitive nodes an explicit `describe(...)` if even description text
must not hint at hidden tools.)

## Wiring a scope into a model

A `SiftScope` exposes the same surface as `Sift`, so it drops into any adapter:

```python
view.openai_tools()      # same 2 specs
view.anthropic_tools()
view.langchain_tools()   # (via the parent)
view.system_prompt
view.dispatch(name, args)
```

Give each model its own `view` and they safely share one catalogue and one index.

## With a deployed server

Pass a scope when serving so an HTTP/MCP endpoint exposes only a subset:

```python
sift.serve_http(scope=view)          # this server only sees the scoped tools
```

See [Deployment](deployment.md). Combine with per-tool [response projection](executing-and-filtering.md#response-projection)
to control both *which* tools a model can use and *what* each one returns.
