# Executing & filtering

Once discovery has surfaced a tool path, the model runs it with `execute_tool`.
This page covers execution, argument handling, response projection, and the
`dispatch` primitive that ties it all together.

## `execute_tool(path, params)`

```python
sift.execute_tool("google_workspace.gmail.read", {"m": 1})
```

Steps SIFT performs:

1. Look up the tool by full path (`KeyError` if it isn't a function path).
2. **Prepare arguments** against the declared params:
   - required param absent or `None` → `ValueError` (an explicit `""` is a value);
   - optional param absent/`None` → its default is used (if declared);
   - values are **coerced to the declared type** — `integer`/`number` (integral
     values stay `int`), `boolean` (`"false"` → `False`), `array`/`object` (JSON
     strings parsed). See [Building tools](building-tools.md#type-coercion-at-the-llm-boundary).
   Only declared params are passed through — unknown keys are ignored.
3. Call your function (`RuntimeError` if no executor is bound — e.g. a
   JSON/imported tool with no function attached).
4. The function must return a `dict` (`TypeError` otherwise).
5. **Project the response** (see below) and return it.

## Response projection

Two owner-configured steps trim what the model sees, in this order:

1. **`transform`** — a callable that reshapes the raw dict (flatten, rename,
   extract). Applied first, to the full result.
2. **`returns`** — a top-level field whitelist. Only these keys survive.

```python
@sift.tool("crm.contacts.get", description="Fetch a contact",
           params={"id": "string:n::contact id"},
           transform=lambda r: {**r["data"], "score": r["meta"]["score"]},
           returns=["name", "email", "score"])
def get_contact(id): ...
```

Projection is a **token saver** (a 20-field record → the 3 fields that matter) and
a **safety boundary** (a `body`, `raw`, or `secret` field the model never needs is
never sent).

### Configure projection after the fact — `set_response`

You don't need to own the `@tool` definition to trim a result. `set_response`
works on **any** registered tool, including imported MCP/OpenAPI ones:

```python
sift.set_response("google_workspace.gmail.read", returns=["id", "subject", "from"])
sift.set_response("integrations.github.list_issues",
                  transform=lambda r: {"issues": [i["title"] for i in r["items"]]})
```

This is the main lever for making verbose upstream tools cheap — see
[Importing ecosystems](importing.md).

## `dispatch` — the single entry point

`dispatch(name, arguments)` runs whichever meta-tool call a model emitted and
returns a **string** (TOON for search, JSON for execute). It's format-agnostic —
`arguments` may be a dict or a JSON string — which is exactly why every provider
adapter is a thin wrapper over it.

```python
sift.dispatch("search_tools", {"domain": "email", "action": "read latest"})
sift.dispatch("execute_tool", {"path": "google_workspace.gmail.read", "params": {"m": 1}})
sift.dispatch("run_code", {"code": "output = call('...', m=1)"})   # code mode
```

Names handled:

| name | behaviour |
|---|---|
| `search_tools` | active request (`domain`/`action`) → else query (`q`) → else browse (`path`) |
| `execute_tool` | run `path` with `params`, project, return JSON |
| `run_code` | run a snippet in the sandbox (see [Code mode](code-mode.md)) |
| `get_tool_schema` | deprecated alias — browse a level (kept for back-compat) |

**Errors are returned, not raised — and they point at the fix.** Any exception
becomes `{"error": "..."}` as the tool result; path problems additionally carry a
`hint` with the recovery move, because weak models read a bare error, conclude
"I don't have that tool" and give up:

```python
sift.dispatch("execute_tool", {"path": "nope.nope.nope"})
# → {"error": "unknown tool 'nope.nope.nope'",
#    "hint": "call search_tools (with q, or domain+action) to discover the correct tool path, ..."}

sift.dispatch("execute_tool", {"tool": "x"})   # wrong argument key
# → {"error": "execute_tool requires a 'path' argument", "hint": "call search_tools ..."}
```

A missing `path` is named as such (never misreported as a permission problem),
and typed-param failures are structured (`parameter 'a': expected an integer,
got 'x'`) so the model can retry with a corrected value.

## Risky tools: the `on_risky` confirmation hook

`risk=True` marks a tool; `on_risky` decides what "risky" means at runtime — the
standard human-in-the-loop confirm, as a hook instead of app plumbing:

```python
def confirm(path: str, args: dict) -> bool:
    return ask_the_user(f"Run {path} with {args}?")   # block/prompt however you like

sift = Sift(on_risky=confirm)
```

Called with the **prepared** (coerced) args right before execution, only for
`risk=True` tools. Return `False` (or raise) to block — the model receives
`risky tool '...' was not confirmed (blocked by the on_risky guard)`. Combine
with `scope(allow_risky=False)` when a model should never even see risky tools.

## Result caps

Anything `dispatch`/`adispatch` returns to the model is capped at
`Sift(max_result_chars=100_000)` (default; `None` disables). Oversized results
are truncated with a marker telling the model the result was cut and pointing
the owner at `set_response(returns=/transform=)` — so a tool that suddenly
returns 1 MB can't silently flood the context.

## Async execution

```python
await sift.aexecute_tool("google_workspace.gmail.read", {"m": 1})
await sift.adispatch("execute_tool", {"path": ..., "params": ...})
```

`async def` tools are awaited natively; calling one through the sync
`execute_tool` raises a `TypeError` that points here. In `adispatch`,
`run_code` is moved to a worker thread (the sandbox may block on a subprocess);
search/browse are cheap and run inline.

## Observability

```python
sift = Sift(observer=lambda event, data: my_tracer.emit(event, **data))
# events: "search"  {"q"/"domain"/"action", "ms"}
#         "execute" {"path", "ok", "ms", "error"?}
#         "run_code" {"ok", "ms"}
```

Observer exceptions are swallowed (never break the tool loop). The same points
log at DEBUG level under the `"sift"` stdlib logger.
