# Building tools

A tool is a Python callable registered at a dotted path with a compact schema.
This page covers every way to define one.

## The hierarchy: `category.service.function`

Every tool lives at a **three-level** dotted path:

```
google_workspace . gmail . read
└── category      └ service └ function
```

- **category** — a broad area (`google_workspace`, `crm`, `web`, `local`).
- **service** — a product/integration within it (`gmail`, `calendar`).
- **function** — the operation (`read`, `send`, `search`).

The path must have exactly two dots or registration raises `ValueError`. The
model discovers tools by walking these levels, so name them the way a user would
describe the need (`email`, `calendar`) — good names improve retrieval.

## The `@tool` decorator

```python
@sift.tool(
    "google_workspace.gmail.send",
    description="Send a new email message",
    params={
        "to":      "string:n::recipient address",
        "subject": "string:n::subject line",
        "body":    "string:n::message body",
    },
    returns=["id", "status"],   # only these fields reach the model
    risk=True,                  # high-impact action (send/delete) → confirm first
)
def gmail_send(to, subject, body):
    return {"id": "sent_1", "status": f"sent to {to}", "raw": {...}}  # "raw" dropped
```

Arguments:

| Arg | Meaning |
|---|---|
| `path` | `category.service.function` (positional, required) |
| `description` | one line; this is the primary text used for retrieval |
| `params` | a dict of parameter specs (see below); defaults to `{}` |
| `returns` | response whitelist — only these top-level fields survive `execute_tool` |
| `risk` | `True` marks a high-impact action, surfaced as `|risk` in the schema |
| `transform` | a callable to reshape the raw result before whitelisting (see below) |
| `examples` | optional "how a user asks for this" phrasings — indexed for retrieval |
| `replace` | registering an existing path raises unless `replace=True` (no silent shadowing) |

The wrapped function must **return a `dict`** (a `TypeError` is raised otherwise).
`async def` tools are supported — execute them via `aexecute_tool`/`adispatch`.

**`params=` omitted? The spec is derived from the signature.** Annotations become
types (`a: int` → `integer`), a missing default means required:

```python
@sift.tool("demo.math.add", description="add two numbers")
def add(a: int, b: int, precise: bool = False):   # → a:integer:n, b:integer:n,
    return {"sum": a + b}                          #   precise:boolean:o:False
```

Prefer explicit `params=` for real catalogues (derived specs have no
descriptions, which retrieval and the model both benefit from) — but a bare
registration is callable, not a trap.

## Parameters

### Compact string form

`"<type>:<req>:<default>:<description>"`

- **type** — `string`, `number`, `integer`, `boolean`, `array`, `object` (see the
  coercion table below).
- **req** — `n` or `r` = required, `o` (or empty) = optional. Any other flag
  raises at registration — a typo must not silently turn a required param optional.
- **default** — used when the param is omitted (optional params only).
- **description** — free text; may contain colons (it's the last field).

```python
params={
    "q": "string:o::search query",       # optional, no default
    "m": "number:o:10:max results",      # optional, default 10
    "to": "string:n::recipient",         # required
}
```

> The `default` field in the string form **cannot contain a colon** (the parser
> splits on `:`). For a default like `is:unread`, use the structured form.

### Structured dict form

Use this when a default contains `:` (e.g. a Gmail query) or you just prefer
explicit keys:

```python
params={
    "q": {"type": "string", "default": "is:unread", "desc": "Gmail search query"},
    "m": {"type": "number", "required": False, "default": 10, "desc": "max results"},
}
```

Keys: `type`, `required` (bool), `default`, `desc` (or `description`). TOON quotes
colon-bearing defaults so the one-line schema stays unambiguous
(`q:string:o:'is:unread'`).

### Type coercion at the LLM boundary

Models routinely send every argument as a string (`"3"`, `"false"`, `"[1,2]"`),
so SIFT coerces values to the declared type before your function sees them:

| declared type | coercion |
|---|---|
| `string` | passed through |
| `integer` (`int`) | → `int` (`"7"` → `7`, `7.9` → `7`) |
| `number` (`float`) | → `float`, but **integral values stay `int`** (`"3"` → `3`, `"3.5"` → `3.5`) — so slicing/pagination with the value works |
| `boolean` (`bool`) | `"true"/"1"/"yes"/"on"` → `True`; `"false"/"0"/"no"/"off"/""` → `False` |
| `array` / `object` | JSON strings are parsed (`"[1,2]"` → `[1, 2]`); native values pass through |

An **unparseable value raises a clean, named error** (`parameter 'a': expected an
integer, got 'x'`) surfaced to the model as a structured tool error it can retry
from. Letting plausible garbage through (`'x' * 4 == 'xxxx'`) propagates
hallucination — SIFT fails loudly at the boundary instead.

**Missing vs empty.** Only an *absent* or `None` argument counts as missing (a
required one raises `ValueError`; an optional one gets its default). An explicit
`""` is a real value — a model can override a non-empty default with an empty
string.

## `returns` — response projection (a whitelist)

`returns` trims the tool's result to just those top-level fields before it reaches
the model. This is a token saver *and* a safety boundary — a `body`/`raw`/
`secret` field the model never needs simply never gets sent:

```python
returns=["id", "subject", "from"]   # a 20-field Gmail message → 3 fields
```

If `returns` is empty, the full dict is returned. See
[Executing & filtering](executing-and-filtering.md#response-projection) for
`transform` and `set_response` (which also work on imported tools).

## `risk` — high-impact actions

Set `risk=True` for anything that sends, deletes, pays, or otherwise can't be
undone. It surfaces as a trailing `|risk` marker in the tool's TOON schema, the
system prompt tells the model to proceed only if the user authorised it, and it
integrates with [scoping](scoping.md) (`allow_risky=False` blocks all risky tools
at once).

## `transform` — reshape before whitelisting

A callable applied to the raw result *before* the `returns` whitelist. Handy for
flattening a verbose upstream response (great for imported MCP/OpenAPI tools):

```python
@sift.tool("google_workspace.gmail.query", description="Search messages",
           params={"q": "string:n::query"},
           returns=["ids"],
           transform=lambda r: {"ids": [m["id"] for m in r["messages"]]})
def gmail_query(q):
    return {"messages": [{"id": "1", ...}, {"id": "2", ...}], "nextPageToken": "..."}
# model sees: {"ids": ["1", "2"]}
```

## Alternatives to the decorator

### `add_tool` — register an existing function

```python
def gmail_read(m=10): ...
sift.add_tool("google_workspace.gmail.read", gmail_read,
              description="Read emails", params={"m": "number:o:10:max"},
              returns=["id", "subject"])
```

`add_tool` returns the `Sift` instance, so calls chain.

### Describe categories and services

Categories and services get a synthesised description from their children by
default. Override it to improve browse output and retrieval:

```python
sift.describe("google_workspace", "Google Workspace: email, calendar, drive")
sift.describe("google_workspace.gmail", "Gmail: read, search and send email")
```

### Load a whole catalogue from JSON

For large or externally-managed catalogues, load the nested JSON format
(`category → services → fns`) directly:

```python
from sift.registry import Registry
sift = Sift(registry=Registry.from_json("registry.json"))
```

```json
{
  "google_workspace": {
    "d": "Google Workspace",
    "services": {
      "gmail": {
        "d": "Gmail",
        "fns": {
          "read": {"d": "Read emails", "p": {"m": "number:o:10:max"},
                   "r": ["id", "subject"], "risk": false}
        }
      }
    }
  }
}
```

Tools loaded from JSON have **no executor bound** — they're discoverable, but
`execute_tool` raises until you bind a function with `registry.bind(path, fn)` or
`sift.add_tool(...)`. Importers (OpenAPI/MCP) do this binding for you — see
[Importing ecosystems](importing.md).

## After registering: build the index

```python
sift.build_index()
```

Call it **once**, after all tools are registered. It flattens the hierarchy into
searchable entries and builds the embedding and/or BM25 indexes. Registering more
tools after building requires another `build_index()` to pick them up.
