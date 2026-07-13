# Code mode & the sandbox

For **composite** tasks, one tool call per turn is wasteful. **Code mode** lets the
model write a short Python snippet that orchestrates many tools in a **single
turn** (the CodeAct / Cloudflare "code mode" pattern), collapsing multi-turn
overhead.

## When code mode is NOT the answer — start here

**We benchmarked code mode against classic tool calling and code mode lost.** On a
100-tool catalogue with a live model: classic 3.1 turns / 6.5k effective tokens vs
code mode 3.3 turns / 8.4k, both at 100% success
([results](../benchmarks/RESULTS.md#code-mode-vs-classic-tool-calling)).

The reason is worth internalising before you reach for this mode. Code mode's headline
pitch is "collapse N round-trips into one". But modern function calling emits
**parallel tool calls** — on our N+1 task the classic condition ran six CRM lookups in
a *single turn*, no sandbox, no Python. **The industry's main argument for code mode is
already free.**

What parallel calls genuinely *cannot* do — and where code mode still earns its keep:

- **Filtering a huge result before it reaches the context.** A parallel call still
  puts its whole payload in the conversation. A snippet can reduce 10,000 rows to 5.
- **Data-dependent calls** — call B's arguments come from call A's *result*.
- **Real control flow** — conditionals, retries, early exit.

Code mode also is not free: it needs a sandbox, it is harder to debug than a clean list
of tool calls, and the model can emit Python that doesn't compile. For a **single tool
call**, writing code is pure overhead — a plain call cannot fail to parse.

That is why `code_tools()` exposes **three** tools, not two:

| Situation | Use | Why |
|---|---|---|
| One call answers it ("what time is it?") | `execute_tool` | nothing to compile, nothing to sandbox |
| 2+ calls, a loop, a conditional | `run_code` | collapses N turns into 1 |
| A big result you only need a slice of | `run_code` | filter in the sandbox, not in the context |

A code-mode surface without `execute_tool` forces Python for every request — which
is how a "what's today's date?" turn ends up costing thousands of tokens.

## Using code mode

```python
tools  = sift.code_tools()          # search_tools + execute_tool + run_code
system = sift.code_system_prompt    # instructions for writing snippets
```

Inside a `run_code` snippet the model has three helpers and assigns its result to
a variable named `output`:

```python
sift.run_code("""
msgs = search('unread email')            # → list of tool paths
inbox = call('google_workspace.gmail.read', m=5)   # execute a tool, get filtered dict
output = {'subjects': [inbox['subject']]}
""")
```

| Helper | Signature | Does |
|---|---|---|
| `call` | `call(path, **params) -> dict` | execute a tool; returns its **filtered** result |
| `search` | `search(q, top_k=5) -> list[str]` | discovery; returns matching tool paths |
| `schema` | `schema(path) -> str` | TOON schema of a tool/level |

`run_code` returns JSON: `{"output": ...}` (your `output` var),
`{"stdout": ...}` (if the snippet printed but set no `output`), or
`{"error": ...}`. There's a tool-call budget (`max_calls`, default 50) so a
runaway loop can't hammer your tools.

`call`/`search` route through the same object you invoke `run_code` on, so on a
[scope](scoping.md) they obey that scope's allow/deny — even inside a snippet.

## Keep `output` small — this is where the savings actually come from

The headline number people quote for code execution (Anthropic reports 150k → 2k
tokens on a large-catalogue task, ~98.7%) comes from **two** things. SIFT gives you
the first for free — the model never loads the full catalogue, it searches. The
second is on your prompt and the model:

> Intermediate values stay in the sandbox for free. Everything you put in `output`
> is re-sent with the entire conversation on **every later turn**.

So a snippet that does `output = call('gmail.search', q='...')` and dumps 17 KB of
raw email bodies into the context has thrown the win away. Filter first:

```python
msgs = call('mail.gmail.search', q='from:boss', m=100)   # 100 messages, in the sandbox
output = [{'id': m['id'], 'subject': m['subject']} for m in msgs][:5]   # 5 rows, in the context
```

`CODE_SYSTEM_PROMPT` states this rule. Enforce it at your boundary too: `returns=`
on each tool (response filtering), `max_result_chars`, and the `on_result` hook.

## Never fail silently

In tool calling, **every wasted round-trip re-sends the whole context** — a
snippet that comes back empty is not a small cost, it is a full turn. So code mode
is built so the model can always recover *from the result itself*:

- **A bare last expression is promoted to `output`**, exactly like a REPL. Models
  routinely end a snippet with the value they mean to return
  (`[m["id"] for m in msgs]`) instead of assigning it. That is not ambiguity — so
  it is honoured, not punished. An explicit `output = …` always wins, and a
  trailing `print(...)` is left alone (stdout already carries it).
- **A policy violation states the policy.** `import datetime` returns the error
  *and* the sandbox rules (`hint`), so the model fixes it on the next try instead
  of guessing. Those rules are generated **from** the enforcement code
  (`sift.sandbox.SANDBOX_RULES`), so they cannot drift out of date.
- **A snippet that produces nothing says so.** If nothing is assigned and nothing
  is printed, `run_code` returns an `error` — not a hollow `{"stdout": ""}` that
  reads as success and teaches the model nothing.
- **…but "empty" is not "failed".** If the snippet *did* set `output` and the value
  is simply empty, you get `{"output": null}`. And when a no-result snippet had
  already executed tools, the error carries a `ran` field naming how many —
  because a retry must never silently re-send an email.

```jsonc
{"error": "no result: nothing was assigned to `output` and nothing was printed",
 "hint":  "assign what you want back, e.g. output = call('some.tool.path', arg=1) …",
 "ran":   "1 tool call(s) already executed — they were NOT undone; do not repeat
           any that have side effects"}
```

## The pluggable sandbox

Snippets execute through a **pluggable backend**, chosen per `Sift`:

```python
from sift.sandbox import InProcessSandbox, SubprocessSandbox

Sift(sandbox=InProcessSandbox())                # default
Sift(sandbox=SubprocessSandbox(timeout=10))     # isolated process
```

### InProcessSandbox (default)

Runs the snippet in-process behind a static + dynamic policy:

- **AST guard** rejects, before running: `import`, dunder/private attribute access
  (`__class__`, `_foo`), dangerous names (`eval`, `exec`, `open`, `getattr`, …),
  class definitions, and the `str.format`/`format_map` escape.
- **Line budget** via `sys.settrace` caps executed lines of the *snippet* (kills
  infinite loops the AST can't see; lines run inside real tool implementations
  are neither counted nor traced).
- **Restricted builtins** — only a safe subset is exposed.

Known in-process gap: a single-expression memory bomb (e.g. `'a' * 10**9`) is one
"line" and allocates before any guard fires — another reason to use the
subprocess backend (with `memory_mb`) for anything you don't fully trust.

Fast, no process overhead. Use it for **catalogues you trust** (your own tools).
It raises the bar but shares your process, so treat it as a guardrail, not a jail.

### SubprocessSandbox (isolated)

Runs the snippet in a **separate process** (`_sandbox_child.py`, launched as a
plain script so the child never imports the `sift` package — or numpy — at all):

- The child holds **no** references to your tools or memory. When the snippet does
  `call(...)`/`search(...)`, the request is **proxied over stdio back to the
  parent**, which executes the real (trusted) tool and returns the filtered result.
- The child gets a **scrubbed environment** (a minimal allowlist: PATH etc.) —
  parent API keys and other secrets never reach the process running untrusted code.
- A **wall-clock watchdog** kills the child on `timeout` — catching C-level hangs
  a Python line budget can't observe (e.g. `sum(range(10**9))`). The clock
  **pauses while the parent runs a proxied tool**: the timeout budgets the
  untrusted snippet, not your own tools — a deep-search tool slower than the
  timeout completes normally instead of being killed with its result discarded.
- On Unix, **CPU and memory rlimits** (`cpu_seconds`, `memory_mb`) are applied.
- The same AST/line-budget policy still runs inside the child. If the child dies
  unexpectedly, the tail of its stderr is surfaced in the error for diagnosis.

```python
SubprocessSandbox(timeout=10, max_lines=200_000, cpu_seconds=10, memory_mb=512)
```

### Security model — be honest about the boundary

`SubprocessSandbox` is a large step up (process isolation, no parent access,
resource caps) **but it is not a VM**. On its own it does not block network or
filesystem syscalls from within the child. For **fully untrusted** snippet input,
run it inside OS-level isolation (a container, seccomp/gVisor, a locked-down
user). The in-process backend, likewise, is a policy guard — not a security
boundary against a determined adversary.

Rule of thumb:

| Snippet source | Backend |
|---|---|
| Your own tools, your own prompts | `InProcessSandbox` (default) |
| Semi-trusted / third-party prompts | `SubprocessSandbox` |
| Fully untrusted input | `SubprocessSandbox` **inside** a container/seccomp |

## Against the frontier — what SIFT does, and what it doesn't

Anthropic's *Code execution with MCP* names six patterns. Where SIFT stands, without
flattering itself:

| Pattern | SIFT |
|---|---|
| **Progressive disclosure** — load tool definitions on demand, not up front | ✅ this *is* SIFT. Their version reads a filesystem; SIFT searches a hybrid index, so the model doesn't have to guess directory names. |
| **Context-efficient results** — filter in the execution environment | ✅ `returns=` filtering, `max_result_chars`, `on_result`, and (since 0.8.0) a prompt that actually asks for it. |
| **Control flow in code** — loops/conditionals instead of chained calls | ✅ |
| **Privacy-preserving** — intermediates never enter the context | ✅ by construction: only `output` leaves the sandbox. |
| **State persistence** — carry results across snippets | ❌ **not supported.** Every `run_code` gets a fresh namespace. Compose within one snippet; SIFT will not pretend to have a session heap it does not have. |
| **Reusable skills** — save a working snippet as a callable | ❌ **not supported.** Register it as a real tool instead: `@sift.tool` — it then gets a schema, response filtering, risk flags and retrieval, which an ad-hoc saved snippet does not. |

The two gaps are deliberate for now. Both are real capabilities (they pay off in
long autonomous sessions), and both would put mutable, model-authored state inside
the trust boundary — which is a different security posture than the one
[security.md](security.md) currently promises. If you need them today, keep the state
in *your* tools (a `kv.get`/`kv.set` pair is a tool like any other) rather than in
the sandbox.
