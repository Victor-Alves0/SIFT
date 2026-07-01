# Code mode & the sandbox

For composite tasks, one tool call per turn is wasteful. **Code mode** lets the
model write a short Python snippet that orchestrates many tools in a **single
turn** (the StackOne/Cloudflare "code mode" pattern), collapsing multi-turn
overhead.

## Using code mode

```python
tools  = sift.code_tools()          # the code-mode surface: search_tools + run_code
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
  and the `str.format`/`format_map` escape.
- **Line budget** via `sys.settrace` caps executed lines (kills infinite loops
  that the AST can't see).
- **Restricted builtins** — only a safe subset is exposed.

Fast, no process overhead. Use it for **catalogues you trust** (your own tools).
It raises the bar but shares your process, so treat it as a guardrail, not a jail.

### SubprocessSandbox (isolated)

Runs the snippet in a **separate process** (`python -m sift._sandbox_child`):

- The child holds **no** references to your tools or memory. When the snippet does
  `call(...)`/`search(...)`, the request is **proxied over stdio back to the
  parent**, which executes the real (trusted) tool and returns the filtered result.
- A **wall-clock watchdog** kills the child on `timeout` — catching C-level hangs
  a Python line budget can't observe (e.g. `sum(range(10**9))`).
- On Unix, **CPU and memory rlimits** (`cpu_seconds`, `memory_mb`) are applied.
- The same AST/line-budget policy still runs inside the child.

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
