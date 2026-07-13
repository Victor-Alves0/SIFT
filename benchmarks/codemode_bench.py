"""Code mode vs classic tool calling — does writing code actually pay?

The claim SIFT makes in 0.8.0 is specific, and until now it was an argument, not a
measurement:

  * code mode wins on COMPOSITE work (many calls, a loop, filtering a big result),
  * classic tool calling wins on a SINGLE call (no code to write, nothing to compile),
  * so a code-mode surface must expose BOTH (`execute_tool` + `run_code`) and let the
    model choose — exposing only `run_code` taxes every single-call request.

This measures all three against a live model, on tasks of both shapes.

It also instruments the two failure modes 0.8.0 fixed, by inspecting every snippet
the model actually writes:

  * ``shim_saves``  — snippet ended in a bare expression and bound no ``output``.
                      Before 0.8.0 this returned {"stdout": ""} — a hollow success:
                      one WASTED ROUND-TRIP, i.e. one whole context re-sent.
  * ``import_tries`` — snippet tried an ``import``. Before 0.8.0 the error didn't say
                      what was allowed instead, so the model guessed again.
  * ``no_result``   — snippet produced nothing at all (not even a trailing expression).

Those counts are a direct, honest price tag for the old behaviour.

Usage (repo root, package venv):
    python benchmarks/codemode_bench.py
    SIFT_BENCH_TASKS=4 python benchmarks/codemode_bench.py     # quick

Reads OPENROUTER_API / SIFT_MODEL from .env.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from sift import Sift
from sift.agentbench import CACHE_READ_WEIGHT, synth_distractors
from sift.codemode import CODE_SYSTEM_PROMPT, code_tool_specs
from sift.metatools import SYSTEM_PROMPT, tool_specs
from sift.sandbox import _binds_output

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
TASK_LIMIT = int(os.environ.get("SIFT_BENCH_TASKS", "0") or 0)
CATALOG_SIZE = int(os.environ.get("SIFT_BENCH_SIZE", "100"))

# The code-mode surface as it shipped in 0.7.0: no execute_tool, and a prompt that
# never mentioned the sandbox's limits or the cost of a fat `output`.
LEGACY_CODE_PROMPT = """You orchestrate tools by WRITING CODE, composing many calls in a single turn.

Tools:
1. search_tools(q) — find tool paths (schema inline) by need.
2. run_code(code)  — runs Python NOW. Inside it you have:
     call(path, **params) -> dict   # execute a tool, returns its filtered result
     search(q) -> [paths]           # discovery; returns matching tool paths
     schema(path) -> str            # TOON schema of a tool/level
   Assign the data you want back to a variable named `output`.

Flow: optionally call search_tools to learn paths, then ONE run_code that performs
all the tool calls and sets `output`. Then answer the user concisely.
"risk" tools (send/delete): only run them if the user authorised it."""


# --------------------------------------------------------------------- catalogue
# Deliberately shaped like the app that reported the field bugs: tools whose results
# are FAT (email bodies, file listings). A tool result that is small to begin with
# cannot show the difference between filtering in the sandbox and filtering in the
# context — which is the whole point of code mode.

_BODY = ("Hi team, following up on the quarterly numbers. " * 40)   # ~2 KB per message


def _messages(m: int, domain: str = "acme.com") -> list[dict]:
    """Headers only — the BODY is behind mail.gmail.read(id), like every real mail API.
    (A search that already leaks the bodies makes the read redundant and the fan-out
    task meaningless — the first version of this benchmark did exactly that.)"""
    return [{"id": f"m{i}", "from": f"user{i}@{domain if i % 2 == 0 else 'other.org'}",
             "subject": f"Re: report {i}", "date": "2026-07-13", "snippet": _BODY[:200]}
            for i in range(m)]


# Distractor domains that would SHADOW a gold tool rather than distract from it.
# `calendar.events.list` does the same job as the gold `cal.google.list`; a benchmark
# that cannot say which one is correct is not a hard benchmark, it is an invalid one.
# (quality.selftest() catches exactly this — build_sift() gates on it below.)
_SHADOWING = {"calendar", "crm"}


def build_sift(size: int) -> Sift:
    s = Sift()

    @s.tool("mail.gmail.search", description="Search and list email messages matching a query",
            params={"q": "string:o::search query", "m": "integer:o:20:how many messages"},
            returns=["messages"],
            examples=["find emails from my boss", "show me my latest emails"])
    def _search(q="", m=20):
        return {"messages": _messages(int(m))}

    # search returns ids + headers; the BODY only comes from read(id). This is the
    # shape an integrator reported: the model reads N ids one execute_tool at a time
    # instead of looping them inside a single run_code.
    @s.tool("mail.gmail.read", description="Read the full body of one email message by its id",
            params={"id": "string:n::the message id"}, returns=["id", "subject", "body"])
    def _read(id):
        return {"id": id, "subject": f"Re: report {id}",
                "body": _BODY + (" QUARTERLY RESULTS ATTACHED." if id in ("m1", "m3") else "")}

    @s.tool("mail.gmail.send", description="Send a new email message to a recipient",
            params={"to": "string:n::recipient", "subject": "string:n::subject",
                    "body": "string:n::body"},
            returns=["id"], risk=True)
    def _send(to, subject, body):
        return {"id": "sent-1"}

    @s.tool("mail.gmail.unread_count", description="How many unread emails are sitting in the inbox",
            params={}, returns=["count"], examples=["how many unread do I have"])
    def _unread():
        return {"count": 7}

    @s.tool("cal.google.list", description="List calendar events for a given day",
            params={"day": "string:o:today:the day, e.g. 2026-07-13"},
            returns=["events"], examples=["what's on my calendar today"])
    def _events(day="today"):
        return {"events": [
            {"id": f"e{i}", "title": f"Meeting {i}", "start": f"{9+i}:00",
             "organizer": f"user{i}@acme.com", "notes": _BODY} for i in range(6)]}

    @s.tool("crm.contacts.get", description="Get a CRM contact record by email address",
            params={"email": "string:n::the contact's email"},
            returns=["email", "name", "account"])
    def _contact(email):
        known = email.endswith("@acme.com") and email[4] in "024"
        return {"email": email, "name": "Known" if known else "", "account": "ACME" if known else ""}

    @s.tool("files.drive.search", description="Search files in Drive by name",
            params={"q": "string:n::name to search for"}, returns=["files"])
    def _files(q):
        return {"files": [{"name": f"{q}-{i}.pdf", "size": 1000 * (i + 1), "owner": "me"}
                          for i in range(8)]}

    @s.tool("utils.time.now", description="Get the current date and time",
            params={}, returns=["iso"], examples=["what's today's date"])
    def _now():
        return {"iso": "2026-07-13T10:00:00"}

    real = {t.path for t in s.registry.tools()}
    for d in synth_distractors(size * 3):
        if d.path in real or d.path.split(".")[0] in _SHADOWING:
            continue
        s.registry.add(d)
        real.add(d.path)
        if len(real) >= size:
            break
    s.build_index()

    # Gate the benchmark on the instrument we ship: if a gold tool cannot be found
    # by its OWN description, the catalogue is broken and any number we print is
    # measuring that, not code mode.
    from sift.quality import selftest
    gold = {p for t in TASKS for p in t.gold}
    bad = [f for f in selftest(s) if f.path in gold]
    if bad:
        raise SystemExit("benchmark catalogue is invalid — gold tools are shadowed:\n" +
                         "\n".join(f"  {f.path} beaten by {f.beaten_by} on {f.query!r}"
                                   for f in bad))
    return s


# --------------------------------------------------------------------- tasks

@dataclass
class Task:
    query: str
    gold: set[str]     # every tool that must have executed
    shape: str         # "single" | "composite"


TASKS: list[Task] = [
    # one call answers it — writing Python here is pure overhead
    Task("What's today's date?", {"utils.time.now"}, "single"),
    Task("How many unread emails do I have?", {"mail.gmail.unread_count"}, "single"),
    Task("Who is the CRM contact for user0@acme.com?", {"crm.contacts.get"}, "single"),
    Task("List my calendar events for today.", {"cal.google.list"}, "single"),
    Task("Find files in Drive named 'budget'.", {"files.drive.search"}, "single"),

    # composite — several calls, a loop, or a big result that must be reduced
    Task("Of my 30 most recent emails, how many are from acme.com? Just the number.",
         {"mail.gmail.search"}, "composite"),
    Task("For each of today's calendar events give me ONLY the title and start time.",
         {"cal.google.list"}, "composite"),
    Task("For every organizer of today's events, check whether they exist in the CRM, "
         "and tell me which ones are known.",
         {"cal.google.list", "crm.contacts.get"}, "composite"),
    Task("What is the total size in bytes of the Drive files named 'report'?",
         {"files.drive.search"}, "composite"),
    Task("Among my 20 latest emails, list the subjects of the ones from acme.com only.",
         {"mail.gmail.search"}, "composite"),

    # the integrator's exact shape: search returns ids, then the body of EACH must be
    # read. The right move is one run_code looping the ids; the tempting move is one
    # execute_tool per id, i.e. a round-trip per email.
    Task("Take my 4 most recent emails, open each one, and tell me which of them "
         "mention 'QUARTERLY'.",
         {"mail.gmail.search", "mail.gmail.read"}, "fanout"),
]


# --------------------------------------------------------------------- harness

@dataclass
class Run:
    success: bool = False
    llm_calls: int = 0
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    executed: set[str] = field(default_factory=set)
    tool_calls: int = 0
    max_fanout: int = 0      # most tool calls the model emitted in ONE turn (parallel)
    snippets: int = 0
    shim_saves: int = 0      # v0.7 would have returned a hollow {"stdout": ""} here
    import_tries: int = 0    # v0.7 error gave no guidance
    no_result: int = 0

    def effective(self) -> float:
        fresh = max(0, self.prompt_tokens - self.cached_tokens)
        return fresh + CACHE_READ_WEIGHT * self.cached_tokens + self.completion_tokens


def inspect_snippet(code: str, run: Run) -> None:
    """Count how often the model hits the exact failure modes 0.8.0 fixed."""
    run.snippets += 1
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return
    if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)):
        run.import_tries += 1
        return
    if _binds_output(tree):
        return
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        run.shim_saves += 1     # bare trailing expression: 0.8.0 promotes, 0.7.0 lost it
    else:
        run.no_result += 1


def make_call_model(api_key: str, model: str):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "X-Title": "SIFT-codemode-bench"}

    def call_model(messages: list[dict], tools: list[dict]) -> dict:
        body = {"model": model, "messages": messages, "tools": tools,
                "usage": {"include": True}, "reasoning": {"effort": "low"}}
        for attempt in range(4):
            resp = requests.post(ENDPOINT, headers=headers, json=body, timeout=180)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:200]}")
        raise RuntimeError("OpenRouter: retries exhausted")

    return call_model


def run_task(call_model, sift: Sift, task: Task, *, system: str, specs: list[dict],
             # generous: the N+1 task needs ~8 round-trips in the classic condition,
             # and truncating it there would hide code mode's real advantage instead
             # of measuring it
             max_steps: int = 14) -> Run:
    r = Run()

    # every execution, including the ones made by call() inside a snippet
    def observer(event: str, data: dict) -> None:
        if event == "execute" and data.get("ok") and data.get("path"):
            r.executed.add(data["path"])

    sift._observer = observer
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task.query}]
    try:
        for _ in range(max_steps):
            resp = call_model(messages, specs)
            r.llm_calls += 1
            u = resp.get("usage") or {}
            r.tokens += u.get("total_tokens") or 0
            r.prompt_tokens += u.get("prompt_tokens") or 0
            r.completion_tokens += u.get("completion_tokens") or 0
            r.cached_tokens += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0

            msg = resp["choices"][0]["message"]
            out = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                out["tool_calls"] = msg["tool_calls"]
            messages.append(out)

            tcs = msg.get("tool_calls") or []
            if not tcs:
                break
            r.tool_calls += len(tcs)
            r.max_fanout = max(r.max_fanout, len(tcs))   # >1 = parallel tool calling
            for tc in tcs:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "run_code":
                    inspect_snippet(args.get("code") or "", r)
                content = sift.dispatch(name, args)
                messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                                 "name": name, "content": content})
    finally:
        sift._observer = None
    r.success = task.gold <= r.executed
    return r


def legacy_code_specs() -> list[dict]:
    """Code mode exactly as 0.7.0 shipped it: no execute_tool, and a search_tools that
    only took ``q`` (so the active request — domain+action — was unreachable)."""
    return [
        {"type": "function", "function": {
            "name": "search_tools",
            "description": "Find tools by need. Returns matches with schema inline.",
            "parameters": {"type": "object",
                           "properties": {"q": {"type": "string", "description": "the need"}},
                           "required": ["q"]}}},
        next(t for t in code_tool_specs() if t["function"]["name"] == "run_code"),
    ]


CONDITIONS = {
    "classic (search+execute)":  lambda s: (SYSTEM_PROMPT, tool_specs()),
    "code 0.7 (as shipped)":     lambda s: (LEGACY_CODE_PROMPT, legacy_code_specs()),
    "code 0.8 (as shipped)":     lambda s: (CODE_SYSTEM_PROMPT, code_tool_specs()),
}


def load_env() -> tuple[str, str]:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key = os.environ.get("OPENROUTER_API", "")
    if not key:
        sys.exit("OPENROUTER_API not set in .env")
    return key, os.environ.get("SIFT_MODEL", "deepseek/deepseek-v4-flash")


def summarise(runs: list[Run]) -> dict:
    n = max(1, len(runs))
    return {
        "success": sum(r.success for r in runs) / n,
        "llm_calls": sum(r.llm_calls for r in runs) / n,
        "raw": sum(r.tokens for r in runs) / n,
        "eff": sum(r.effective() for r in runs) / n,
        "tool_calls": sum(r.tool_calls for r in runs) / n,
        "max_fanout": max((r.max_fanout for r in runs), default=0),
        "snippets": sum(r.snippets for r in runs),
        "shim_saves": sum(r.shim_saves for r in runs),
        "import_tries": sum(r.import_tries for r in runs),
        "no_result": sum(r.no_result for r in runs),
    }


def main() -> None:
    key, model = load_env()
    call_model = make_call_model(key, model)
    tasks = TASKS[:TASK_LIMIT] if TASK_LIMIT else TASKS
    print(f"model={model}  catalog={CATALOG_SIZE} tools  tasks={len(tasks)}\n", flush=True)

    sift = build_sift(CATALOG_SIZE)
    results: dict[str, list[Run]] = {}
    for cond, make in CONDITIONS.items():
        system, specs = make(sift)
        runs = []
        for i, t in enumerate(tasks, 1):
            try:
                r = run_task(call_model, sift, t, system=system, specs=specs)
            except Exception as exc:
                print(f"  ! {cond} task {i}: {str(exc)[:110]}", flush=True)
                r = Run()
            runs.append(r)
            print(f"  {cond:<26} {i:>2}/{len(tasks)} [{t.shape:<9}] "
                  f"{'OK' if r.success else '..'} {r.llm_calls} turns {r.tokens:>6}t "
                  f"{t.query[:38]}", flush=True)
        results[cond] = runs
        print(flush=True)

    for shape in ("single", "composite", "fanout", "all"):
        pick = [i for i, t in enumerate(tasks) if shape in (t.shape, "all")]
        if not pick:
            continue
        print(f"\n=== {shape.upper()} tasks ({len(pick)}) " + "=" * 46)
        print(f"{'condition':<26} | {'success':>7} | {'turns':>5} | {'calls':>5} | "
              f"{'par':>3} | {'raw tok':>8} | {'eff tok':>8}")
        print("-" * 88)
        for cond, runs in results.items():
            s = summarise([runs[i] for i in pick])
            print(f"{cond:<26} | {s['success']*100:6.0f}% | {s['llm_calls']:5.1f} | "
                  f"{s['tool_calls']:5.1f} | {s['max_fanout']:3.0f} | "
                  f"{s['raw']:8.0f} | {s['eff']:8.0f}")

    print("\n=== snippets the model wrote (the 0.8.0 failure modes) " + "=" * 19)
    print(f"{'condition':<26} | {'snippets':>8} | {'bare expr':>9} | {'imports':>7} | {'no result':>9}")
    print("-" * 74)
    for cond, runs in results.items():
        s = summarise(runs)
        if not s["snippets"]:
            continue
        print(f"{cond:<26} | {s['snippets']:8.0f} | {s['shim_saves']:9.0f} | "
              f"{s['import_tries']:7.0f} | {s['no_result']:9.0f}")
    print("\n'bare expr' + 'imports' + 'no result' = turns 0.7.0 would have WASTED:")
    print("each one returned a hollow success or an error with no guidance, and every")
    print("wasted turn re-sends the whole conversation.")

    out = ROOT / "benchmarks" / "codemode_results.json"
    out.write_text(json.dumps({c: summarise(r) for c, r in results.items()}, indent=2),
                   encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
