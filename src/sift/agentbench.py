"""Agent-level (downstream) benchmark — SIFT vs the flat-catalogue baseline.

The flat condition is what most tool/MCP setups do today: every tool is dumped
into the model's context as a function-calling spec. The SIFT condition exposes
only the 3 meta-tools. Same model, same tasks, same catalogue — we vary the
catalogue size with distractors (ToolMenuBench methodology) and measure success,
token cost and tool-call accuracy.

LLM-agnostic: pass a ``call_model(messages, tools) -> dict`` callable returning
OpenRouter/OpenAI-style JSON (``{"choices":[{"message":...}], "usage":{...}}``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .gateway import Gateway
from .metatools import SYSTEM_PROMPT, tool_specs
from .registry import Registry, ToolDef

# --------------------------------------------------------------------- catalog

# A broad domain × service matrix. Combined with a fixed action set this yields
# 400+ distinct, plausible distractor tools — enough for 250-tool catalogues with
# realistic near-duplicate and cross-domain distractors (ToolMenuBench style).
_DOMAINS: dict[str, list[str]] = {
    "crm": ["contacts", "leads", "accounts", "deals", "companies"],
    "payments": ["invoices", "charges", "payouts", "refunds"],
    "calendar": ["events", "reminders", "availability"],
    "weather": ["forecast", "alerts", "history"],
    "maps": ["places", "routes", "geocoding"],
    "music": ["tracks", "playlists", "albums"],
    "notes": ["notes", "notebooks"],
    "tasks": ["items", "projects", "labels"],
    "hr": ["employees", "payroll", "leave"],
    "inventory": ["products", "warehouses", "orders"],
    "social": ["posts", "comments", "followers"],
    "sms": ["messages", "numbers"],
    "translate": ["text", "documents"],
    "images": ["generation", "edits"],
    "datastore": ["records", "tables", "collections"],
    "billing": ["subscriptions", "plans", "usage"],
    "support": ["tickets", "articles", "agents"],
    "analytics": ["events", "reports", "funnels"],
    "storage": ["objects", "buckets", "shares"],
    "video": ["clips", "streams", "captions"],
    "ecommerce": ["carts", "checkout", "coupons"],
    "iot": ["devices", "sensors", "scenes"],
    "travel": ["flights", "hotels", "cars"],
    "finance": ["stocks", "portfolios", "watchlists"],
    "fitness": ["activities", "sleep", "nutrition"],
}

_ACTIONS: list[tuple[str, str]] = [
    ("list", "List"),
    ("get", "Get"),
    ("search", "Search"),
    ("create", "Create"),
    ("update", "Update"),
    ("delete", "Delete"),
]
_RISKY_ACTIONS = {"delete"}


def synth_distractors(limit: int) -> list[ToolDef]:
    """Generate up to ``limit`` plausible-but-irrelevant tools (distractors).

    Deterministic order, so a given size is reproducible across runs.
    """
    out: list[ToolDef] = []
    for domain in sorted(_DOMAINS):
        for service in _DOMAINS[domain]:
            for action, verb in _ACTIONS:
                out.append(ToolDef(
                    path=f"{domain}.{service}.{action}",
                    description=f"{verb} {service} in the {domain} system",
                    params={"id": "string:o::resource id", "q": "string:o::query filter"},
                    returns=["ok"],
                    risk=action in _RISKY_ACTIONS,
                    fn=lambda **kw: {"ok": True},
                ))
                if len(out) >= limit:
                    return out
    return out


def build_catalog(gold_registry_json: str, size: int) -> Registry:
    """Gold tools (from a JSON registry) + distractors, padded to ``size`` tools.

    Every tool gets a mock executor so trajectories can actually run.
    """
    reg = Registry.from_json(gold_registry_json)
    for t in list(reg.tools()):
        reg.bind(t.path, _mock_executor(t))

    gold_n = sum(1 for _ in reg.tools())
    for d in synth_distractors(max(0, size - gold_n)):
        reg.add(d)
    return reg


def _mock_executor(tool: ToolDef):
    fields = list(tool.returns) or ["ok"]
    def _run(**kwargs):
        return {f: (kwargs.get(f) if f in kwargs else f"<{f}>") for f in fields}
    return _run


# --------------------------------------------------------------------- results

@dataclass
class RunResult:
    success: bool = False
    llm_calls: int = 0
    tool_calls: int = 0
    wrong_calls: int = 0
    tokens: int = 0  # total tokens (prompt + completion), raw
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens served from cache (cheaper)
    truncated: bool = False
    executed: list[str] = field(default_factory=list)


# cache reads are billed at roughly a tenth of fresh input — used to turn raw
# token counts into a cost-weighted "effective tokens" figure.
CACHE_READ_WEIGHT = 0.1


@dataclass
class ConditionStats:
    condition: str
    catalog_size: int
    tasks: int = 0
    success_rate: float = 0.0
    avg_tokens: float = 0.0
    avg_effective_tokens: float = 0.0  # cost-weighted (cached input discounted)
    avg_cached_tokens: float = 0.0
    avg_tool_calls: float = 0.0
    avg_llm_calls: float = 0.0
    avg_wrong_calls: float = 0.0


def _assistant(msg: dict) -> dict:
    out = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def _loop(call_model, system: str, user: str, tools: list[dict], handle, gold: str,
          max_steps: int) -> RunResult:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    r = RunResult()
    for _ in range(max_steps):
        resp = call_model(messages, tools)
        r.llm_calls += 1
        usage = resp.get("usage") or {}
        r.tokens += usage.get("total_tokens", 0) or 0
        r.prompt_tokens += usage.get("prompt_tokens", 0) or 0
        r.completion_tokens += usage.get("completion_tokens", 0) or 0
        r.cached_tokens += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        msg = resp["choices"][0]["message"]
        messages.append(_assistant(msg))

        tcs = msg.get("tool_calls") or []
        if not tcs:
            break
        for tc in tcs:
            r.tool_calls += 1
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            content, executed = handle(name, args)
            if executed:
                r.executed.append(executed)
                if executed != gold:
                    r.wrong_calls += 1
            messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                             "name": name, "content": content})
    else:
        r.truncated = True
    r.success = gold in r.executed
    return r


# --------------------------------------------------------------------- conditions

_FLAT_SYSTEM = ("You are an assistant with the following tools. Pick the single right tool, "
                "call it with valid arguments, then answer the user. Tool names use '__' where a "
                "dotted path would be.")


def _flat_specs(reg: Registry) -> list[dict]:
    specs = []
    for t in reg.tools():
        props, required = {}, []
        for name, p in t.params.items():
            props[name] = {"type": "number" if p.type == "number" else "string", "description": p.desc}
            if p.required:
                required.append(name)
        specs.append({"type": "function", "function": {
            "name": t.path.replace(".", "__"),
            "description": t.description,
            "parameters": {"type": "object", "properties": props, "required": required},
        }})
    return specs


def run_flat(call_model, gateway: Gateway, query: str, gold: str, *, max_steps: int = 8) -> RunResult:
    specs = _flat_specs(gateway.reg)

    def handle(name: str, args: dict):
        path = name.replace("__", ".")
        try:
            res = gateway.execute_tool(path, args)
            return json.dumps(res, default=str, ensure_ascii=False), path
        except Exception as exc:  # tool not found / bad args
            return json.dumps({"error": str(exc)}), None

    return _loop(call_model, _FLAT_SYSTEM, query, specs, handle, gold, max_steps)


def run_sift(call_model, sift, query: str, gold: str, *, max_steps: int = 10) -> RunResult:
    specs = tool_specs()

    def handle(name: str, args: dict):
        out = sift.dispatch(name, args)
        executed = None
        if name == "execute_tool" and '"error"' not in out:
            executed = args.get("path")
        return out, executed

    return _loop(call_model, SYSTEM_PROMPT, query, specs, handle, gold, max_steps)


def _effective(r: RunResult) -> float:
    """Cost-weighted tokens: cached input billed at CACHE_READ_WEIGHT of fresh."""
    fresh_prompt = max(0, r.prompt_tokens - r.cached_tokens)
    return fresh_prompt + CACHE_READ_WEIGHT * r.cached_tokens + r.completion_tokens


def aggregate(condition: str, size: int, runs: list[RunResult]) -> ConditionStats:
    n = max(1, len(runs))
    return ConditionStats(
        condition=condition,
        catalog_size=size,
        tasks=len(runs),
        success_rate=sum(r.success for r in runs) / n,
        avg_tokens=sum(r.tokens for r in runs) / n,
        avg_effective_tokens=sum(_effective(r) for r in runs) / n,
        avg_cached_tokens=sum(r.cached_tokens for r in runs) / n,
        avg_tool_calls=sum(r.tool_calls for r in runs) / n,
        avg_llm_calls=sum(r.llm_calls for r in runs) / n,
        avg_wrong_calls=sum(r.wrong_calls for r in runs) / n,
    )
