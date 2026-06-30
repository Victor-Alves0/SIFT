"""Evaluation harness (the Benchmarks branch of the design).

Focuses on FILTER-LEVEL METRICS — deterministic, no LLM cost: do we expose the
right tool before the agent acts? Plus a TOON-vs-JSON token comparison.

Downstream agent metrics (which require running an LLM) are intentionally left
out of the default harness so this stays cheap and CI-friendly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import toon
from .registry import Registry, ToolDef, param_dict


@dataclass
class Task:
    query: str
    gold: str
    needs_risky: bool = False


def load_tasks(path: str | Path) -> list[Task]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Task(t["query"], t["gold"], bool(t.get("needs_risky", False))) for t in data]


@dataclass
class FilterMetrics:
    tasks: int = 0
    top_k: int = 0
    gold_exposure: float = 0.0          # fraction with gold visible in top-k
    no_visible_tool_rate: float = 0.0   # fraction with gold absent
    avg_visible_tools: float = 0.0
    avg_extra_tools: float = 0.0        # noise per task
    mrr: float = 0.0                    # mean reciprocal rank of gold
    risky_exposure_rate: float = 0.0
    unauthorized_risky_rate: float = 0.0
    misses: list[str] = field(default_factory=list)

    def format(self) -> str:
        return (
            f"FILTER-LEVEL METRICS  (tasks={self.tasks}, top_k={self.top_k})\n"
            f"  Gold next-tool exposure : {self.gold_exposure*100:5.1f}%   (target: high)\n"
            f"  No-visible-tool rate    : {self.no_visible_tool_rate*100:5.1f}%   (target: 0%)\n"
            f"  Average visible tools   : {self.avg_visible_tools:5.2f}    (target: low)\n"
            f"  Extra tools exposed     : {self.avg_extra_tools:5.2f}    (noise/task)\n"
            f"  MRR (gold rank)         : {self.mrr:5.3f}    (target: ->1.0)\n"
            f"  Risky-tool exposure     : {self.risky_exposure_rate*100:5.1f}%\n"
            f"  Unauthorized risky      : {self.unauthorized_risky_rate*100:5.1f}%   (target: 0%)\n"
        )


def run_filter(sift, tasks: list[Task], top_k: int = 5) -> FilterMetrics:
    """Run discovery for each task and compute the filter-level metrics."""
    risky = sift.registry.risky_paths()
    m = FilterMetrics(tasks=len(tasks), top_k=top_k)

    gold_hits = sum_visible = sum_extra = risky_tasks = unauth = 0
    sum_rr = 0.0

    for t in tasks:
        results = sift.search_tools(t.query, top_k)
        visible = len(results)
        sum_visible += visible

        rank = 0
        risky_shown = False
        for i, r in enumerate(results):
            if r.path == t.gold:
                rank = i + 1
            if r.path in risky:
                risky_shown = True

        if rank:
            gold_hits += 1
            sum_rr += 1.0 / rank
            sum_extra += visible - 1
        else:
            sum_extra += visible
            m.misses.append(f"{t.query!r} expected {t.gold}")
        if risky_shown:
            risky_tasks += 1
            if not t.needs_risky:
                unauth += 1

    n = max(1, len(tasks))
    m.gold_exposure = gold_hits / n
    m.no_visible_tool_rate = (len(tasks) - gold_hits) / n
    m.avg_visible_tools = sum_visible / n
    m.avg_extra_tools = sum_extra / n
    m.mrr = sum_rr / n
    m.risky_exposure_rate = risky_tasks / n
    m.unauthorized_risky_rate = unauth / n
    return m


@dataclass
class TokenReport:
    functions: int = 0
    toon_tokens: int = 0
    compact_json_tokens: int = 0
    verbose_json_tokens: int = 0
    reduction_vs_compact: float = 0.0
    reduction_vs_verbose: float = 0.0

    def format(self) -> str:
        avg_toon = self.toon_tokens // self.functions if self.functions else 0
        avg_verbose = self.verbose_json_tokens // self.functions if self.functions else 0
        return (
            f"TOKEN COST  ({self.functions} functions, ~4 chars/token estimate)\n"
            f"  TOON total           : {self.toon_tokens:5d} tokens  (~{avg_toon}/tool)\n"
            f"  Compact JSON total   : {self.compact_json_tokens:5d} tokens\n"
            f"  Verbose JSON Schema  : {self.verbose_json_tokens:5d} tokens  (~{avg_verbose}/tool, OpenAPI baseline)\n"
            f"  Reduction TOON vs compact : {self.reduction_vs_compact*100:5.1f}%\n"
            f"  Reduction TOON vs verbose : {self.reduction_vs_verbose*100:5.1f}%\n"
        )


def _verbose_json_schema(tool: ToolDef) -> str:
    props, required = {}, []
    for name, p in tool.params.items():
        jt = "number" if p.type == "number" else "string"
        prop = {"type": jt, "description": p.desc}
        if p.default:
            prop["default"] = p.default
        props[name] = prop
        if p.required:
            required.append(name)
    schema = {
        "name": tool.path,
        "description": tool.description,
        "parameters": {"type": "object", "properties": props, "required": required},
    }
    return json.dumps(schema, indent=2, ensure_ascii=False)


def token_report(registry: Registry) -> TokenReport:
    rep = TokenReport()
    for tool in registry.tools():
        rep.functions += 1
        rep.toon_tokens += toon.estimate_tokens(toon.encode_function(tool))
        p_serialisable = {name: param_dict(p) for name, p in tool.params.items()}
        compact = json.dumps({"d": tool.description, "p": p_serialisable, "r": tool.returns},
                             ensure_ascii=False)
        rep.compact_json_tokens += toon.estimate_tokens(compact)
        rep.verbose_json_tokens += toon.estimate_tokens(_verbose_json_schema(tool))
    if rep.compact_json_tokens:
        rep.reduction_vs_compact = 1 - rep.toon_tokens / rep.compact_json_tokens
    if rep.verbose_json_tokens:
        rep.reduction_vs_verbose = 1 - rep.toon_tokens / rep.verbose_json_tokens
    return rep
