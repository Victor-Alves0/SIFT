"""BFCL-style function-call accuracy eval.

Methodology of the Berkeley Function Calling Leaderboard: given a request and a
catalogue of functions, did the model pick the RIGHT function with the RIGHT
arguments — in a single shot? We check the catalogue exposed flat (the standard
function-calling setup) and AST-compare the emitted call to the gold answer.

This is the BFCL *methodology* on your own catalogue, not the official leaderboard
run. (tau-bench is intentionally out of scope: it needs a stateful, multi-turn
customer-service environment with a backing DB — an external harness, not a
metric you can compute from a tool schema.)

LLM-agnostic: pass ``call_model(messages, tools) -> dict`` (OpenAI-style JSON).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .agentbench import _flat_specs
from .registry import Registry

_SYSTEM = ("You are a function-calling engine. Call exactly ONE function that "
           "satisfies the user's request, with correct arguments. Tool names use "
           "'__' where a dotted path would be.")


@dataclass
class Case:
    query: str
    gold_path: str
    gold_args: dict = field(default_factory=dict)  # args that must match (subset)


@dataclass
class EvalReport:
    cases: int = 0
    function_acc: float = 0.0  # right function chosen
    arg_acc: float = 0.0       # right function AND required args correct
    no_call_rate: float = 0.0  # model didn't call any tool
    misses: list[str] = field(default_factory=list)

    def format(self) -> str:
        return (
            f"BFCL-STYLE FUNCTION-CALL ACCURACY  (cases={self.cases})\n"
            f"  Function accuracy : {self.function_acc*100:5.1f}%   (right tool)\n"
            f"  Arg accuracy      : {self.arg_acc*100:5.1f}%   (right tool + args)\n"
            f"  No-call rate      : {self.no_call_rate*100:5.1f}%\n"
        )


def bfcl_style(call_model, registry: Registry, cases: list[Case]) -> EvalReport:
    specs = _flat_specs(registry)
    rep = EvalReport(cases=len(cases))
    fn_ok = arg_ok = no_call = 0

    for c in cases:
        messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": c.query}]
        resp = call_model(messages, specs)
        tcs = resp["choices"][0]["message"].get("tool_calls") or []
        if not tcs:
            no_call += 1
            rep.misses.append(f"no call: {c.query!r}")
            continue
        called = tcs[0]["function"]["name"].replace("__", ".")
        try:
            args = json.loads(tcs[0]["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}

        if called == c.gold_path:
            fn_ok += 1
            if all(str(args.get(k)) == str(v) for k, v in c.gold_args.items()):
                arg_ok += 1
            else:
                rep.misses.append(f"bad args: {c.query!r} -> {args}")
        else:
            rep.misses.append(f"wrong tool: {c.query!r} -> {called}")

    n = max(1, len(cases))
    rep.function_acc = fn_ok / n
    rep.arg_acc = arg_ok / n
    rep.no_call_rate = no_call / n
    return rep
