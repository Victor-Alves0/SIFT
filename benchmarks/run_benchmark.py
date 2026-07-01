"""Run the SIFT-vs-flat benchmark against a live model via OpenRouter.

Usage (from repo root, with the package venv):
    python benchmarks/run_benchmark.py

Reads OPENROUTER_API and SIFT_MODEL from .env. Compares two conditions across
several catalogue sizes and prints a market-style comparison table.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

from sift import Sift
from sift.agentbench import RunResult, aggregate, build_catalog, run_flat, run_sift
from sift.bench import load_tasks

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "core" / "data" / "registry.json"
TASKS = ROOT / "core" / "data" / "bench_tasks.json"
SIZES = [int(x) for x in os.environ.get("SIFT_BENCH_SIZES", "25,100,250").split(",")]
TASK_LIMIT = int(os.environ.get("SIFT_BENCH_TASKS", "0") or 0)
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def load_env() -> tuple[str, str]:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key = os.environ.get("OPENROUTER_API", "")
    model = os.environ.get("SIFT_MODEL", "deepseek/deepseek-v4-flash")
    if not key:
        sys.exit("OPENROUTER_API not set in .env")
    return key, model


def make_call_model(api_key: str, model: str):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "X-Title": "SIFT-benchmark"}

    def call_model(messages: list[dict], tools: list[dict]) -> dict:
        body = {
            "model": model,
            "messages": messages,
            "tools": tools,
            # ask OpenRouter to report cached-token accounting; keep routing cheap
            "usage": {"include": True},
            "reasoning": {"effort": "low"},
        }
        for attempt in range(4):
            resp = requests.post(ENDPOINT, headers=headers, json=body, timeout=120)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503):
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError("OpenRouter: retries exhausted")

    return call_model


def main() -> None:
    key, model = load_env()
    call_model = make_call_model(key, model)
    tasks = load_tasks(TASKS)
    if TASK_LIMIT:
        tasks = tasks[:TASK_LIMIT]
    print(f"model={model}  tasks={len(tasks)}  sizes={SIZES}\n", flush=True)

    all_stats = []
    for size in SIZES:
        reg = build_catalog(str(REGISTRY), size)
        sift = Sift(registry=reg)
        print(f"[size={size}] building semantic index ({sum(1 for _ in reg.tools())} tools)...", flush=True)
        sift.build_index()

        flat_runs, sift_runs = [], []
        flat_errors = 0
        for i, t in enumerate(tasks, 1):
            try:
                fr = run_flat(call_model, sift.gateway, t.query, t.gold)
            except Exception as exc:  # e.g. provider rejects 250 tools in one call
                fr = RunResult()
                flat_errors += 1
                print(f"  [{size}] flat error on task {i}: {str(exc)[:120]}", flush=True)
            try:
                sr = run_sift(call_model, sift, t.query, t.gold)
            except Exception as exc:
                sr = RunResult()
                print(f"  [{size}] sift error on task {i}: {str(exc)[:120]}", flush=True)
            flat_runs.append(fr)
            sift_runs.append(sr)
            print(f"  [{size}] {i}/{len(tasks)} {t.query[:34]:<34} "
                  f"flat={'OK' if fr.success else '..'}({fr.tokens:>6}t)  "
                  f"sift={'OK' if sr.success else '..'}({sr.tokens:>5}t)", flush=True)
        if flat_errors:
            print(f"  [{size}] flat condition had {flat_errors} API error(s) "
                  f"(likely too many tools in one request)", flush=True)

        all_stats.append(aggregate("flat (market baseline)", size, flat_runs))
        all_stats.append(aggregate("SIFT (2 meta-tools)", size, sift_runs))

    print("\n" + "=" * 104)
    print(f"{'catalog':>7} | {'condition':<24} | {'success':>7} | {'raw tok':>8} | "
          f"{'eff tok':>8} | {'eff x':>6} | {'tools':>6} | {'wrong':>6}")
    print("-" * 104)
    by_size: dict[int, dict] = {}
    for s in all_stats:
        by_size.setdefault(s.catalog_size, {})[s.condition.split()[0]] = s
    for size in SIZES:
        flat = by_size[size]["flat"]
        sift_s = by_size[size]["SIFT"]
        ratio = flat.avg_effective_tokens / sift_s.avg_effective_tokens if sift_s.avg_effective_tokens else 0
        for s in (flat, sift_s):
            x = f"{flat.avg_effective_tokens / s.avg_effective_tokens:4.1f}x" if s.avg_effective_tokens else "  -"
            print(f"{size:>7} | {s.condition:<24} | {s.success_rate*100:6.1f}% | "
                  f"{s.avg_tokens:8.0f} | {s.avg_effective_tokens:8.0f} | {x:>6} | "
                  f"{s.avg_tool_calls:6.2f} | {s.avg_wrong_calls:6.2f}")
        print(f"{'':>7} | -> SIFT {ratio:.1f}x cheaper (effective tokens)")
    print("=" * 104)

    out = ROOT / "benchmarks" / "results.json"
    out.write_text(json.dumps([s.__dict__ for s in all_stats], indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
