"""Needle-in-a-haystack on the MCP-Zero dataset (308 servers / ~2.8k tools).

Runs SIFT's discovery over the PUBLIC MCP-tools catalogue released with the
MCP-Zero paper (arXiv 2506.01056) — the first evaluation of SIFT on a dataset we
didn't construct. For every tool in the catalogue, discovery must surface that
tool from the full index ("needle"), under two conditions:

  - query-only:      search_tools(tool_description)
  - active request:  search_request(domain=server name+description,
                                    action=tool_description)

Notes on comparability (honesty first):
  - MCP-Zero routes model-authored requests with OpenAI text-embedding-3-large
    (a paid API, 3072-dim). SIFT here uses its default LOCAL bge-small (384-dim)
    + hybrid BM25 — the point is accuracy WITHOUT an embedding API.
  - Using the tool's own description as the request mirrors the paper's premise
    (model requests align with tool docs); it is a self-retrieval needle test,
    not a replication of their LLM-in-the-loop runs.

Dataset (333 MB, includes their precomputed embeddings, which we ignore):
    python -m gdown 1RjBGU-AGdHdhUABoeYSztbfQlD0hjUBn -O mcp_tools_with_embedding.json
    python benchmarks/mcpzero_needle.py path/to/mcp_tools_with_embedding.json
"""
from __future__ import annotations

import json
import re
import sys
import time

from sift import Sift
from sift.importers._common import compress_params, sanitize_text
from sift.registry import ToolDef
from sift.toon import estimate_tokens


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", str(text).lower()).strip("_") or "x"


def load_catalog(path: str) -> tuple[Sift, list[tuple[str, str, str]]]:
    """Register the whole dataset; returns (sift, cases) where each case is
    (tool_path, server_text, tool_description)."""
    data = json.load(open(path, encoding="utf-8"))
    sift = Sift(index_cache=path + ".sift-index.npz")
    cases: list[tuple[str, str, str]] = []
    skipped = 0
    seen_svc: set[str] = set()

    for server in data:
        svc = _slug(server.get("name", ""))
        while svc in seen_svc:   # the dataset has same-named servers
            svc += "_"
        seen_svc.add(svc)
        server_desc = sanitize_text(server.get("description", ""), max_len=250)
        sift.describe(f"mcp.{svc}", server_desc)
        seen: set[str] = set()
        for tool in server.get("tools", []) or []:
            desc = sanitize_text(tool.get("description", ""), max_len=250)
            if not desc:
                skipped += 1
                continue
            fn = _slug(tool.get("name", ""))
            while fn in seen:            # rare same-name collisions after slugging
                fn += "_"
            seen.add(fn)
            path_ = f"mcp.{svc}.{fn}"
            sift.registry.add(ToolDef(path_, desc,
                                      compress_params(tool.get("parameter") or {})))
            cases.append((path_, f"{server.get('name', '')} {server_desc}", desc))

    print(f"catalogue: {len(cases)} tools across {len(data)} servers "
          f"({skipped} skipped: empty description)")
    return sift, cases


def flat_payload_tokens(sift: Sift) -> int:
    """Approximate tokens a FLAT setup would inject: every tool as a JSON spec."""
    from sift.registry import input_schema_for
    specs = [{"name": t.path, "description": t.description,
              "parameters": input_schema_for(t)} for t in sift.registry.tools()]
    return estimate_tokens(json.dumps(specs))


def main(path: str) -> None:
    sift, cases = load_catalog(path)

    t0 = time.perf_counter()
    sift.build_index()
    print(f"index build: {time.perf_counter() - t0:.1f}s "
          f"(warm rebuilds load the cache in ~ms)")

    flat = flat_payload_tokens(sift)
    surface = estimate_tokens(sift.system_prompt) + estimate_tokens(
        json.dumps(sift.openai_tools()))
    print(f"flat schema payload: ~{flat:,} tokens | SIFT fixed surface: ~{surface} tokens "
          f"({flat // max(surface, 1)}x smaller)\n")

    q1 = q5 = a1 = a5 = 0
    t0 = time.perf_counter()
    for i, (gold, server_text, desc) in enumerate(cases, 1):
        hits = [r.path for r in sift.search_tools(desc, top_k=15)
                if r.kind == "function"][:5]
        q1 += hits[:1] == [gold]
        q5 += gold in hits

        hits = [r.path for r in sift.search_request(server_text, desc, top_k=5)
                if r.kind == "function"][:5]
        a1 += hits[:1] == [gold]
        a5 += gold in hits

        if i % 250 == 0:
            print(f"  {i}/{len(cases)}  q@1={q1 / i:.1%} a@1={a1 / i:.1%}", flush=True)

    n = len(cases)
    dt = time.perf_counter() - t0
    print(f"\nMCP-tools needle ({n} tools, full catalogue in the index):")
    print(f"{'condition':<38}{'top-1':>8}{'top-5':>8}")
    print(f"{'query-only (tool description)':<38}{q1 / n:>8.1%}{q5 / n:>8.1%}")
    print(f"{'active request (domain + action)':<38}{a1 / n:>8.1%}{a5 / n:>8.1%}")
    print(f"\n{n * 2} searches in {dt:.0f}s ({dt / (n * 2) * 1000:.0f} ms/search, "
          f"local embeddings — no API)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mcp_tools_with_embedding.json")
