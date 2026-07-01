"""The gateway: the meta-tools the LLM actually sees.

    search_tools(query | domain+action | path)  -> discovery + inspect  [Search·Inspect]
    execute_tool(path, params)                   -> run + filter         [Trigger·Filter]

``search_tools`` merges discovery and inspection: it returns matches with their
TOON schema inline (semantic query, structured active request, or hierarchy
browse via ``path``), so the model calls ``execute_tool`` directly. The internal
``get_tool_schema`` powers the browse path and stays as a back-compat alias.
The model never sees the full catalogue — it discovers tools by navigating.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import toon
from .embeddings import Embedder, cosine
from .registry import Param, Registry
from .retrieval import BM25, rank_order, rrf


@dataclass
class SearchResult:
    path: str
    kind: str
    description: str
    score: float

    def as_dict(self) -> dict:
        return {"path": self.path, "kind": self.kind, "d": self.description,
                "score": round(self.score, 4)}


class Gateway:
    def __init__(self, registry: Registry, embedder: Embedder | None = None, *,
                 retrieval: str = "hybrid", reranker=None, min_score: float = 0.0) -> None:
        if retrieval not in ("hybrid", "embedding", "bm25"):
            raise ValueError("retrieval must be 'hybrid', 'embedding' or 'bm25'")
        if retrieval in ("hybrid", "embedding") and embedder is None:
            raise ValueError(f"retrieval={retrieval!r} needs an embedder")
        self.reg = registry
        self.embedder = embedder
        self.retrieval = retrieval
        self.reranker = reranker  # object with .rerank(query, docs) -> list[float]
        self.min_score = min_score  # relevance floor; below it, search returns nothing
        self._entries: list = []
        self._vectors: list = []
        self._bm25: BM25 | None = None
        self._toon_cache: dict[str, str] = {}

    # ------------------------------------------------------------- index
    def build_index(self) -> "Gateway":
        entries = self.reg.search_entries()
        if not entries:
            raise ValueError("registry is empty — register tools before build_index()")
        self._entries = entries
        texts = [e.text for e in entries]
        if self.retrieval in ("hybrid", "embedding"):
            self._vectors = self.embedder.embed(texts)
        if self.retrieval in ("hybrid", "bm25"):
            self._bm25 = BM25(texts)
        return self

    # --------------------------------------------------- meta-tool: search
    def search_tools(self, query: str, top_k: int = 5, *, predicate=None) -> list[SearchResult]:
        if not self._entries:
            raise RuntimeError("index not built — call build_index() first")
        top_k = max(1, top_k)

        emb_scores = bm_scores = None
        if self.retrieval in ("hybrid", "embedding"):
            qv = self.embedder.embed([query])[0]
            emb_scores = [cosine(qv, v) for v in self._vectors]
        if self.retrieval in ("hybrid", "bm25"):
            bm_scores = self._bm25.scores(query)

        # relevance floor: if nothing is actually close, return no matches
        if self.min_score > 0:
            if emb_scores:                      # calibrated cosine in [0, 1]
                relevant = max(emb_scores) >= self.min_score
            elif bm_scores:                     # bm25-only: at least one term must match
                relevant = max(bm_scores) > 0
            else:
                relevant = True
            if not relevant:
                return []

        if self.retrieval == "embedding":
            fused = {i: s for i, s in enumerate(emb_scores)}
        elif self.retrieval == "bm25":
            fused = {i: s for i, s in enumerate(bm_scores)}
        else:  # hybrid via Reciprocal Rank Fusion
            fused = rrf([rank_order(emb_scores), rank_order(bm_scores)])

        order = sorted(fused, key=lambda i: fused[i], reverse=True)

        # scope filter (allow/deny): keep only permitted paths before truncation
        if predicate is not None:
            order = [i for i in order if predicate(self._entries[i].path)]

        # optional cross-encoder rerank over the fused shortlist
        if self.reranker is not None and order:
            shortlist = order[: max(top_k * 4, 10)]
            rr = self.reranker.rerank(query, [self._entries[i].text for i in shortlist])
            order = [i for _, i in sorted(zip(rr, shortlist), key=lambda p: p[0], reverse=True)] \
                + order[len(shortlist):]

        results = []
        for i in order[:top_k]:
            e = self._entries[i]
            results.append(SearchResult(e.path, e.kind, e.description, round(fused[i], 4)))
        return results

    # ---------------------------- meta-tool: search (structured / active request)
    def search_request(self, domain: str, action: str, top_k: int = 3, *,
                       predicate=None) -> list[SearchResult]:
        """Two-stage 'active tool request' routing (the MCP-Zero idea, on our
        hybrid signals).

        The model states its intent as two fields instead of one raw query —
        ``domain`` (platform / permission area, e.g. "email", "google workspace")
        and ``action`` (operation + target, e.g. "read the latest message"). A
        model-authored request aligns better with tool docs than a user's raw
        query, which is what lifts accuracy over query-only retrieval.

        Stage 1 scores ``domain`` against services; stage 2 scores ``action``
        against functions; the two are fused per MCP-Zero's rule
        ``(s_server·s_tool)·max(s_server, s_tool)`` — but each ``s_*`` here is our
        hybrid (embedding blended with normalised BM25), not dense-only.
        """
        if not self._entries:
            raise RuntimeError("index not built — call build_index() first")
        top_k = max(1, top_k)
        domain = (domain or "").strip()
        action = (action or "").strip()
        if not action:  # nothing to route the operation on — fall back
            return self.search_tools(domain, top_k, predicate=predicate) if domain else []
        if not domain:  # no domain hint — plain semantic search on the action
            return self.search_tools(action, top_k, predicate=predicate)

        a_rel = self._relevance(action)
        if self.min_score > 0 and (not a_rel or max(a_rel) < self.min_score):
            return []
        d_rel = self._relevance(domain)
        svc_score = {e.path: d_rel[i] for i, e in enumerate(self._entries)
                     if e.kind == "service"}

        # If the domain hint resonates with no service, every function would tie
        # at 0 and ordering would be arbitrary — a wrong (possibly risky) tool
        # could surface on top. Degrade gracefully to ranking on the action alone
        # (functions only, to match the normal request output).
        if not svc_score or max(svc_score.values()) <= 0.0:
            res = self.search_tools(action, top_k=max(top_k * 4, top_k), predicate=predicate)
            fns = [r for r in res if r.kind == "function"]
            return (fns or res)[:top_k]

        scored: list[tuple[float, int]] = []
        for i, e in enumerate(self._entries):
            if e.kind != "function":
                continue
            if predicate is not None and not predicate(e.path):
                continue
            s_server = svc_score.get(e.path.rsplit(".", 1)[0], 0.0)
            s_tool = a_rel[i]
            scored.append(((s_server * s_tool) * max(s_server, s_tool), i))
        scored.sort(key=lambda p: p[0], reverse=True)

        # optional cross-encoder rerank over the fused shortlist (query = the action)
        order = [i for _, i in scored]
        if self.reranker is not None and order:
            shortlist = order[: max(top_k * 4, 10)]
            rr = self.reranker.rerank(action, [self._entries[i].text for i in shortlist])
            order = [i for _, i in sorted(zip(rr, shortlist), key=lambda p: p[0], reverse=True)] \
                + order[len(shortlist):]
            fused_by_i = {i: s for s, i in scored}
            scored = [(fused_by_i[i], i) for i in order]

        results = []
        for s, i in scored[:top_k]:
            e = self._entries[i]
            results.append(SearchResult(e.path, e.kind, e.description, round(s, 4)))
        return results

    def search_request_compact(self, domain: str, action: str, top_k: int = 3, *,
                               predicate=None) -> str:
        """TOON-rendered structured-request results (schema inline), like
        ``search_compact`` but for the two-field active request."""
        results = self.search_request(domain, action, top_k=top_k, predicate=predicate)
        return self._render_compact(results, top_k)

    def _relevance(self, query: str) -> list[float]:
        """Per-entry relevance in [0, 1] for ``query`` — the hybrid signal used by
        the multiplicative request-routing score (embedding cosine blended with
        max-normalised BM25). Magnitudes matter here, so this does NOT use RRF."""
        emb = bm = None
        if self.retrieval in ("hybrid", "embedding"):
            qv = self.embedder.embed([query])[0]
            emb = [cosine(qv, v) for v in self._vectors]
        if self.retrieval in ("hybrid", "bm25"):
            raw = self._bm25.scores(query)
            top = max(raw) if raw else 0.0
            bm = [x / top for x in raw] if top > 0 else raw
        if emb is not None and bm is not None:
            return [(e + b) / 2 for e, b in zip(emb, bm)]
        return emb if emb is not None else (bm or [])

    def search_compact(self, query: str, top_k: int = 3, *, predicate=None) -> str:
        """Agent-facing discovery: top function matches rendered as TOON, schema
        inline — so the model can call execute_tool directly (merges Search +
        Inspect). No scores/kind noise; capped to ``top_k`` lines.
        """
        results = self.search_tools(query, top_k=max(top_k * 4, top_k), predicate=predicate)
        return self._render_compact(results, top_k)

    def _render_compact(self, results: list[SearchResult], top_k: int) -> str:
        if not results:
            return "# no matching tools — none of the available tools fit this request."
        seen: set[str] = set()
        lines: list[str] = []
        for r in results:
            if r.kind != "function" or r.path in seen:
                continue
            seen.add(r.path)
            lines.append(toon.encode_function(self.reg.tool(r.path)))
            if len(lines) >= top_k:
                break
        if not lines:  # defensive: no function entries indexed
            lines = [f"{r.path}|{r.description}" for r in results[:top_k]]
        return "# matches — call execute_tool with one of these paths (schema inline):\n" + "\n".join(lines)

    # --------------------------------------------------- meta-tool: inspect
    def get_tool_schema(self, path: str) -> str:
        """Return a TOON view of the level (what the agent reads)."""
        path = path.strip(". ")
        if path in self._toon_cache:
            return self._toon_cache[path]

        depth = -1 if path == "" else path.count(".")
        if depth == -1:
            out = "# categories (call get_tool_schema on a path to drill in)\n" + toon.encode_categories(self.reg)
        elif depth == 0:
            self.reg.services(path)  # validates / raises
            out = f"# services of {path}\n" + toon.encode_category(self.reg, path)
        elif depth == 1:
            self.reg.functions(path)  # validates / raises
            out = (f"# functions of {path} (path|desc|param:type:req[:default]|r:fields[|risk])\n"
                   + toon.encode_service(self.reg, path))
        else:
            out = toon.encode_function(self.reg.tool(path))

        self._toon_cache[path] = out
        return out

    def schema_dict(self, path: str) -> dict:
        """Structured view (for adapters that want JSON, not TOON)."""
        return self.reg.schema(path)

    # --------------------------------------------------- meta-tool: execute
    def execute_tool(self, path: str, params: dict | None = None):
        tool = self.reg.tool(path)  # raises KeyError if not a function path
        args = self._prepare_args(tool.params, params or {})

        if tool.fn is None:
            raise RuntimeError(f"tool {path!r} has no executor bound (use @sift.tool or sift.bind)")
        raw = tool.fn(**args)
        if not isinstance(raw, dict):
            raise TypeError(f"executor for {path!r} must return a dict, got {type(raw).__name__}")

        # owner-configured projection: transform (reshape) then field whitelist
        result = tool.transform(raw) if tool.transform is not None else raw
        if tool.returns and isinstance(result, dict):
            result = {k: result[k] for k in tool.returns if k in result}
        return result

    @staticmethod
    def _prepare_args(spec: dict[str, Param], params: dict) -> dict:
        out: dict = {}
        for name, p in spec.items():
            val = params.get(name)
            if val in (None, ""):
                if p.required:
                    raise ValueError(f"missing required parameter {name!r} ({p.desc})")
                if p.default != "":
                    out[name] = _coerce(p.type, p.default)
                continue
            out[name] = _coerce(p.type, val)
        return out


def _coerce(typ: str, value):
    if typ == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value
