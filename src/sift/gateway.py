"""The gateway: the 3 meta-tools the LLM actually sees.

    search_tools(query)        -> semantic discovery               [Search]
    get_tool_schema(path)      -> hierarchical navigation (TOON)    [Inspect]
    execute_tool(path, params) -> run + response filtering          [Trigger + Filter]

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

    def search_compact(self, query: str, top_k: int = 3, *, predicate=None) -> str:
        """Agent-facing discovery: top function matches rendered as TOON, schema
        inline — so the model can call execute_tool directly (merges Search +
        Inspect). No scores/kind noise; capped to ``top_k`` lines.
        """
        results = self.search_tools(query, top_k=max(top_k * 4, top_k), predicate=predicate)
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
