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

import json
from dataclasses import dataclass
from inspect import isawaitable as _isawaitable

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
                 retrieval: str = "hybrid", reranker=None, min_score: float = 0.0,
                 on_risky=None, on_result=None) -> None:
        if retrieval not in ("hybrid", "embedding", "bm25"):
            raise ValueError("retrieval must be 'hybrid', 'embedding' or 'bm25'")
        if retrieval in ("hybrid", "embedding") and embedder is None:
            raise ValueError(f"retrieval={retrieval!r} needs an embedder")
        self.reg = registry
        self.embedder = embedder
        self.retrieval = retrieval
        self.reranker = reranker  # object with .rerank(query, docs) -> list[float]
        # human-in-the-loop guard: called before a risk=True tool runs, with the
        # prepared args; return False (or raise) to block. None = risk tools run.
        self.on_risky = on_risky
        # global post-filter applied to EVERY tool result after projection —
        # e.g. prompt-injection scrubbing of untrusted tool output. (path, result) -> result
        self.on_result = on_result
        self._result_cache: dict = {}   # (path, params-json) -> (expiry, result)
        # Relevance floor; below it, search returns nothing. Same scale in both
        # search modes (see _passes_floor): max embedding cosine when an embedder
        # exists, else "any BM25 term matched".
        self.min_score = min_score
        self._entries: list = []
        self._vectors: list = []
        self._bm25: BM25 | None = None
        self._toon_cache: dict[str, str] = {}

    # ------------------------------------------------------------- index
    def build_index(self, *, cache: "str | None" = None) -> "Gateway":
        """Build (or load) the search index.

        ``cache`` is a file path for persisting document vectors: on a hit
        (same texts + same embedding model) the vectors are loaded instead of
        re-embedded — cold starts drop from tens of seconds to milliseconds on
        large catalogues. The BM25 side is always rebuilt (pure Python, fast).
        """
        entries = self.reg.search_entries()
        if not entries:
            raise ValueError("registry is empty — register tools before build_index()")
        self._entries = entries
        if self.retrieval in ("hybrid", "embedding"):
            self._vectors = self._load_or_embed([e.text for e in entries], cache)
            # remember text->vector so a REBUILD (tools added later) only embeds
            # what actually changed — incremental, not from scratch
            self._text_vectors = dict(zip((e.text for e in entries), self._vectors))
        if self.retrieval in ("hybrid", "bm25"):
            self._bm25 = BM25([e.lex for e in entries])
        return self

    def _load_or_embed(self, texts: list[str], cache: "str | None"):
        if cache is None:
            # incremental path: vectors carried over from a previous build (set
            # via reuse_vectors) are kept; only new/changed texts get embedded
            known = dict(getattr(self, "_reuse_vectors", None) or {})
            missing = [t for t in dict.fromkeys(texts) if t not in known]
            if missing:
                known.update(zip(missing, self.embedder.embed(missing)))
            return [known[t] for t in texts]
        import hashlib
        from pathlib import Path

        import numpy as np

        model = getattr(self.embedder, "model_name", type(self.embedder).__name__)
        key = hashlib.sha256(("\x00".join(texts) + "\x01" + model).encode()).hexdigest()
        path = Path(cache)
        if path.exists():
            try:
                with path.open("rb") as fh:
                    data = np.load(fh, allow_pickle=False)
                    if str(data["key"]) == key:
                        return list(data["vectors"])
            except Exception:  # corrupt/old cache -> silently re-embed below
                pass
        vectors = self.embedder.embed(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            np.savez(fh, key=np.array(key), vectors=np.stack([np.asarray(v) for v in vectors]))
        return vectors

    def _embed_query(self, query: str):
        """Query-side embedding — uses the embedder's asymmetric ``embed_query``
        when it has one (E5-style prefixes), else plain ``embed``."""
        fn = getattr(self.embedder, "embed_query", self.embedder.embed)
        return fn([query])[0]

    # --------------------------------------------------- meta-tool: search
    def search_tools(self, query: str, top_k: int = 5, *, predicate=None) -> list[SearchResult]:
        if not self._entries:
            raise RuntimeError("index not built — call build_index() first")
        top_k = max(1, top_k)

        emb_scores = bm_scores = None
        if self.retrieval in ("hybrid", "embedding"):
            qv = self._embed_query(query)
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
            # zero is "no term matched", not a rank — an all-zero tie must not
            # surface an arbitrary tool as if it were relevant
            fused = {i: s for i, s in enumerate(bm_scores) if s > 0}
            if not fused:
                return []
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
        if not a_rel or max(a_rel) <= 0.0:   # zero signal on the action -> no matches
            return []
        if not self._passes_floor(action):
            return []
        d_rel = self._relevance(domain)
        svc_score = {e.path: d_rel[i] for i, e in enumerate(self._entries)
                     if e.kind == "service"}

        # If the domain hint resonates with no service, every function would tie
        # at 0 and ordering would be arbitrary — a wrong (possibly risky) tool
        # could surface on top. Degrade gracefully to ranking on the action alone
        # (functions only, to match the normal request output).
        if not svc_score or max(svc_score.values()) <= 0.0:
            res = self.search_tools(action, top_k=top_k * 4, predicate=predicate)
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

    def _passes_floor(self, query: str) -> bool:
        """The relevance floor on the SAME scale as ``search_tools`` uses: max
        embedding cosine when an embedder exists, else "any BM25 term matched" —
        so one ``min_score`` value works across both search modes."""
        if self.min_score <= 0:
            return True
        if self.retrieval in ("hybrid", "embedding"):
            qv = self._embed_query(query)
            return max(cosine(qv, v) for v in self._vectors) >= self.min_score
        return max(self._bm25.scores(query)) > 0

    def _relevance(self, query: str) -> list[float]:
        """Per-entry relevance in [0, 1] for ``query`` — the hybrid signal used by
        the multiplicative request-routing score (embedding cosine blended with
        max-normalised BM25). Magnitudes matter here, so this does NOT use RRF."""
        emb = bm = None
        if self.retrieval in ("hybrid", "embedding"):
            qv = self._embed_query(query)
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
        results = self.search_tools(query, top_k=top_k * 4, predicate=predicate)
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
    def get_tool_schema(self, path: str, *, predicate=None) -> str:
        """Return a TOON view of the level (what the agent reads).

        ``predicate`` scopes the browse: nodes whose tools it rejects are omitted
        entirely (a scoped model must not even see denied schemas). Predicated
        views bypass the cache, which only holds the unscoped rendering.
        """
        path = path.strip(". ")
        if predicate is None and path in self._toon_cache:
            return self._toon_cache[path]

        depth = -1 if path == "" else path.count(".")
        if depth == -1:
            out = ("# categories (browse deeper with search_tools(path=...))\n"
                   + toon.encode_categories(self.reg, predicate=predicate))
        elif depth == 0:
            self.reg.services(path)  # validates / raises
            out = f"# services of {path}\n" + toon.encode_category(self.reg, path, predicate=predicate)
        elif depth == 1:
            self.reg.functions(path)  # validates / raises
            out = (f"# functions of {path} (path|desc|param:type:req[:default]|r:fields[|risk])\n"
                   + toon.encode_service(self.reg, path, predicate=predicate))
        else:
            if predicate is not None and not predicate(path):
                raise PermissionError(f"tool {path!r} is not visible in this scope")
            out = toon.encode_function(self.reg.tool(path))

        if predicate is None:
            self._toon_cache[path] = out
        return out

    def invalidate_schema_cache(self) -> None:
        """Drop cached TOON renderings — call after mutating a registered tool
        (e.g. ``set_response``) so stale schemas aren't served."""
        self._toon_cache.clear()

    def browse(self, path: str, top_k: int = 3, *, predicate=None) -> str:
        """Model-facing browse: list a hierarchy level, but if ``path`` isn't a
        real category/service/function, treat the guess as a search query instead
        of erroring (the model often guesses a plausible category name — this
        saves the wasted round-trip). Empty path still lists categories."""
        p = path.strip(". ")
        try:
            return self.get_tool_schema(p, predicate=predicate)
        except KeyError:
            if p:  # a guessed name that isn't a real node -> search on it
                return self.search_compact(p, top_k, predicate=predicate)
            raise

    def schema_dict(self, path: str) -> dict:
        """Structured view (for adapters that want JSON, not TOON)."""
        return self.reg.schema(path)

    # --------------------------------------------------- meta-tool: execute
    def execute_tool(self, path: str, params: dict | None = None):
        tool = self.reg.tool(path)  # raises KeyError if not a function path
        args = self._prepare_args(tool.params, params or {})

        if tool.fn is None:
            raise RuntimeError(f"tool {path!r} has no executor bound (use @sift.tool or sift.bind)")
        cached = self._cache_get(tool, path, args)
        if cached is not None:
            return cached
        self._check_risky(path, tool, args)
        if tool.timeout:
            raw = _run_with_timeout(tool.fn, args, tool.timeout, path)
        else:
            raw = tool.fn(**args)
        if _isawaitable(raw):
            raw.close()  # don't leak the un-awaited coroutine
            raise TypeError(f"tool {path!r} is async — call it via aexecute_tool/adispatch")
        return self._finish(path, tool, args, raw)

    async def aexecute_tool(self, path: str, params: dict | None = None):
        """Async twin of ``execute_tool`` — awaits ``async def`` tools natively
        (sync tools are called inline; offload them yourself if they block)."""
        import asyncio
        tool = self.reg.tool(path)
        args = self._prepare_args(tool.params, params or {})
        if tool.fn is None:
            raise RuntimeError(f"tool {path!r} has no executor bound (use @sift.tool or sift.bind)")
        cached = self._cache_get(tool, path, args)
        if cached is not None:
            return cached
        self._check_risky(path, tool, args)
        raw = tool.fn(**args)
        if _isawaitable(raw):
            if tool.timeout:
                try:
                    raw = await asyncio.wait_for(raw, tool.timeout)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"tool {path!r} exceeded its {tool.timeout}s timeout") from None
            else:
                raw = await raw
        return self._finish(path, tool, args, raw)

    def _check_risky(self, path: str, tool, args: dict) -> None:
        if tool.risk and self.on_risky is not None:
            if not self.on_risky(path, args):
                raise PermissionError(
                    f"risky tool {path!r} was not confirmed (blocked by the on_risky guard)")

    # -------- opt-in result cache (idempotent reads asked repeatedly) --------
    @staticmethod
    def _cache_key(path: str, args: dict) -> str:
        return path + "\x00" + json.dumps(args, sort_keys=True, default=str)

    def _cache_get(self, tool, path: str, args: dict):
        if not tool.cacheable:
            return None
        import time
        hit = self._result_cache.get(self._cache_key(path, args))
        if hit is not None and hit[0] > time.monotonic():
            return hit[1]
        return None

    def _finish(self, path: str, tool, args: dict, raw):
        if not isinstance(raw, dict):
            raise TypeError(f"executor for {path!r} must return a dict, got {type(raw).__name__}")
        # owner-configured projection: transform (reshape) then field whitelist
        result = tool.transform(raw) if tool.transform is not None else raw
        if tool.returns and isinstance(result, dict):
            result = {k: result[k] for k in tool.returns if k in result}
        if self.on_result is not None:   # global post-filter (e.g. injection scrub)
            result = self.on_result(path, result)
        if tool.cacheable:
            import time
            self._result_cache[self._cache_key(path, args)] = (
                time.monotonic() + tool.cache_ttl, result)
        return result

    @staticmethod
    def _prepare_args(spec: dict[str, Param], params: dict) -> dict:
        out: dict = {}
        for name, p in spec.items():
            # only absence/None means "missing" — an explicit "" is a real value
            # (so a model can override a non-empty default with an empty string)
            if name not in params or params[name] is None:
                if p.required:
                    raise ValueError(f"missing required parameter {name!r} ({p.desc})")
                if p.default != "":
                    out[name] = _coerce(p.type, p.default)
                continue
            try:
                out[name] = _coerce(p.type, params[name])
            except ValueError as exc:
                raise ValueError(f"parameter {name!r}: {exc}") from None
        return out


_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off", ""})


def _run_with_timeout(fn, args: dict, timeout: float, path: str):
    """Run a SYNC tool with a wall-clock cap. Honest semantics: Python threads
    can't be force-killed — on timeout the caller gets a TimeoutError and moves
    on, while the orphaned call finishes in the background (daemon thread) and
    its result is discarded. Use it to unblock the agent loop, not to stop the
    underlying work."""
    import threading
    box: dict = {}

    def _target():
        try:
            box["value"] = fn(**args)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"tool {path!r} exceeded its {timeout}s timeout "
                           "(the call keeps running in the background; its result is discarded)")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _coerce(typ: str, value):
    """Coerce a value from the LLM boundary to the declared param type.

    Models routinely send everything as strings ("3", "false", "[1,2]"), so each
    type accepts its string form. An unparseable value raises a clean ValueError
    (surfaced to the model as a structured tool error it can retry from) —
    letting plausible garbage through (``'x' * 4``) propagates hallucination.
    """
    typ = (typ or "").lower()
    if typ in ("integer", "int"):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            raise ValueError(f"expected an integer, got {value!r}") from None
    if typ in ("number", "float"):
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"expected a number, got {value!r}") from None
        # keep integral numbers as int: tools slice/paginate/index with these,
        # and float(3) would break them (ints still work wherever floats do)
        return int(f) if f.is_integer() else f
    if typ in ("boolean", "bool"):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in _TRUE_STRINGS:
                return True
            if v in _FALSE_STRINGS:
                return False
        raise ValueError(f"expected a boolean, got {value!r}")
    if typ in ("array", "object", "list", "dict"):
        want = list if typ in ("array", "list") else dict
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"expected {typ} (JSON), got unparseable {value!r}") from None
        if not isinstance(value, want):
            raise ValueError(f"expected {typ}, got {type(value).__name__}")
        return value
    return value
