# Discovery & retrieval

Discovery is the first half of every agent turn: given a need, surface the few
relevant tools (with their schema) so the model can execute directly. This is the
`search_tools` meta-tool. It has three modes.

## Three ways to call `search_tools`

### 1. Simple query

```python
sift.search_tools("read my last email", top_k=3)   # → list[SearchResult]
sift.dispatch("search_tools", {"q": "read my last email"})   # → TOON string
```

Good for quick, one-shot needs. Ranked by the configured retrieval backend.

### 2. Active tool request (`domain` + `action`)

Instead of one raw query, the model states a **structured intent**:

- **`domain`** — the platform / permission area (`email`, `calendar`, `crm`).
- **`action`** — the operation + target (`read the latest message`).

```python
sift.search_request(domain="email", action="read the latest message")
sift.dispatch("search_tools", {"domain": "email", "action": "read the latest message"})
```

**Why it's better.** A model-authored request aligns more closely with tool
documentation than a user's raw phrasing, which measurably lifts routing accuracy
when the catalogue is large (the [MCP-Zero](https://arxiv.org/abs/2506.01056)
result: query-only retrieval plateaus ~65–72%, structured requests reach ~90%+).

**How SIFT routes it — two stages, fused.**

1. Score the `domain` against every **service** → `s_server`.
2. Score the `action` against every **function** → `s_tool`.
3. Combine per function with MCP-Zero's rule:
   `score = (s_server · s_tool) · max(s_server, s_tool)`.

The twist over MCP-Zero: each `s_*` here is SIFT's **hybrid** signal (local
embedding cosine blended with normalised BM25), not a single dense model behind a
paid API. A useful side effect of the multiplicative score: a strong `domain`
match **zeroes out** tools in the wrong domain even when the `action` matches
them — e.g. `action="read"` matches both `gmail.read` and `filesystem.read`, but
`domain="email"` pushes Gmail to the top (the filesystem tool gets `s_server=0`).

If only one field is given, it degrades gracefully: `action` only → a plain query
search; `domain` only → a plain query search on the domain text.

### 3. Browse the hierarchy (`path`)

No search — just list a level. Useful when the model wants to explore:

```python
sift.dispatch("search_tools", {"path": ""})                       # categories
sift.dispatch("search_tools", {"path": "google_workspace"})       # services
sift.dispatch("search_tools", {"path": "google_workspace.gmail"}) # functions
```

The facade method `sift.get_tool_schema(path)` does the same and is what powers
the browse path internally (also kept as a deprecated `get_tool_schema` alias in
`dispatch` for back-compat).

## What comes back

`search_tools` / `search_request` (the Python methods) return a list of
`SearchResult(path, kind, description, score)`. Through `dispatch` (what an LLM
sees) you get **compact TOON** — the top function matches, one per line, schema
inline, so the model can call `execute_tool` next without a separate inspect step:

```
# matches — call execute_tool with one of these paths (schema inline):
google_workspace.gmail.read|Read emails from the inbox|m:number:o:10|r:id,subject,from,snippet,date
```

If nothing clears the relevance floor, you get an explicit
`# no matching tools — none of the available tools fit this request.` instead of a
misleading nearest-but-wrong tool.

## Retrieval backends

Choose at construction:

```python
Sift(retrieval="hybrid")     # default — embeddings + BM25 fused with RRF
Sift(retrieval="embedding")  # dense only
Sift(retrieval="bm25")       # lexical only — NO model download, zero deps
```

- **hybrid** (default): fuses dense embeddings (paraphrase/semantics) and BM25
  (exact terms — names, ids, rare words) with Reciprocal Rank Fusion, which needs
  no score normalisation or tuning. Best general default.
- **embedding**: dense only. Requires an embedder.
- **bm25**: lexical only. Requires no model at all — fast and dependency-light,
  great for tests, CI, or catalogues with distinctive names.

`hybrid` and `embedding` need an embedder; the default is local `fastembed`. Plug
in any object with `embed(texts) -> list[vector]` (OpenAI, Cohere, a sidecar):

```python
Sift(embedder=my_embedder)
Sift(model_name="BAAI/bge-base-en-v1.5")   # pick a different fastembed model
```

## Reranking (optional)

A cross-encoder reranker re-scores the fused shortlist with a query×document model
— more accurate for the final order than bi-encoder cosine, at the cost of extra
latency and a model download:

```python
from sift.rerank import FastEmbedReranker
sift = Sift(reranker=FastEmbedReranker())   # local, ONNX, no API key
```

Any object with `rerank(query, docs) -> list[float]` works. It applies to both
query search and the active request (reranking on the `action`).

## Relevance floor (`min_score`)

By default discovery always returns its best guesses. Set a floor so that when
*nothing* is actually close, discovery returns nothing — an honest "no matching
tools" instead of the nearest irrelevant tool:

```python
sift = Sift(min_score=0.3)   # cosine floor; tune per embedding model
```

With an embedder the floor is a calibrated cosine in `[0, 1]`; with `bm25` only,
the floor becomes "at least one query term matched".

> **Scale caveat:** the floor is compared against different signals per mode — in
> `search_tools` it's the max embedding cosine; in `search_request` it's the max
> hybrid blend (cosine + normalised BM25)/2. A threshold tuned on one mode does
> not transfer 1:1 to the other; tune against the mode you actually use.

## Tuning checklist

- **Wrong tool chosen?** Improve `description`s and category/service names; prefer
  the active request (`domain` + `action`); add a reranker.
- **Returns an irrelevant tool for off-topic asks?** Set `min_score`.
- **Exact names/ids matter (SKUs, function names)?** Keep `hybrid` (BM25 half
  catches them) or use `bm25`.
- **No network / reproducible tests?** Use `retrieval="bm25"`.
