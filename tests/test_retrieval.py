"""Hybrid retrieval: BM25, RRF fusion, bm25-only mode, reranker hook."""
from sift import Sift
from sift.retrieval import BM25, rank_order, rrf


def test_bm25_ranks_relevant():
    docs = ["read emails from the gmail inbox", "list files in google drive", "search the web"]
    scores = BM25(docs).scores("read my email")
    assert rank_order(scores)[0] == 0


def test_rrf_fuses():
    fused = rrf([[2, 0, 1], [0, 2, 1]])  # docs 0 and 2 are top-1 in one list each
    order = sorted(fused, key=lambda i: fused[i], reverse=True)
    assert set(order[:2]) == {0, 2}
    assert order[-1] == 1  # doc 1 is mid/low in both


def _bm25_sift() -> Sift:
    s = Sift(retrieval="bm25")  # no embedder => no model download

    @s.tool("google_workspace.gmail.read", description="Read emails from the inbox",
            params={"m": "number:o:10:max"}, returns=["id"])
    def _r(m=10):
        return {"id": "1"}

    @s.tool("google_workspace.drive.list", description="List files in Google Drive",
            params={"q": "string:o::filter"}, returns=["id"])
    def _d(q=""):
        return {"id": "2"}

    return s.build_index()


def test_bm25_only_search():
    s = _bm25_sift()
    res = s.search_tools("read my email inbox", top_k=1)
    assert res[0].path == "google_workspace.gmail.read"


class _FakeReranker:
    """Always prefers documents mentioning 'drive'."""
    def rerank(self, query, docs):
        return [1.0 if "drive" in d.lower() else 0.0 for d in docs]


def test_reranker_reorders():
    s = Sift(retrieval="bm25", reranker=_FakeReranker())

    @s.tool("mail.gmail.read", description="Read the list of emails in the inbox", params={}, returns=["id"])
    def _r():
        return {"id": "1"}

    @s.tool("storage.drive.list", description="List files in Google Drive", params={}, returns=["id"])
    def _d():
        return {"id": "2"}

    s.build_index()
    # both docs match "list" (zero-score docs are no longer rerank candidates);
    # BM25 alone leans mail ("emails list"), but the reranker forces drive on top
    res = s.search_tools("emails list", top_k=2)
    assert "drive" in res[0].path


def test_bm25_zero_scores_return_nothing():
    """An all-zero BM25 tie is 'nothing matched', not an arbitrary winner —
    even without a min_score configured."""
    s = _bm25_sift()
    assert s.search_tools("zzqq xyzzy nomatch", top_k=3) == []


def test_stemming_matches_inflections():
    docs = ["delete a calendar event", "read emails from the inbox"]
    b = BM25(docs)
    assert rank_order(b.scores("deleted events"))[0] == 0   # deleted~delete, events~event
    assert b.scores("email")[1] > 0                          # email~emails


def test_min_score_returns_no_match():
    s = Sift(retrieval="bm25", min_score=0.1)

    @s.tool("mail.gmail.read", description="Read emails from the inbox", params={}, returns=["id"])
    def _r():
        return {"id": "1"}

    s.build_index()
    # gibberish shares no terms with any tool -> no match
    assert s.search_tools("zzqq xyzzy nomatch", top_k=3) == []
    assert "no matching tools" in s.dispatch("search_tools", {"q": "zzqq xyzzy nomatch"})
    # a real query still works
    assert s.search_tools("read emails inbox", top_k=1)
