"""Index persistence: vectors cached across builds; invalidated on change."""

from sift import Sift


from conftest import FakeEmbedder


class CountingEmbedder(FakeEmbedder):
    """Text-sensitive embedder that counts embed() calls (cache hit = no call)."""

    model_name = "counting-test-embedder"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return super().embed(texts)


def _sift(embedder, cache) -> Sift:
    s = Sift(embedder=embedder, retrieval="embedding", index_cache=str(cache))

    @s.tool("mail.gmail.read", description="Read emails", params={}, returns=["id"])
    def _r():
        return {"id": "1"}

    return s


def test_second_build_hits_cache(tmp_path):
    cache = tmp_path / "index.npz"
    e1 = CountingEmbedder()
    _sift(e1, cache).build_index()
    assert e1.calls == 1 and cache.exists()

    e2 = CountingEmbedder()          # fresh process, same catalogue
    s2 = _sift(e2, cache).build_index()
    assert e2.calls == 0             # loaded from cache — nothing re-embedded
    assert s2.search_tools("read emails", top_k=1)[0].path == "mail.gmail.read"


def test_cache_invalidated_when_catalogue_changes(tmp_path):
    cache = tmp_path / "index.npz"
    _sift(CountingEmbedder(), cache).build_index()

    e2 = CountingEmbedder()
    s2 = _sift(e2, cache)

    @s2.tool("files.disk.read", description="Read a file", params={}, returns=["p"])
    def _f():
        return {"p": "x"}

    s2.build_index()
    assert e2.calls == 1             # texts changed -> hash mismatch -> re-embed


def test_corrupt_cache_falls_back(tmp_path):
    cache = tmp_path / "index.npz"
    cache.write_bytes(b"not an npz file")
    e = CountingEmbedder()
    _sift(e, cache).build_index()    # must not raise
    assert e.calls == 1
