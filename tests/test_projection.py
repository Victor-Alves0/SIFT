"""Configurable response projection — trim verbose tool results to fewer tokens."""
from sift import Sift


def test_transform_reshapes_verbose_result():
    s = Sift(retrieval="bm25")

    @s.tool("google_workspace.gmail.query", description="Search emails", params={"q": "string:n::q"})
    def query(q):
        # a verbose MCP-style result
        return {"messages": [{"id": "a", "snippet": "hi", "from": "x"},
                             {"id": "b", "snippet": "yo", "from": "y"}], "nextPageToken": "t"}

    # owner: I only want the ids back (far fewer tokens)
    s.set_response("google_workspace.gmail.query",
                   transform=lambda r: {"ids": [m["id"] for m in r["messages"]]})
    s.build_index()
    assert s.execute_tool("google_workspace.gmail.query", {"q": "x"}) == {"ids": ["a", "b"]}


def test_returns_override_on_existing_tool():
    s = Sift(retrieval="bm25")

    @s.tool("mail.gmail.read", description="Read email", params={})
    def read():
        return {"id": "1", "subject": "s", "body": "big payload", "from": "x"}

    s.set_response("mail.gmail.read", returns=["id", "subject"])
    s.build_index()
    assert s.execute_tool("mail.gmail.read") == {"id": "1", "subject": "s"}


def test_transform_then_whitelist():
    s = Sift(retrieval="bm25")

    @s.tool("a.b.c", description="x", params={})
    def f():
        return {"data": {"id": 1, "x": 2}, "junk": 9}

    s.set_response("a.b.c", transform=lambda r: r["data"], returns=["id"])
    s.build_index()
    assert s.execute_tool("a.b.c") == {"id": 1}
