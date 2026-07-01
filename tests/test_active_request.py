"""Active Tool Request — structured (domain + action) two-stage routing.

The MCP-Zero idea (a model-authored request beats a raw query), fused over SIFT's
hybrid signals with the (s_server·s_tool)·max(s_server, s_tool) score.
"""
from sift import Sift


def test_search_request_ranks_gold(sift):
    res = sift.search_request("email", "read messages in the inbox", top_k=3)
    assert res
    assert res[0].path == "google_workspace.gmail.read"


def test_domain_breaks_the_tie(sift):
    """'read' matches both gmail.read and filesystem.read; the email domain lifts
    gmail above the filesystem tool (s_server=0 zeroes the filesystem score)."""
    res = sift.search_request("email", "read", top_k=3)
    assert res[0].path == "google_workspace.gmail.read"


def test_dispatch_routes_active_request(sift):
    out = sift.dispatch("search_tools", {"domain": "email", "action": "read messages in the inbox"})
    assert "google_workspace.gmail.read" in out
    assert "r:" in out  # schema comes back inline, like a normal search


def test_action_only_falls_back_to_query(sift):
    res = sift.search_request("", "read my last email", top_k=3)
    assert any(r.path == "google_workspace.gmail.read" for r in res)


def test_domain_only_falls_back_to_query(sift):
    res = sift.search_request("read emails inbox", "", top_k=3)
    assert any(r.path == "google_workspace.gmail.read" for r in res)


def test_scope_enforced_on_active_request(sift):
    view = sift.scope(allow=["local.*"])
    out = view.dispatch("search_tools", {"domain": "filesystem", "action": "read a text file"})
    assert "google_workspace" not in out          # denied by scope
    assert "local.filesystem.read" in out


def test_unmatched_domain_degrades_to_action(sift):
    """When the domain hint matches no service (lexical miss, no embedder to
    paraphrase), don't let an arbitrary 0-score tie surface a wrong/risky tool —
    fall back to ranking on the action alone."""
    s = Sift(retrieval="bm25")

    @s.tool("google_workspace.gmail.read", description="Read emails from the inbox",
            params={"m": "number:o:10:max"}, returns=["id"])
    def _r(m=10):
        return {"id": "1"}

    @s.tool("crm.contacts.delete", description="Delete a contact",
            params={"id": "string:n::id"}, returns=["ok"], risk=True)
    def _d(id):
        return {"ok": True}

    s.build_index()
    # "email" doesn't lexically match "emails"; without the guard the 0-score tie
    # would put crm.contacts.delete (sorts first) on top. The action must win.
    res = s.search_request("email", "read the latest message", top_k=2)
    assert res[0].path == "google_workspace.gmail.read"


def test_active_request_bm25_only():
    """No embedder: the routing runs on normalised BM25 alone."""
    s = Sift(retrieval="bm25")

    @s.tool("mail.gmail.read", description="Read emails from the inbox",
            params={}, returns=["id"])
    def _r():
        return {"id": "1"}

    @s.tool("files.disk.read", description="Read a file from local disk",
            params={"path": "string:n::path"}, returns=["path"])
    def _f(path):
        return {"path": path}

    s.build_index()
    res = s.search_request("mail", "read the inbox", top_k=2)
    assert res[0].path == "mail.gmail.read"
