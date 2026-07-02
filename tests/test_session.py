"""Session memory: discovered tools get promoted to real specs next turn."""
import json

from sift import Sift
from sift.session import SiftSession, promoted_name


def _sift() -> Sift:
    s = Sift(retrieval="bm25")

    @s.tool("mail.gmail.read", description="Read emails from the inbox",
            params={"m": "number:o:10:max results"}, returns=["id"])
    def _r(m=10):
        return {"id": "1"}

    @s.tool("crm.contacts.delete", description="Delete a contact",
            params={"id": "string:n::contact id"}, returns=["ok"], risk=True)
    def _d(id):
        return {"ok": True}

    return s.build_index()


def test_search_records_discovery_and_promotes():
    session = _sift().session()
    assert len(session.tools()) == 2                     # just the meta-tools

    session.dispatch("search_tools", {"q": "read emails inbox"})
    assert "mail.gmail.read" in session.discovered

    names = [t["function"]["name"] for t in session.tools()]
    assert promoted_name("mail.gmail.read") in names     # promoted next turn
    spec = next(t for t in session.tools()
                if t["function"]["name"] == "mail__gmail__read")
    assert spec["function"]["parameters"]["properties"]["m"]["type"] == "number"


def test_promoted_tool_executes_directly():
    session = _sift().session()
    session.dispatch("search_tools", {"q": "read emails inbox"})
    out = json.loads(session.dispatch("mail__gmail__read", {"m": 1}))
    assert out == {"id": "1"}                            # no re-search needed


def test_promoted_risky_tool_is_flagged():
    session = _sift().session()
    session.dispatch("search_tools", {"q": "delete a contact"})
    spec = next(t for t in session.tools()
                if t["function"]["name"] == "crm__contacts__delete")
    assert "risk" in spec["function"]["description"]


def test_max_promoted_keeps_most_recent():
    session = _sift().session(max_promoted=1)
    session.dispatch("search_tools", {"q": "read emails inbox"})
    session.dispatch("search_tools", {"q": "delete a contact"})
    promoted = [t["function"]["name"] for t in session.tools()
                if "__" in t["function"]["name"]]
    assert promoted == ["crm__contacts__delete"]         # capped, most recent wins


def test_session_over_scope_enforces_allow():
    view = _sift().scope(allow=["mail.*"])
    session = SiftSession(view)
    session.dispatch("search_tools", {"q": "delete a contact"})
    assert "crm.contacts.delete" not in session.discovered   # scope filtered it
    out = json.loads(session.dispatch("crm__contacts__delete", {"id": "1"}))
    assert "not allowed" in out["error"]                     # execute enforced too
