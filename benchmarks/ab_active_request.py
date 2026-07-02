"""A/B: raw-query retrieval vs the active tool request (domain + action).

Multi-domain catalogue with deliberate verb collisions (read/list/send/delete
across gmail, calendar, drive, slack, crm, jira). For each case we compare top-1
routing accuracy of:

  - query-only:      search_tools(raw_user_query)
  - active request:  search_request(domain, action)

Both forms carry the same information; the active request just structures it the
way a model would author it (the MCP-Zero finding). Run:

    python benchmarks/ab_active_request.py           # hybrid (downloads the model once)
    python benchmarks/ab_active_request.py bm25      # lexical only, no download

Caveat: this is a small, author-constructed catalogue — treat the numbers as
directional, not as an independent benchmark.
"""
import sys

from sift import Sift

RETRIEVAL = sys.argv[1] if len(sys.argv) > 1 else "hybrid"

CATALOG = {
    "google_workspace.gmail.read":      "Read messages from the email inbox",
    "google_workspace.gmail.send":      "Send an email message to a recipient",
    "google_workspace.gmail.delete":    "Delete an email message from the mailbox",
    "google_workspace.calendar.list":   "List upcoming calendar events",
    "google_workspace.calendar.create": "Create a new calendar event / meeting",
    "google_workspace.calendar.delete": "Delete or cancel a calendar event",
    "google_workspace.drive.read":      "Read the contents of a file in Drive",
    "google_workspace.drive.list":      "List files and folders in Drive",
    "google_workspace.drive.delete":    "Delete a file from Drive",
    "slack.messages.send":              "Post a message to a Slack channel",
    "slack.messages.read":              "Read recent messages from a Slack channel",
    "crm.contacts.read":                "Read a contact record from the CRM",
    "crm.contacts.create":              "Create a new contact in the CRM",
    "crm.contacts.delete":              "Delete a contact from the CRM",
    "jira.issues.create":               "Create a new Jira issue / ticket",
    "jira.issues.list":                 "List Jira issues in a project",
    "jira.issues.delete":               "Delete a Jira issue",
}

# (raw user query, domain, action, gold path)
CASES = [
    ("cancel my 3pm",                  "calendar", "cancel an event",       "google_workspace.calendar.delete"),
    ("what's on my agenda",            "calendar", "list upcoming events",  "google_workspace.calendar.list"),
    ("drop that contact",              "crm",      "delete a contact",      "crm.contacts.delete"),
    ("ping the team on slack",         "slack",    "send a message",        "slack.messages.send"),
    ("remove that ticket",             "jira",     "delete an issue",       "jira.issues.delete"),
    ("get rid of that file",           "drive",    "delete a file",         "google_workspace.drive.delete"),
    ("open that doc",                  "drive",    "read a file",           "google_workspace.drive.read"),
    ("did anyone message me on slack", "slack",    "read channel messages", "slack.messages.read"),
    ("log a new bug",                  "jira",     "create an issue",       "jira.issues.create"),
    ("add someone to the crm",         "crm",      "create a contact",      "crm.contacts.create"),
    ("trash that email",               "email",    "delete a message",      "google_workspace.gmail.delete"),
    ("shoot them an email",            "email",    "send a message",        "google_workspace.gmail.send"),
    ("check my inbox",                 "email",    "read messages",         "google_workspace.gmail.read"),
    ("set up a meeting",               "calendar", "create an event",       "google_workspace.calendar.create"),
]


def main() -> None:
    sift = Sift(retrieval=RETRIEVAL)
    for path, desc in CATALOG.items():
        sift.add_tool(path, (lambda: {"ok": True}), description=desc, returns=["ok"])
    sift.build_index()

    def top_function(results):
        """The agent-facing view: dispatch renders FUNCTIONS only (services are
        navigation nodes), so top-1 here means 'first executable match'."""
        return [r for r in results if r.kind == "function"][:1]

    q_hits = a_hits = 0
    print(f"retrieval={RETRIEVAL}  catalogue={len(CATALOG)} tools  cases={len(CASES)}\n")
    print(f"{'raw user query':<31}{'q':>2}{'a':>2}  gold")
    for query, domain, action, gold in CASES:
        q_top = top_function(sift.search_tools(query, top_k=4))
        a_top = top_function(sift.search_request(domain, action, top_k=4))
        q_ok = bool(q_top) and q_top[0].path == gold
        a_ok = bool(a_top) and a_top[0].path == gold
        q_hits += q_ok
        a_hits += a_ok
        note = "  <- active fixed it" if (a_ok and not q_ok) else ""
        print(f"{query:<31}{'Q' if q_ok else '.':>2}{'A' if a_ok else '.':>2}  {gold}{note}")

    n = len(CASES)
    print(f"\nTOP-1  query-only: {q_hits}/{n} = {q_hits / n:.0%}"
          f"   active request: {a_hits}/{n} = {a_hits / n:.0%}")


if __name__ == "__main__":
    main()
