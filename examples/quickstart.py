"""SIFT quickstart — register tools, build the index, exercise the 2 meta-tools.

Run:  python examples/quickstart.py
(First run downloads the local embedding model.)
"""
from sift import Sift
from sift.bench import Task, run_filter, token_report

sift = Sift()


@sift.tool(
    "google_workspace.gmail.read",
    description="Read emails from the inbox, newest first",
    params={"q": "string:o::Gmail search query (default is:unread)", "m": "number:o:10:max results"},
    returns=["id", "subject", "from", "snippet", "date"],
)
def gmail_read(q="is:unread", m=10):
    # a real impl would call the Gmail API here
    return {"id": "msg_1", "subject": "Meeting tomorrow", "from": "joao@acme.com",
            "snippet": "Confirming our meeting.", "date": "2026-06-30",
            "body": "filtered out by the response whitelist"}


@sift.tool(
    "google_workspace.gmail.send",
    description="Send a new email",
    params={"to": "string:n::recipient", "subject": "string:n::subject", "body": "string:n::body"},
    returns=["id", "status"],
    risk=True,
)
def gmail_send(to, subject, body):
    return {"id": "sent_1", "status": f"sent to {to}"}


@sift.tool(
    "web.search.run",
    description="Search the web and return top results",
    params={"q": "string:n::search query", "n": "number:o:5:number of results"},
    returns=["title", "url", "snippet"],
)
def web_search(q, n=5):
    return {"title": f"Result for {q}", "url": "https://example.com", "snippet": "..."}


sift.build_index()

print("== search_tools (simple query) ==")
for r in sift.search_tools("read my last email", top_k=3):
    print(f"  {r.score:.3f}  {r.path}")

print("\n== search_request (active tool request: domain + action) ==")
for r in sift.search_request("email", "read the latest message", top_k=3):
    print(f"  {r.score:.3f}  {r.path}")

print("\n== get_tool_schema (TOON) ==")
print(sift.get_tool_schema("google_workspace.gmail.read"))

print("\n== execute_tool (filtered) ==")
print(sift.execute_tool("google_workspace.gmail.read", {"m": 1}))

print("\n== token_report ==")
print(token_report(sift.registry).format())

print("== filter metrics ==")
tasks = [
    Task("read my last email", "google_workspace.gmail.read"),
    Task("send a message to someone", "google_workspace.gmail.send", needs_risky=True),
    Task("search the web for news", "web.search.run"),
]
print(run_filter(sift, tasks, top_k=2).format())

print("To wire into an LLM:  sift.openai_tools() / sift.langchain_tools() / sift.serve_mcp()")
