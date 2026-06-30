"""Run SIFT as an OpenAPI HTTP tool server (for OpenWebUI / REST clients).

    python examples/serve_http.py
    # OpenAPI spec: http://localhost:8000/openapi.json   docs: http://localhost:8000/docs

In OpenWebUI: Settings -> Tools -> add an OpenAPI tool server with this URL.
Set SIFT_API_KEY to require a bearer token. Customize the tools below.
Requires:  pip install "sift-tools[server]"
"""
import os

from sift import Sift

sift = Sift()


@sift.tool("google_workspace.gmail.read", description="Read emails from the inbox",
           params={"m": "number:o:10:max"}, returns=["id", "subject", "from", "snippet"])
def gmail_read(m=10):
    return {"id": "1", "subject": "Meeting", "from": "joao@acme.com",
            "snippet": "Confirming.", "body": "filtered out"}


@sift.tool("web.search.run", description="Search the web", params={"q": "string:n::query"},
           returns=["title", "url", "snippet"])
def web_search(q):
    return {"title": f"Result for {q}", "url": "https://example.com", "snippet": "..."}


sift.build_index()

host = os.getenv("SIFT_HOST", "0.0.0.0")
port = int(os.getenv("SIFT_PORT", "8000"))
print(f"SIFT OpenAPI server on http://{host}:{port}  (docs at /docs)")
sift.serve_http(host=host, port=port)
