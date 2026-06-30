"""Run SIFT AS an MCP server — define/import your tools here, then serve.

    python examples/serve_mcp.py          # stdio (Claude Desktop, local clients)
    python examples/serve_mcp.py sse      # HTTP/SSE (OpenWebUI, remote clients)

Requires the mcp extra:  pip install "sift-tools[mcp]"

The MCP client sees only the 3 meta-tools (search_tools / get_tool_schema /
execute_tool) and discovers your whole catalogue through them.
"""
import sys

from sift import Sift

sift = Sift()


@sift.tool("google_workspace.gmail.read", description="Read emails from the inbox",
           params={"m": "number:o:10:max"}, returns=["id", "subject", "from", "snippet"])
def gmail_read(m=10):
    # replace with a real Gmail API call
    return {"id": "1", "subject": "Meeting", "from": "joao@acme.com",
            "snippet": "Confirming.", "body": "filtered out"}


@sift.tool("web.search.run", description="Search the web", params={"q": "string:n::query"},
           returns=["title", "url", "snippet"])
def web_search(q):
    return {"title": f"Result for {q}", "url": "https://example.com", "snippet": "..."}


# Import an existing ecosystem instead of (or alongside) hand-written tools:
# from sift.importers.openapi import register_openapi, httpx_request
# register_openapi(sift, spec, category="acme", request=httpx_request("https://api.acme.com"))
# from sift.importers.mcp_proxy import connect_mcp_stdio
# connect_mcp_stdio(sift, "npx", ["-y", "@modelcontextprotocol/server-github"],
#                   category="integrations", service="github")

sift.build_index()

transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
print(f"serving SIFT over MCP ({transport}) ...", file=sys.stderr)
sift.serve_mcp(transport=transport)
