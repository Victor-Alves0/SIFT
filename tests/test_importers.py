"""Importers: MCP listing + OpenAPI spec -> hierarchy nodes."""
from sift import Sift
from sift.importers._common import compress_params, looks_destructive
from sift.importers.mcp import tools_from_listing
from sift.importers.openapi import tools_from_openapi


def test_compress_params():
    schema = {
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": "the query", "default": "is:unread"},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["q"],
    }
    out = compress_params(schema)
    assert out["q"] == {"type": "string", "required": True, "default": "is:unread", "desc": "the query"}
    # integer stays integer since the full boundary type system (was flattened to number)
    assert out["limit"] == {"type": "integer", "required": False, "default": "10", "desc": ""}


def test_destructive_heuristic():
    assert looks_destructive("delete_file")
    assert looks_destructive("send_email")
    assert not looks_destructive("read_file")


def test_mcp_listing_to_tools():
    listing = [
        {"name": "search", "description": "Search the repo",
         "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}},
        {"name": "delete_issue", "description": "Delete an issue", "inputSchema": {}},
    ]
    defs = tools_from_listing(listing, category="integrations", service="github")
    paths = {d.path for d in defs}
    assert "integrations.github.search" in paths
    assert "integrations.github.delete_issue" in paths
    delete = next(d for d in defs if d.path.endswith("delete_issue"))
    assert delete.risk is True


def test_openapi_to_tools_and_register():
    spec = {
        "paths": {
            "/users/{id}": {
                "get": {"operationId": "getUser", "tags": ["users"], "summary": "Get a user",
                        "parameters": [{"name": "id", "required": True, "schema": {"type": "string"}}]},
                "delete": {"operationId": "deleteUser", "tags": ["users"], "summary": "Delete a user",
                           "parameters": [{"name": "id", "required": True, "schema": {"type": "string"}}]},
            }
        }
    }
    defs = tools_from_openapi(spec, category="acme")
    paths = {d.path for d in defs}
    assert "acme.users.getuser" in paths
    assert "acme.users.deleteuser" in paths
    delete = next(d for d in defs if d.path.endswith("deleteuser"))
    assert delete.risk is True  # DELETE method => risky

    # register into a Sift registry
    s = Sift()
    from sift.importers.openapi import register_openapi
    n = register_openapi(s, spec, category="acme")
    assert n == 2
    assert s.registry.tool("acme.users.getuser").description == "Get a user"


def test_openapi_execution_bind():
    spec = {
        "paths": {
            "/users/{id}": {
                "get": {"operationId": "getUser", "tags": ["users"],
                        "parameters": [{"name": "id", "required": True, "schema": {"type": "string"}}]},
            }
        }
    }
    calls = []

    def fake_request(method, route, params):
        calls.append((method, route, params))
        return {"id": params["id"], "name": "Ada"}

    from sift.importers.openapi import register_openapi
    s = Sift()
    register_openapi(s, spec, category="acme", request=fake_request)

    tool = s.registry.tool("acme.users.getuser")
    assert tool.fn is not None
    out = tool.fn(id="7")
    assert out == {"id": "7", "name": "Ada"}
    assert calls == [("GET", "/users/{id}", {"id": "7"})]


def test_mcp_execution_bind():
    listing = [{"name": "search", "description": "Search",
                "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}}]

    def fake_exec(tool_name, params):
        return {"tool": tool_name, "q": params.get("q")}

    from sift.importers.mcp import register_listing
    s = Sift()
    register_listing(s, listing, category="integrations", service="github", executor=fake_exec)
    tool = s.registry.tool("integrations.github.search")
    assert tool.fn(q="bug") == {"tool": "search", "q": "bug"}
