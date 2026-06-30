"""MCP execution proxy — normalisation of CallToolResult into a plain dict.

(The live stdio connection needs an actual MCP server and the `mcp` extra, so it
is not exercised here; the result-normalisation is the testable pure part.)
"""
from types import SimpleNamespace

from sift.importers import StdioMcpProxy, connect_mcp_stdio  # importable without mcp
from sift.importers.mcp_proxy import _content_to_dict


def _result(texts=(), is_error=False, structured=None):
    content = [SimpleNamespace(type="text", text=t) for t in texts]
    return SimpleNamespace(content=content, isError=is_error, structuredContent=structured)


def test_json_object_passthrough():
    assert _content_to_dict(_result(['{"name": "Ada", "id": 7}'])) == {"name": "Ada", "id": 7}


def test_plain_text_wrapped():
    assert _content_to_dict(_result(["hello"])) == {"content": "hello", "isError": False}


def test_error_flag_preserved():
    out = _content_to_dict(_result(['{"msg": "boom"}'], is_error=True))
    assert out["isError"] is True and out["msg"] == "boom"


def test_json_non_object_wrapped():
    assert _content_to_dict(_result(["[1, 2, 3]"])) == {"result": [1, 2, 3], "isError": False}


def test_structured_content_fallback():
    assert _content_to_dict(_result([], structured={"a": 1})) == {"a": 1}


def test_symbols_exported():
    assert StdioMcpProxy is not None and connect_mcp_stdio is not None
