"""Adapter tests — format conversion + full-loop drive with FAKE clients.

These prove the provider-agnostic contract offline: any client exposing the
expected `create(...)` works. Live integration is smoke-tested separately
(see benchmarks/ / docs) against a real provider.
"""
import json
from types import SimpleNamespace

from sift.adapters.anthropic import anthropic_tools, run_agent as run_anthropic
from sift.adapters.openai import run_agent as run_openai


# --------------------------------------------------------------- format checks

def test_openai_tool_specs(sift):
    specs = sift.openai_tools()
    names = {s["function"]["name"] for s in specs}
    assert names == {"search_tools", "execute_tool"}  # 2 meta-tools
    assert all(s["type"] == "function" for s in specs)


def test_anthropic_tool_format(sift):
    tools = anthropic_tools(sift)
    names = {t["name"] for t in tools}
    assert names == {"search_tools", "execute_tool"}
    # Anthropic uses input_schema, not the OpenAI function wrapper
    for t in tools:
        assert "input_schema" in t
        assert "function" not in t
        assert t["input_schema"]["type"] == "object"


# --------------------------------------------------------- OpenAI-shaped client

class FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _tc(self, name, args):
        return SimpleNamespace(id="c_" + name, type="function",
                               function=SimpleNamespace(name=name, arguments=json.dumps(args)))

    def _resp(self, content, tool_calls):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls))])

    def _create(self, model, messages, tools, extra_body=None):
        last = messages[-1]
        if last["role"] != "tool":
            return self._resp(None, [self._tc("search_tools", {"q": "read email"})])
        if last["name"] == "search_tools":
            path = [ln for ln in last["content"].splitlines() if not ln.startswith("#")][0].split("|")[0]
            return self._resp(None, [self._tc("execute_tool", {"path": path, "params": {"m": 1}})])
        return self._resp("done", None)


def test_openai_run_agent_offline(sift):
    answer = run_openai(sift, FakeOpenAIClient(), "any/model", "read my last email")
    assert answer == "done"


# ------------------------------------------------------ Anthropic-shaped client

class FakeAnthropicClient:
    def __init__(self):
        self.messages = SimpleNamespace(create=self._create)

    def _use(self, name, inp):
        return SimpleNamespace(type="tool_use", id="u_" + name, name=name, input=inp)

    def _text(self, txt):
        return SimpleNamespace(type="text", text=txt)

    def _create(self, model, system, tools, max_tokens, messages, **extra):
        last = messages[-1]
        # first turn: user string; tool results come back as a list of blocks
        if isinstance(last["content"], str):
            return SimpleNamespace(stop_reason="tool_use",
                                   content=[self._use("search_tools", {"q": "read email"})])
        names = [b.get("type") for b in last["content"]] if isinstance(last["content"], list) else []
        if "tool_result" in names:
            # decide based on what we just ran (peek previous assistant block)
            prev = messages[-2]["content"]
            ran = next((b.name for b in prev if getattr(b, "type", None) == "tool_use"), None)
            if ran == "search_tools":
                # grab a path from the tool_result content
                result_text = last["content"][0]["content"]
                path = [ln for ln in result_text.splitlines() if not ln.startswith("#")][0].split("|")[0]
                return SimpleNamespace(stop_reason="tool_use",
                                       content=[self._use("execute_tool", {"path": path, "params": {"m": 1}})])
            return SimpleNamespace(stop_reason="end_turn", content=[self._text("done")])
        return SimpleNamespace(stop_reason="end_turn", content=[self._text("done")])


def test_anthropic_run_agent_offline(sift):
    answer = run_anthropic(sift, FakeAnthropicClient(), "claude-x", "read my last email")
    assert answer == "done"


# ---------------------- Anthropic native tool search (defer_loading) ----------

def test_deferred_tools_shape(sift):
    from sift.adapters.anthropic import deferred_tools

    tools = deferred_tools(sift, keep=("google_workspace.gmail.read",))
    by_name = {t["name"]: t for t in tools}

    search = by_name["search_tools"]
    assert "defer_loading" not in search               # the search tool loads up front
    assert "domain" in search["input_schema"]["properties"]

    kept = by_name["google_workspace__gmail__read"]
    assert kept["defer_loading"] is False              # keep= stays non-deferred
    deferred = by_name["google_workspace__gmail__send"]
    assert deferred["defer_loading"] is True
    assert deferred["input_schema"]["properties"]["to"]["type"] == "string"
    assert "risk" in deferred["description"]           # risk flag surfaces


def test_tool_search_result_returns_references(sift):
    from sift.adapters.anthropic import tool_search_result

    block = tool_search_result(sift, "srch_1", {"domain": "email", "action": "read the inbox"})
    assert block["type"] == "tool_result" and block["tool_use_id"] == "srch_1"
    names = [r["tool_name"] for r in block["content"]]
    assert "google_workspace__gmail__read" in names
    assert all(r["type"] == "tool_reference" for r in block["content"])


class FakeDeferredClient:
    """Speaks the defer_loading protocol: search -> references -> direct call."""

    def __init__(self):
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, model, system, tools, max_tokens, messages, **extra):
        last = messages[-1]
        if isinstance(last["content"], str):   # first turn
            return SimpleNamespace(stop_reason="tool_use", content=[SimpleNamespace(
                type="tool_use", id="s1", name="search_tools",
                input={"domain": "email", "action": "read the latest message"})])
        first = last["content"][0]
        if first["tool_use_id"] == "s1":       # got references back -> call the tool
            name = first["content"][0]["tool_name"]
            return SimpleNamespace(stop_reason="tool_use", content=[SimpleNamespace(
                type="tool_use", id="e1", name=name, input={"m": 1})])
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text="done")])


def test_run_agent_deferred_offline(sift):
    from sift.adapters.anthropic import run_agent_deferred

    assert run_agent_deferred(sift, FakeDeferredClient(), "claude-x", "read my email") == "done"


# ------------------------------------------------- OpenAI Responses API driver

class FakeResponsesClient:
    def __init__(self):
        self.responses = SimpleNamespace(create=self._create)

    def _fc(self, call_id, name, args):
        return SimpleNamespace(type="function_call", call_id=call_id,
                               name=name, arguments=json.dumps(args))

    def _create(self, model, instructions, input, tools, **kw):
        last = input[-1]
        if isinstance(last, dict) and last.get("type") == "function_call_output":
            if input[-2]["name"] == "search_tools":
                path = [ln for ln in last["output"].splitlines()
                        if not ln.startswith("#")][0].split("|")[0]
                return SimpleNamespace(output=[
                    self._fc("c2", "execute_tool", {"path": path, "params": {"m": 1}})])
            return SimpleNamespace(output=[SimpleNamespace(type="message")],
                                   output_text="done")
        return SimpleNamespace(output=[self._fc("c1", "search_tools", {"q": "read email"})])


def test_openai_responses_run_agent_offline(sift):
    from sift.adapters.openai import run_agent_responses

    assert run_agent_responses(sift, FakeResponsesClient(), "gpt-x", "read my email") == "done"


# ------------------------------------------------------- Gemini-shaped client

class FakeGeminiClient:
    def __init__(self):
        self.models = SimpleNamespace(generate_content=self._generate)

    def _fc(self, name, args):
        part = SimpleNamespace(function_call=SimpleNamespace(name=name, args=args))
        content = SimpleNamespace(parts=[part])
        return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text="")

    def _generate(self, model, contents, config):
        last = contents[-1]
        parts = last["parts"] if isinstance(last, dict) else last.parts
        first = parts[0]
        if isinstance(first, dict) and "text" in first:      # opening user turn
            return self._fc("search_tools", {"q": "read email"})
        if isinstance(first, dict) and "function_response" in first:
            resp = first["function_response"]
            if resp["name"] == "search_tools":
                text = resp["response"]["result"]
                path = [ln for ln in text.splitlines() if not ln.startswith("#")][0].split("|")[0]
                return self._fc("execute_tool", {"path": path, "params": {"m": 1}})
            done = SimpleNamespace(parts=[])
            return SimpleNamespace(candidates=[SimpleNamespace(content=done)], text="done")
        return self._fc("search_tools", {"q": "read email"})


def test_gemini_run_agent_offline(sift):
    from sift.adapters.gemini import gemini_tools, run_agent

    decls = gemini_tools(sift)[0]["function_declarations"]
    assert {d["name"] for d in decls} == {"search_tools", "execute_tool"}
    assert run_agent(sift, FakeGeminiClient(), "gemini-x", "read my email") == "done"
