"""Live smoke test of the provider adapters against a real model (via OpenRouter).

Proves the OpenAI-SDK path and the LangChain path with the real SDKs — not just
fakes. Reads OPENROUTER_API and SIFT_MODEL from .env.

    python examples/smoke_live.py
"""
import os
from pathlib import Path

from sift import Sift

ROOT = Path(__file__).resolve().parents[1]
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

KEY = os.environ["OPENROUTER_API"]
MODEL = os.environ.get("SIFT_MODEL", "deepseek/deepseek-v4-flash")
BASE = "https://openrouter.ai/api/v1"


def build_sift() -> Sift:
    s = Sift()

    @s.tool("google_workspace.gmail.read", description="Read emails from the inbox",
            params={"q": "string:o::query", "m": "number:o:10:max"},
            returns=["id", "subject", "from", "snippet"])
    def gmail_read(q="", m=10):
        return {"id": "1", "subject": "Meeting tomorrow", "from": "joao@acme.com",
                "snippet": "Confirming.", "body": "FILTERED OUT"}

    @s.tool("web.search.run", description="Search the web", params={"q": "string:n::query"},
            returns=["title", "url"])
    def web_search(q):
        return {"title": "result", "url": "https://example.com"}

    s.build_index()
    return s


def smoke_openai_sdk(sift) -> None:
    print("\n=== OpenAI SDK (via OpenRouter) ===")
    from openai import OpenAI
    from sift.adapters.openai import run_agent

    client = OpenAI(base_url=BASE, api_key=KEY)
    answer = run_agent(sift, client, MODEL, "qual foi meu último email?",
                       verbose=True, extra_body={"reasoning": {"effort": "low"}})
    print("ANSWER:", answer)
    assert answer, "empty answer"
    print("PASS ✓")


def smoke_langchain(sift) -> None:
    print("\n=== LangChain (ChatOpenAI -> OpenRouter) ===")
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL, base_url=BASE, api_key=KEY, temperature=0)
    tools = sift.langchain_tools()
    by_name = {t.name: t for t in tools}
    llm_tools = llm.bind_tools(tools)

    messages = [SystemMessage(sift.system_prompt), HumanMessage("qual foi meu último email?")]
    for _ in range(8):
        ai = llm_tools.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            print("ANSWER:", ai.content)
            assert ai.content, "empty answer"
            print("PASS ✓")
            return
        for tc in ai.tool_calls:
            print(f"  ↳ {tc['name']}({tc['args']})")
            result = by_name[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    raise RuntimeError("langchain loop did not finish")


if __name__ == "__main__":
    sift = build_sift()
    results = {}
    for name, fn in (("openai_sdk", smoke_openai_sdk), ("langchain", smoke_langchain)):
        try:
            fn(sift)
            results[name] = "PASS"
        except Exception as exc:
            results[name] = f"FAIL: {exc}"
            print(f"FAIL ✗ {exc}")
    print("\n--- summary ---")
    for k, v in results.items():
        print(f"  {k}: {v}")
