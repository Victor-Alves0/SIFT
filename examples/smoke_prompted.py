"""Live smoke of the PROMPTED adapter: drive SIFT with plain text completions
(no native tool calling used at all). Proves models without function-calling work.
"""
import os
from pathlib import Path

from openai import OpenAI

from sift import Sift
from sift.adapters.prompted import run_agent, single_decision

ROOT = Path(__file__).resolve().parents[1]
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

KEY = os.environ["OPENROUTER_API"]
MODEL = os.environ.get("SIFT_MODEL", "deepseek/deepseek-v4-flash")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)


def generate(prompt: str) -> str:
    # plain text completion — NO tools param: emulates a model without tool calling
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    return resp.choices[0].message.content or ""


sift = Sift()


@sift.tool("google_workspace.gmail.read", description="Read emails from the inbox",
           params={"m": "number:o:10:max"}, returns=["id", "subject", "from"])
def gmail_read(m=10):
    return {"id": "1", "subject": "Meeting", "from": "joao@acme.com", "body": "secret"}


sift.build_index()

print("=== prompted run_agent (text protocol) ===")
answer = run_agent(sift, generate, "qual foi meu último email?", verbose=True)
print("ANSWER:", answer)

print("\n=== single_decision (weakest models) ===")
out = single_decision(sift, generate, "leia meu último email")
print("DECISION:", out["path"], out["args"], "->", out["result"])
print("\nPASS ✓" if answer and out["result"].get("subject") else "CHECK")
