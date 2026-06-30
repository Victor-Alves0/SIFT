"""Live smoke of CODE MODE: the model writes one snippet that calls several
tools in a single turn (via OpenRouter). Proves run_code end-to-end."""
import json
import os
from pathlib import Path

from openai import OpenAI

from sift import Sift

ROOT = Path(__file__).resolve().parents[1]
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

KEY = os.environ["OPENROUTER_API"]
MODEL = os.environ.get("SIFT_MODEL", "deepseek/deepseek-v4-flash")

sift = Sift()


@sift.tool("google_workspace.gmail.read", description="Read emails from the inbox",
           params={"m": "number:o:10:max"}, returns=["id", "subject", "from"])
def gmail_read(m=10):
    return {"id": "1", "subject": "Meeting", "from": "joao@acme.com", "body": "x"}


@sift.tool("crm.contacts.find", description="Find a CRM contact by email",
           params={"email": "string:n::email"}, returns=["name", "company"])
def crm_find(email):
    return {"name": "João Silva", "company": "Acme", "phone": "secret"}


sift.build_index()

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=KEY)
messages = [
    {"role": "system", "content": sift.code_system_prompt},
    {"role": "user", "content": "leia meu último email e descubra o nome e a empresa de quem enviou"},
]
tools = sift.code_tools()
used_code = False

for _ in range(8):
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tools,
                                          extra_body={"reasoning": {"effort": "low"}})
    msg = resp.choices[0].message
    assistant = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        assistant["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    messages.append(assistant)

    if not msg.tool_calls:
        print("\nANSWER:", msg.content)
        break
    for tc in msg.tool_calls:
        if tc.function.name == "run_code":
            used_code = True
            print("\n--- model wrote code ---")
            print(json.loads(tc.function.arguments).get("code"))
            print("------------------------")
        out = sift.dispatch(tc.function.name, tc.function.arguments)
        print(f"↳ {tc.function.name} => {out[:160]}")
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": out})

print(f"\nused run_code (composed tools in one turn): {used_code}")
print("PASS ✓" if used_code else "NOTE: model answered without code mode")
