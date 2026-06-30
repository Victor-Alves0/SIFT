"""BFCL-style accuracy harness (offline with a scripted model)."""
import json

from sift.evalsuite import Case, bfcl_style


def _fake_model(decider):
    """decider(query) -> (encoded_name_or_None, args_dict)."""
    def call_model(messages, tools):
        query = messages[-1]["content"]
        name, args = decider(query)
        tool_calls = []
        if name:
            tool_calls = [{"id": "1", "type": "function",
                           "function": {"name": name, "arguments": json.dumps(args)}}]
        return {"choices": [{"message": {"content": None, "tool_calls": tool_calls}}]}
    return call_model


def test_bfcl_all_correct(sift):
    cases = [Case("read my last email", "google_workspace.gmail.read", {"m": 1})]
    cm = _fake_model(lambda q: ("google_workspace__gmail__read", {"m": 1}))
    rep = bfcl_style(cm, sift.registry, cases)
    assert rep.function_acc == 1.0
    assert rep.arg_acc == 1.0
    assert rep.no_call_rate == 0.0


def test_bfcl_wrong_tool_and_bad_args(sift):
    cases = [
        Case("read my last email", "google_workspace.gmail.read", {"m": 1}),
        Case("send an email", "google_workspace.gmail.send"),
    ]
    # always returns gmail.read with wrong arg -> 2nd case wrong tool, 1st bad args
    cm = _fake_model(lambda q: ("google_workspace__gmail__read", {"m": 99}))
    rep = bfcl_style(cm, sift.registry, cases)
    assert rep.function_acc == 0.5   # right tool only for case 1
    assert rep.arg_acc == 0.0        # case 1 had wrong m
