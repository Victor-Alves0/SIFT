"""Constrained-decoding helpers for weak / local models.

Small models often can't emit valid tool calls on their own. Feeding a JSON
schema (Outlines, LM Format Enforcer, vLLM ``guided_json``) or a GBNF grammar
(llama.cpp) to the decoder *guarantees* parseable output. These helpers produce
both for SIFT's prompted protocol.
"""
from __future__ import annotations

from .metatools import META_TOOL_NAMES


def tool_call_json_schema() -> dict:
    """JSON Schema for one prompted step: a tool call OR a final answer.

    Hand this to a structured-output decoder so even a small model can only emit
    ``{"tool": ..., "args": {...}}`` or ``{"answer": "..."}``.
    """
    return {
        "title": "SiftStep",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": list(META_TOOL_NAMES)},
                    "args": {"type": "object"},
                },
                "required": ["tool", "args"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        ],
    }


# A standard JSON grammar (llama.cpp GBNF). Constrains output to valid JSON, which
# is enough to make a weak model's tool call parseable; pair with the prompt
# protocol for the {"tool"|"answer", ...} envelope.
_JSON_GBNF = r"""
root    ::= ws object ws
object  ::= "{" ws ( pair (ws "," ws pair)* )? ws "}"
pair    ::= string ws ":" ws value
array   ::= "[" ws ( value (ws "," ws value)* )? ws "]"
value   ::= object | array | string | number | "true" | "false" | "null"
string  ::= "\"" char* "\""
char    ::= [^"\\] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex     ::= [0-9a-fA-F]
number  ::= "-"? int frac? exp?
int     ::= "0" | [1-9] [0-9]*
frac    ::= "." [0-9]+
exp     ::= [eE] [-+]? [0-9]+
ws      ::= [ \t\n\r]*
"""


def json_gbnf() -> str:
    """A GBNF grammar (llama.cpp) constraining generation to valid JSON."""
    return _JSON_GBNF.strip() + "\n"
