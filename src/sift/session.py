"""Per-conversation tool memory — discovered tools become real tools.

Without a session, the model re-searches for the same tool every turn. A
``SiftSession`` records which tools discovery surfaced and, from the next turn
on, *promotes* them to first-class function specs (the same pattern Anthropic's
tool search uses when it expands ``tool_reference`` blocks): the model calls the
tool directly, no search round-trip.

    session = sift.session()                      # or SiftSession(scope)
    tools  = session.tools()                      # meta-tools + promoted tools
    out    = session.dispatch(name, args)         # records discoveries, routes
                                                  # promoted names automatically

Rebuild ``tools()`` each turn — it grows as the conversation discovers more.
Promoted tool names are the dotted path with ``.`` → ``__`` (LLM tool-name rules
don't allow dots): ``google_workspace.gmail.read`` → ``google_workspace__gmail__read``.
"""
from __future__ import annotations

import json

from .registry import input_schema_for


def promoted_name(path: str) -> str:
    return path.replace(".", "__")


def function_spec(tool) -> dict:
    """OpenAI-style function spec for a concrete tool (used by pinned tools and
    session promotion): named by its ``.`` → ``__`` path, risk flagged."""
    return {"type": "function", "function": {
        "name": promoted_name(tool.path),
        "description": tool.description + (" [risk: confirm first]" if tool.risk else ""),
        "parameters": input_schema_for(tool),
    }}


class SiftSession:
    """Wraps a ``Sift`` or ``SiftScope``; same ``dispatch`` contract."""

    def __init__(self, target, *, max_promoted: int = 10) -> None:
        self._target = target
        self._max = max_promoted
        self._discovered: dict[str, None] = {}   # ordered set of paths

    # ------------------------------------------------------------- inspection
    @property
    def discovered(self) -> list[str]:
        return list(self._discovered)

    def _registry(self):
        reg = getattr(self._target, "registry", None)
        return reg if reg is not None else self._target._sift.registry  # SiftScope

    # ---------------------------------------------------------------- surface
    def tools(self) -> list[dict]:
        """Current OpenAI-style specs: the 2 meta-tools + one real spec per
        discovered tool (capped at ``max_promoted``, most recent kept)."""
        specs = list(self._target.openai_tools())
        reg = self._registry()
        pinned = set(getattr(self._sift(), "_pinned", []))
        for path in list(self._discovered)[-self._max:]:
            if path in pinned:
                continue  # already a first-class spec via pin(); don't duplicate
            specs.append(function_spec(reg.tool(path)))
        return specs

    def _sift(self):
        return getattr(self._target, "_sift", self._target)  # Sift or SiftScope's parent

    @property
    def system_prompt(self) -> str:
        return self._target.system_prompt

    def dispatch(self, name: str, arguments: dict | str) -> str:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        if "__" in name:  # a promoted tool called directly by its flat name
            path = name.replace("__", ".")
            return self._target.dispatch("execute_tool", {"path": path, "params": args})
        out = self._target.dispatch(name, args)
        if name == "search_tools":
            self._record(out)
        return out

    # ---------------------------------------------------------------- helpers
    def _record(self, toon_text: str) -> None:
        """Remember function paths surfaced by a search (TOON: path is the first
        ``|`` field of each non-comment line)."""
        reg = self._registry()
        for line in toon_text.splitlines():
            if line.startswith("#") or "|" not in line:
                continue
            path = line.split("|", 1)[0].strip()
            if path.count(".") == 2:
                try:
                    reg.tool(path)
                except KeyError:
                    continue
                self._discovered[path] = None
