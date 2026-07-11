"""Catalog quality toolkit — make "write good tools" executable.

Three complementary instruments:

- :func:`lint` — static checks on the registered catalogue (missing/short
  descriptions, undocumented params, near-duplicate tools, TOON bloat, …).
- :func:`selftest` — retrieval self-test: every tool must be findable with its
  OWN description/examples. A tool that can't be found with its own phrasing is
  a tool the model will never find.
- :class:`GapTracker` — an observer that turns production telemetry into
  decisions: which user needs found NO tool (catalogue gaps) and which tools are
  hot enough to :meth:`~sift.Sift.pin` (``suggest_pins``).

    report = quality.lint(sift)
    print(report.format())

    failures = quality.selftest(sift)

    tracker = quality.GapTracker()
    sift = Sift(observer=tracker); ...
    tracker.gaps()          # [(query, count), ...] — needs with no tool
    tracker.suggest_pins()  # [(path, executions), ...] — pin candidates
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class LintIssue:
    severity: str   # "error" | "warn"
    path: str       # tool path or node ("" = catalogue-wide)
    message: str


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity == "warn"]

    def format(self) -> str:
        if not self.issues:
            return "catalog lint: clean — no issues found."
        lines = [f"catalog lint: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        for i in self.issues:
            lines.append(f"  [{i.severity}] {i.path or '<catalog>'}: {i.message}")
        return "\n".join(lines)


def lint(sift, *, dup_threshold: float = 0.92, max_desc_len: int = 200,
         min_desc_len: int = 15) -> LintReport:
    """Static catalogue checks. Requires ``build_index()`` for the duplicate
    check (it reuses the built vectors); everything else works without it."""
    issues: list[LintIssue] = []
    tools = list(sift.registry.tools())

    for t in tools:
        if not t.description.strip():
            issues.append(LintIssue("error", t.path, "missing description — undiscoverable"))
        elif len(t.description) < min_desc_len:
            issues.append(LintIssue("warn", t.path,
                                    f"description is very short ({len(t.description)} chars) — "
                                    "retrieval has little to match on"))
        elif len(t.description) > max_desc_len:
            issues.append(LintIssue("warn", t.path,
                                    f"description is long ({len(t.description)} chars) — "
                                    "bloats every TOON line it appears in"))
        undocumented = [p.name for p in t.params.values() if not p.desc]
        if undocumented:
            issues.append(LintIssue("warn", t.path,
                                    f"param(s) without description: {', '.join(undocumented)} "
                                    "(the model fills these by guessing)"))

    # fragmentation: a category with a single tool wastes a hierarchy level
    cats = Counter(t.parts[0] for t in tools)
    for cat, n in sorted(cats.items()):
        if n == 1:
            issues.append(LintIssue("warn", cat,
                                    "category has a single tool — consider folding it "
                                    "into a broader category"))

    # near-duplicates: two tools the retriever can't reliably tell apart
    gateway = getattr(sift, "_gateway", None)
    if gateway is not None and gateway._vectors:
        import numpy as np
        fn_idx = [i for i, e in enumerate(gateway._entries) if e.kind == "function"]
        if len(fn_idx) >= 2:
            m = np.stack([np.asarray(gateway._vectors[i], dtype=np.float32) for i in fn_idx])
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            sims = (m / norms) @ (m / norms).T
            for a in range(len(fn_idx)):
                for b in range(a + 1, len(fn_idx)):
                    if sims[a, b] >= dup_threshold:
                        pa = gateway._entries[fn_idx[a]].path
                        pb = gateway._entries[fn_idx[b]].path
                        issues.append(LintIssue(
                            "warn", pa,
                            f"near-duplicate of {pb} (cosine {sims[a, b]:.2f}) — "
                            "retrieval may pick either; differentiate the descriptions"))
    return LintReport(issues)


@dataclass
class SelfTestFailure:
    path: str      # the tool that should have been found
    query: str     # the phrasing used (its own description or example)
    rank: int      # where it actually landed (0 = not in top_k at all)
    beaten_by: str # who took the top spot


def selftest(sift, *, top_k: int = 5) -> list[SelfTestFailure]:
    """Retrieval self-test: search each tool's own description (and each of its
    ``examples``) and require the tool itself at rank 1. Failures list who beat
    it — usually a near-duplicate or an over-general description."""
    failures: list[SelfTestFailure] = []
    for t in sift.registry.tools():
        queries = [t.description] + list(t.examples)
        for q in queries:
            if not q.strip():
                continue
            hits = [r.path for r in sift.search_tools(q, top_k=top_k * 3)
                    if r.kind == "function"][:top_k]
            if not hits or hits[0] != t.path:
                rank = hits.index(t.path) + 1 if t.path in hits else 0
                failures.append(SelfTestFailure(t.path, q, rank, hits[0] if hits else "<nothing>"))
    return failures


class GapTracker:
    """An observer that accumulates discovery misses and execution counts.

    Attach it at construction — ``Sift(observer=GapTracker())`` — or compose it
    inside your own observer. Thread-safe enough for CPython dict/Counter ops.
    """

    def __init__(self) -> None:
        self.misses: Counter = Counter()      # query text -> times nothing matched
        self.executions: Counter = Counter()  # tool path -> successful runs

    def __call__(self, event: str, data: dict) -> None:
        if event == "search" and data.get("hits") == 0:
            text = data.get("q") or " ".join(
                x for x in (data.get("domain"), data.get("action")) if x)
            if text:
                self.misses[text] += 1
        elif event == "execute" and data.get("ok") and data.get("path"):
            self.executions[data["path"]] += 1

    def gaps(self, top: int = 20) -> list[tuple[str, int]]:
        """User needs that matched NO tool — your catalogue's blind spots."""
        return self.misses.most_common(top)

    def suggest_pins(self, top: int = 5, min_count: int = 3) -> list[tuple[str, int]]:
        """Most-executed tools — candidates for :meth:`~sift.Sift.pin` (skip the
        search round-trip for what's asked constantly)."""
        return [(p, n) for p, n in self.executions.most_common(top) if n >= min_count]
