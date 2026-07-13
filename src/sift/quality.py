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
- :func:`suggest_min_score` — calibrate the relevance floor, so discovery can say
  "nothing fits, answer directly" instead of always returning its best guess.

    report = quality.lint(sift)
    print(report.format())

    failures = quality.selftest(sift)

    print(quality.suggest_min_score(sift, negatives=["what's the weather on Mars"]).format())

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
class FloorSuggestion:
    """A calibrated value for ``Sift(min_score=...)`` — the relevance floor below
    which discovery answers "no tool fits" instead of handing back its best guess."""
    suggested: float
    weakest_positive: float          # the lowest score a query your catalog CAN serve got
    weakest_query: str               # ...and which one — raise the floor past this and it breaks
    strongest_negative: float | None = None   # the best score an out-of-catalog query got
    strongest_negative_query: str | None = None
    separated: bool = True           # False = a negative outscored a positive: no floor works

    def format(self) -> str:
        lines = [f"suggested min_score = {self.suggested:.3f}",
                 f"  weakest in-catalog query scored {self.weakest_positive:.3f} "
                 f"({self.weakest_query!r})"]
        if self.strongest_negative is not None:
            lines.append(f"  strongest out-of-catalog query scored {self.strongest_negative:.3f} "
                         f"({self.strongest_negative_query!r})")
            if not self.separated:
                lines.append("  ⚠ NOT SEPARABLE: an out-of-catalog query outscored a real one. "
                             "No floor can tell them apart — fix the descriptions first "
                             "(run lint/selftest); the suggestion below only protects recall.")
        else:
            lines.append("  no negatives given: this is a CEILING, not a calibration — the "
                         "highest floor that still admits every tool. Pass negatives= "
                         "(things your catalog cannot do) for a real recommendation.")
        return "\n".join(lines)


def suggest_min_score(sift, *, negatives: list[str] | None = None,
                      margin: float = 0.9) -> FloorSuggestion:
    """Calibrate the relevance floor from the catalogue instead of guessing it.

    ``min_score`` defaults to 0.0, which means discovery ALWAYS returns its top-k —
    however bad the match. That is discovery's own hollow success: it teaches the
    model that a tool always exists, so it forces one. A floor is what lets SIFT
    answer "nothing here fits, answer directly" — but only if the number is right,
    and nobody can guess it for someone else's catalogue.

    Scores are on exactly the scale the floor uses (max embedding cosine over the
    index — see ``Gateway._passes_floor``). Positives are every tool's own
    description and examples. ``negatives`` are needs your catalogue genuinely
    cannot serve ("what's the weather on Mars") — supply them and you get a real
    midpoint; omit them and you get the ceiling that preserves recall.
    """
    gw = sift.gateway
    if gw.retrieval == "bm25" or not gw._vectors:
        raise ValueError("min_score calibration needs embeddings "
                         "(retrieval='hybrid' or 'embedding') and a built index")
    from .embeddings import cosine

    def score(q: str) -> float:
        qv = gw._embed_query(q)
        return max(cosine(qv, v) for v in gw._vectors)

    positives = [(score(q), q) for t in sift.registry.tools()
                 for q in ([t.description] + list(t.examples)) if q.strip()]
    if not positives:
        raise ValueError("no tool descriptions to calibrate against")
    weakest, weakest_q = min(positives)

    if not negatives:
        # No negatives: the most we can honestly say is "go above this and you start
        # rejecting queries your catalogue CAN serve".
        return FloorSuggestion(round(weakest * margin, 3), round(weakest, 3), weakest_q)

    strongest, strongest_q = max((score(q), q) for q in negatives if q.strip())
    separated = strongest < weakest
    suggested = (weakest + strongest) / 2 if separated else weakest * margin
    return FloorSuggestion(round(suggested, 3), round(weakest, 3), weakest_q,
                           round(strongest, 3), strongest_q, separated)


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
