# Catalog quality toolkit

"Write clear, descriptive tools" is standard guidance — `sift.quality` makes it
**executable**. Three instruments, all offline, all using the index and
telemetry you already have.

## `lint()` — static catalogue checks

```python
from sift import quality
report = quality.lint(sift)      # after build_index()
print(report.format())
```

Flags, per tool: missing description (**error** — undiscoverable), very short or
very long descriptions (retrieval has nothing to match on / TOON bloat),
params without descriptions (the model fills them by guessing), single-tool
categories (wasted hierarchy level), and — using the built vectors —
**near-duplicate tools** (cosine ≥ `dup_threshold`, default 0.92): two tools the
retriever can't reliably tell apart, so it may pick either.

`report.errors` / `report.warnings` give you programmatic access — fail CI on
errors if you want a quality gate.

## `selftest()` — can each tool be found with its own phrasing?

```python
for f in quality.selftest(sift):
    print(f"{f.path}: '{f.query}' ranked {f.rank or 'not in top-k'}, "
          f"beaten by {f.beaten_by}")
```

Searches every tool's **own description** (and each of its `examples=`) and
requires the tool itself at rank 1. A tool that can't be found with its own
words is a tool the model will never find — the failure names who outranked it
(usually a near-duplicate or an over-general description).

## `GapTracker` — telemetry → decisions

```python
tracker = quality.GapTracker()
sift = Sift(observer=tracker)
# ... production traffic ...

tracker.gaps()          # [(query, times), ...] — user needs that matched NOTHING
tracker.suggest_pins()  # [(path, executions), ...] — hot tools worth pin()ning
```

An observer that accumulates discovery **misses** (searches that returned no
tool — your catalogue's blind spots) and **execution counts** (candidates for
[`pin`](discovery.md#pinning-hot-tools-skip-discovery-entirely), which removes
the search round-trip for what's asked constantly). Compose it inside your own
observer if you already have one — it's just a callable.

## The loop

1. Before shipping: `lint()` clean, `selftest()` green.
2. In production: `GapTracker` attached.
3. Periodically: fill `gaps()` with new tools, `pin()` what `suggest_pins()`
   surfaces, re-run 1.
