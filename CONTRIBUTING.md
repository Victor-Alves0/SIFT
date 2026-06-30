# Contributing to SIFT

Thanks for your interest! SIFT is a small, dependency-light Python package.

## Dev setup

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Run checks

```bash
ruff check src tests examples benchmarks
pytest -q
```

Tests are offline and deterministic (a fake embedder / `retrieval="bm25"`), so they
need no model download or API keys. Live smoke tests under `examples/` (`smoke_*.py`)
do need an OpenRouter key in `.env`.

## Guidelines

- Keep the core dependency-light: heavy/optional integrations go behind extras
  (`[openai]`, `[anthropic]`, `[langchain]`, `[mcp]`, `[server]`).
- New behavior needs a test. Keep `ruff` clean.
- The public surface is the `Sift` facade and the adapters; prefer adding to those.

## Releasing

1. Bump `version` in `pyproject.toml` and update `CHANGELOG.md`.
2. Tag: `git tag vX.Y.Z && git push --tags`.
3. The `Publish` workflow builds and publishes to PyPI (Trusted Publishing).
