# Contributing to RagWarden

Thanks for your interest. RagWarden is built in sequenced phases; each phase produces a working,
testable increment with explicit acceptance criteria.

## Development setup

```bash
git clone https://github.com/utsavopal/ragwarden
cd ragwarden
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pip install --group dev            # pip >= 25.1 (PEP 735); or: pip install -r requirements-dev.txt
```

## Before opening a PR

```bash
ruff check .
ruff format --check .
mypy
pytest                 # offline; real-model tests are deselected by default
pytest -m model        # opt-in: downloads real ML models
```

## Ground rules

- **Do not change `ragwarden/contracts.py`** without a design discussion — those are frozen contracts.
- Detectors and adapters must **lazy-import** their heavy dependency and raise `MissingExtraError`
  when it's absent. Importing `ragwarden` must never pull `torch` / `transformers`.
- Where the build spec marks an **OPEN DECISION**, implement it behind a config flag; do not pick
  one silently.
- New behavior needs a test. New public API needs a docstring.
- Update `CHANGELOG.md` under `[Unreleased]`.

## Verification harness

`tests/` is split into `unit/`, `integration/`, and `benchmark/`. Benchmark results are checked in
under `benchmarks/results/` and must reflect the actual output of the code, never aspirational
numbers.
