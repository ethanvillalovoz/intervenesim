# Contributing

Thanks for improving InterveneSim. Please keep changes reproducible, local-first, and
explicit about what a result does and does not establish.

## Setup

```bash
uv sync --locked --extra dev
uv run intervenesim doctor
```

Before opening a pull request, run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run intervenesim smoke
```

Changes to the benchmark protocol should update `docs/benchmark.md`. Changes to reported
numbers must include the resolved configuration, per-episode CSV, aggregate CSV, and system
manifest needed to audit the claim. Do not commit datasets or model checkpoints unless a
release specifically requires them.
