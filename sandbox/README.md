# Sandbox

This directory preserves development artifacts that are no longer part of the
active BankScope pipeline. They remain in the repository so earlier approaches,
experiments, and project evolution can be reviewed with a mentor.

Files under `sandbox/` are reference material only. They may contain outdated
paths, dependencies, assumptions, or interfaces and should not be used by the
current pipeline without review.

## Contents

- `app.py`: empty application placeholder from the initial project structure.
- `docs/`: the original long-form roadmap, retained as historical planning
  context.
- `notebooks/`: exploratory parser, chunking, table-proxy, and embedding
  notebooks. Reusable checks were moved into active tests or scripts.
- `scripts/`: superseded chunking, LLM table-proxy, and setup smoke-test
  scripts.
- `tests/`: former setup and embedding smoke checks that were replaced by
  smaller responsibility-based tests.

The active project workflow is documented in `README.md`, `docs/roadmap.md`,
and `docs/data_pipeline.md`.
