# Python source layout

**Status:** active application source.

The project uses the packaging `src` layout configured in `pyproject.toml`. The importable package
is `bankscope`, while user-facing commands remain in the repository-level `scripts/` folder.

```text
src/
└── bankscope/
    ├── api.py
    ├── io.py
    ├── chat/
    ├── config/
    ├── evaluation/
    ├── generation/
    ├── llm/
    ├── parsing/
    ├── retrieval/
    └── sec/
```

```mermaid
flowchart LR
    Scripts[CLI scripts] --> Package[bankscope package]
    API[FastAPI] --> Package
    Tests[tests] --> Package
    Package --> Data[validated local artifacts]
```

Keeping application code below `src` prevents accidental imports from the repository root and
makes tests exercise the same installed package used by scripts. Install it in editable mode with
`python -m pip install -e ".[dev,llm]"`.

Do not place generated files, CLI-only orchestration, or archived experiments here. See the
[package architecture](bankscope/README.md) and [script guide](../scripts/README.md).

[Back to repository guide](../README.md)

