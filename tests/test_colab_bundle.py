import json
import zipfile
from pathlib import Path

from scripts.build_colab_bundle import build_bundle


def test_build_bundle_contains_required_inputs_and_hash_manifest(tmp_path: Path) -> None:
    root = tmp_path / "project"
    required = (
        "README.md",
        "pyproject.toml",
        "scripts/embed.py",
        "scripts/evaluate.py",
        "data/processed/chunks.jsonl",
        "data/processed/tables.jsonl",
        "data/processed/lexical_glossary_locators_v1.jsonl",
        "data/processed/manifest.json",
        "data/evaluation/queries.jsonl",
        "docs/decisions/002-repository-overhaul.md",
        "docs/decisions/009-complete-primary-filings.md",
        "src/bankscope/__init__.py",
    )
    for relative in required:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    output = tmp_path / "bundle.zip"
    result = build_bundle(root, output)

    assert result["file_count"] == len(required)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert set(required).issubset(names)
        assert "bundle_manifest.json" in names
        manifest = json.loads(archive.read("bundle_manifest.json"))
    assert manifest["format_version"] == 1
    assert {entry["path"] for entry in manifest["files"]} == set(required)
