"""Build the minimal, hash-audited ZIP consumed by the BankScope Colab GPU notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/bankscope_colab_gpu_bundle.zip"

REQUIRED_FILES = (
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
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def collect_bundle_files(root: Path) -> list[Path]:
    paths = [root / relative for relative in REQUIRED_FILES]
    paths.extend(sorted((root / "src/bankscope").rglob("*.py")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Colab bundle inputs are missing: {missing}")
    return paths


def build_bundle(root: Path, output: Path) -> dict[str, object]:
    files = collect_bundle_files(root)
    entries: list[dict[str, object]] = []
    payloads: list[tuple[str, bytes]] = []
    for path in files:
        archive_name = path.relative_to(root).as_posix()
        content = path.read_bytes()
        payloads.append((archive_name, content))
        entries.append(
            {
                "path": archive_name,
                "size": len(content),
                "sha256": sha256_bytes(content),
            }
        )

    bundle_manifest = {
        "format_version": 1,
        "purpose": "BankScope Colab GPU embedding and baseline retrieval evaluation",
        "files": entries,
    }
    manifest_content = (json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for archive_name, content in payloads:
                archive.writestr(archive_name, content)
            archive.writestr("bundle_manifest.json", manifest_content)
        os.replace(temporary_path, output)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return {
        "path": str(output.resolve()),
        "file_count": len(entries),
        "size": output.stat().st_size,
        "sha256": sha256_bytes(output.read_bytes()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Bundle already exists: {args.output}. Use --overwrite.")
    result = build_bundle(ROOT, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
