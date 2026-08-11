"""Run one synthetic JSON-mode generation compatibility probe with no filing data."""

from __future__ import annotations

import argparse
import json

from bankscope.config.settings import get_settings
from bankscope.generation import GPT51_CANDIDATE_MODEL, generate_answer
from bankscope.llm import create_openai_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help=f"Model deployment to probe; the current candidate is {GPT51_CANDIDATE_MODEL}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    result = generate_answer(
        "What was Example Bank's synthetic test ratio on December 31, 2025?",
        [
            {
                "target_chunk_id": "synthetic-evidence-1",
                "record_type": "text",
                "ticker": "TEST",
                "evidence": (
                    "Synthetic compatibility evidence: Example Bank's test ratio was 12.34% "
                    "on December 31, 2025."
                ),
                "metadata": {"report_date": "2025-12-31"},
            }
        ],
        client=create_openai_client(settings),
        model=args.model,
        expected_ticker="TEST",
        expected_bank_name="Example Bank",
    )
    if result["status"] != "supported" or result["answer_type"] != "numeric":
        raise RuntimeError("The compatibility probe did not produce a supported numeric answer.")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
