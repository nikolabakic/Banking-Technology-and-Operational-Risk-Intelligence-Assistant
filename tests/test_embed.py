from types import SimpleNamespace

import pytest

from scripts.embed import validate_input_lengths


class WordTokenizer:
    def encode(self, text: str, **_: object) -> list[str]:
        return ["<special>", *text.split()]


def test_validate_input_lengths_accepts_input_at_limit() -> None:
    model = SimpleNamespace(tokenizer=WordTokenizer())

    validate_input_lengths(model, ["one two three"], max_seq_length=4)


def test_validate_input_lengths_rejects_truncation() -> None:
    model = SimpleNamespace(tokenizer=WordTokenizer())

    with pytest.raises(ValueError, match="5 tokens; maximum is 4"):
        validate_input_lengths(model, ["one two three four"], max_seq_length=4)
