from types import SimpleNamespace

import pytest

from scripts.embed import resolve_device, validate_input_lengths


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


def test_resolve_device_uses_cuda_for_auto_when_available() -> None:
    assert resolve_device("auto", cuda_available=True) == "cuda"
    assert resolve_device("auto", cuda_available=False) == "cpu"


def test_resolve_device_rejects_unavailable_cuda() -> None:
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda", cuda_available=False)
