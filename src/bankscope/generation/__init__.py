"""Grounded answer generation from retrieved BankScope evidence."""

from bankscope.generation.answer_generator import (
    GPT51_CANDIDATE_MODEL,
    GenerationValidationError,
    NumericFacts,
    generate_answer,
)
from bankscope.generation.pipeline import AnswerRun, SingleBankAnswerPipeline

__all__ = [
    "AnswerRun",
    "GPT51_CANDIDATE_MODEL",
    "GenerationValidationError",
    "NumericFacts",
    "SingleBankAnswerPipeline",
    "generate_answer",
]
