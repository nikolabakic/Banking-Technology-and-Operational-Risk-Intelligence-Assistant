"""Grounded answer generation from retrieved BankScope evidence."""

from bankscope.generation.answer_generator import generate_answer
from bankscope.generation.pipeline import AnswerRun, SingleBankAnswerPipeline

__all__ = ["AnswerRun", "SingleBankAnswerPipeline", "generate_answer"]
