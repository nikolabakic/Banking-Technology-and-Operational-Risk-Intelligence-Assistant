"""OpenAI-compatible client construction for BankScope."""

from bankscope.llm.client import create_langchain_chat_model, create_openai_client

__all__ = ["create_langchain_chat_model", "create_openai_client"]
