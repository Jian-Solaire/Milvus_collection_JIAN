"""LLM client for generating responses."""
from core.llm_factory import LLMClient


class LLMGenerator:
    """Generate response using LLM."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate(self, prompt: str) -> str:
        """Generate response from prompt."""
        pass
