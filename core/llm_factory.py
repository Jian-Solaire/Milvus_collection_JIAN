"""LLM factory for creating LLM clients."""
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract LLM client."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from prompt."""
        pass
