from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract base class for all LLM adapters.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
