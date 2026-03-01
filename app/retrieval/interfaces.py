from abc import ABC, abstractmethod


class IProcesUserQuestion(ABC):
    """An interface for answering on user questions."""

    @abstractmethod
    async def get_answer(self, question: str) -> dict:
        pass
