from abc import ABC, abstractmethod
from typing import AsyncGenerator

from domain.entities.models import Answer


class IRetrievalService(ABC):
    @abstractmethod
    async def retrieve_context(self, question: str) -> tuple[str, dict, list[str]]:
        pass


class ILLMService(ABC):
    @abstractmethod
    async def get_answer(self, context: str, question: str) -> str:
        pass

    @abstractmethod
    async def stream_answer(self, context: str, question: str) -> AsyncGenerator[str, None]:
        pass


class IImageService(ABC):
    @abstractmethod
    async def process_images(self, answer: str, img_uids: list[str]) -> str:
        pass


class IQuestionService(ABC):
    @abstractmethod
    async def get_answer(self, question: str) -> Answer:
        pass

    @abstractmethod
    async def stream_answer(self, question: str) -> AsyncGenerator[dict, None]:
        pass
