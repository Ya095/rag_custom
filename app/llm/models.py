from typing import List, Any
from langchain_ollama import OllamaLLM as OllamaClient
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input

from core.config import TEXT_MODEL, IMAGE_DESCRIPTION_MODEL


class OllamaLLMWrapper(Runnable):
    _instance = None

    def __new__(
        cls,
        model: str = TEXT_MODEL,
        temperature: float = 0.1,
        num_predict: int | None = None,
    ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = OllamaClient(
                model=model, temperature=temperature, num_predict=num_predict
            )
        return cls._instance

    def invoke(
        self,
        input_str: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ):
        return self._instance.model.invoke(input_str, config, **kwargs)

    async def ainvoke(
        self,
        input_str: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ):
        return await self._instance.model.ainvoke(input_str, config, **kwargs)


class LLaVAWrapper(Runnable):
    _instance = None

    def __new__(
        cls,
        model: str = IMAGE_DESCRIPTION_MODEL,
        temperature: float = 0.4,
        num_predict: int | None = None,
    ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = OllamaClient(
                model=model, temperature=temperature, num_predict=num_predict
            )
        return cls._instance

    def invoke(
        self,
        input_str: str,
        images: List[str] = None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str:
        return self._instance.model.invoke(input_str, images=images or [])

    async def ainvoke(
        self,
        input_str: str,
        images: List[str] = None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str:
        return await self._instance.model.ainvoke(input_str, images=images or [])


text_model = OllamaLLMWrapper()
image_model = LLaVAWrapper()
