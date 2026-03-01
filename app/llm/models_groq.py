import asyncio
import time
from functools import lru_cache

from langchain_core.runnables import RunnableSerializable, RunnableConfig
from langchain_core.runnables.utils import Input
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import PrivateAttr

from config import config as app_config
from llm.prompts import IMAGE_SUMMARY_PROMPT


@lru_cache
def get_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=app_config.llm_config.api_token,
        base_url=app_config.llm_config.base_url,
        max_retries=app_config.llm_config.max_retries,
    )


class GroqTextRunnable(RunnableSerializable):
    """Class for working with text models."""
    model: str
    temperature: float = 0

    _client: AsyncOpenAI = PrivateAttr()
    _last_call: float = PrivateAttr(default=0)
    _lock: asyncio.Lock = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = get_async_client()
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    def invoke(self, *args, **kwargs):
        raise TypeError('Use ainvoke')

    async def ainvoke(
        self,
        input_data: Input,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str:

        # -------- LOCAL RPM LIMIT --------
        rpm = None
        if config and 'rpm' in config:
            rpm = config['rpm']
            if rpm < 0 or not isinstance(rpm, int):
                raise ValueError('RPM must be positive integer!')

        if rpm is not None:
            min_interval: float = 60 / (rpm + 1)

            async with self._lock:
                now: float = time.monotonic()
                elapsed: float = now - self._last_call

                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)

                self._last_call = time.monotonic()

        messages: list[dict] = self._build_messages(input_data)
        response: ChatCompletion = await self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )

        return response.choices[0].message.content

    def _build_messages(self, input_data: Input) -> list[dict[str, str]]:
        """Building msg for llm model with correct format."""

        if isinstance(input_data, list):
            return input_data
        return [{'role': 'user', 'content': str(input_data)}]


class GroqVisionRunnable(RunnableSerializable):
    """Class for working with image models."""
    model: str
    temperature: float = 0
    _client: AsyncOpenAI = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client: AsyncOpenAI = get_async_client()

    def invoke(self, input_data: Input, config: RunnableConfig | None = None, **kwargs):
        raise TypeError('Please use ainvoke method.')

    async def ainvoke(self, input_data: Input, config: RunnableConfig | None = None, **kwargs) -> str:
        """input should be base64 formatted."""

        image_b64: str = f'data:image/jpeg;base64,{input_data}'

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': IMAGE_SUMMARY_PROMPT},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': image_b64
                        },
                    },
                ],
            }
        ]

        response = await self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )

        return response.choices[0].message.content


text_model = GroqTextRunnable(model=app_config.llm_config.text_model, temperature=0.4)
answer_model = GroqTextRunnable(model=app_config.llm_config.text_model, temperature=0)
image_model = GroqVisionRunnable(model=app_config.llm_config.image_model, temperature=0.5)
