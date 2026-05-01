import asyncio
import logging
import pickle
import re
from typing import AsyncGenerator

from domain.entities.models import Answer, Source
from unstructured.staging.base import elements_from_dicts

from .interfaces import (
    IQuestionService,
    IRetrievalService,
    ILLMService,
    IImageService,
)

logger = logging.getLogger(__name__)

IMG_PATTERN = re.compile(r"\[\[IMG:([^\]]+)\]\]")


class QuestionService(IQuestionService):
    def __init__(
        self,
        retrieval_service: IRetrievalService,
        llm_service: ILLMService,
        image_service: IImageService,
    ):
        self._retrieval_service = retrieval_service
        self._llm_service = llm_service
        self._image_service = image_service

    async def get_answer(self, question: str) -> Answer:
        logger.info('Processing question: %s', question)

        context, source_data, img_uids = await self._retrieval_service.retrieve_context(question)
        logger.debug('Context:\n%s', context)

        llm_raw_answer: str = await self._llm_service.get_answer(context, question)
        llm_answer: str = await self._image_service.process_images(llm_raw_answer, img_uids)

        sources = [
            Source(document_name=doc_name, page=page)
            for doc_name, pages in source_data.items()
            for page in pages
        ]

        return Answer(text=llm_answer, sources=sources)

    async def _fetch_image_src(self, img_id: str) -> str | None:
        img_b64_raw = await self._retrieval_service.chroma_work.get_content_from_storage(img_id)
        if img_b64_raw is None:
            return None

        element_dict = pickle.loads(img_b64_raw)
        element_list = await asyncio.to_thread(elements_from_dicts, [element_dict])
        return element_list[0].metadata.image_base64

    async def stream_answer(self, question: str) -> AsyncGenerator[dict, None]:
        logger.info('Processing streaming question: %s', question)

        context, source_data, img_uids = await self._retrieval_service.retrieve_context(question)
        logger.debug('Context retrieved, starting stream...')

        sources = [
            {'document_name': doc_name, 'page': page}
            for doc_name, pages in source_data.items()
            for page in pages
        ]
        yield {'type': 'sources', 'data': sources}

        text_buffer = ''
        async for token in self._llm_service.stream_answer(context, question):
            text_buffer += token

            while ']]' in text_buffer:
                end_idx = text_buffer.index(']]')
                segment = text_buffer[: end_idx + 2]
                text_buffer = text_buffer[end_idx + 2 :]

                img_match = IMG_PATTERN.search(segment)
                if img_match:
                    text_before = segment[: img_match.start()]
                    img_id = img_match.group(1)

                    if text_before:
                        yield {
                            'type': 'block',
                            'data': {'type': 'text', 'content': text_before},
                        }

                    img_src = await self._fetch_image_src(img_id)
                    yield {
                        'type': 'block',
                        'data': {'type': 'image', 'id': img_id, 'src': img_src},
                    }
                else:
                    yield {
                        'type': 'block',
                        'data': {'type': 'text', 'content': segment},
                    }

        if text_buffer:
            yield {'type': 'block', 'data': {'type': 'text', 'content': text_buffer}}

        yield {'type': 'done', 'data': None}
        logger.info('Streaming complete')
