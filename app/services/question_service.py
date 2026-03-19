import logging

from domain.entities.models import Answer, Source
from .interfaces import (
    IQuestionService,
    IRetrievalService,
    ILLMService,
    IImageService,
)

logger = logging.getLogger(__name__)


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
        logger.info(f"Processing question: {question}")

        context, source_data, img_uids = await self._retrieval_service.retrieve_context(
            question
        )

        logger.debug(f"Context:\n{context}")

        llm_raw_answer = await self._llm_service.get_answer(context, question)

        llm_answer = await self._image_service.process_images(llm_raw_answer, img_uids)

        sources = [
            Source(document_name=doc_name, page=page)
            for doc_name, pages in source_data.items()
            for page in pages
        ]

        return Answer(text=llm_answer, sources=sources)
