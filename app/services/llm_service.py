import logging
from typing import AsyncGenerator

from llm.chains import rag_answer_chain
from llm.models_groq import answer_model
from llm.prompts import RAG_ANSWER_PROMPT
from .interfaces import ILLMService


logger = logging.getLogger(__name__)


class LLMService(ILLMService):
    async def get_answer(self, context: str, question: str) -> str:
        logger.info('Waiting answer from model...')

        chain = rag_answer_chain()
        answer: str = await chain.ainvoke({'context': context, 'question': question})

        logger.debug('LLM raw answer: %s', answer)

        return answer

    async def stream_answer(self, context: str, question: str) -> AsyncGenerator[str, None]:
        logger.info('Starting streaming answer from model...')

        prompt = RAG_ANSWER_PROMPT.format(context=context, question=question)
        messages = [{'role': 'user', 'content': prompt}]

        stream_generator = await answer_model.ainvoke(messages, stream=True)
        async for token in stream_generator:
            yield token

        logger.debug('Streaming complete')
