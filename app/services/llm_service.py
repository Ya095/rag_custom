import logging

from llm.chains import rag_answer_chain
from .interfaces import ILLMService

logger = logging.getLogger(__name__)


class LLMService(ILLMService):
    async def get_answer(self, context: str, question: str) -> str:
        logger.info("Waiting answer from model...")

        chain = rag_answer_chain()
        answer: str = await chain.ainvoke({"context": context, "question": question})

        logger.debug(f"LLM raw answer: {answer}")

        return answer
