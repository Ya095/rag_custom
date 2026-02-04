from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.base import Other
from unstructured.documents.elements import Element

from llm.models import text_model, image_model, answer_model
from llm.preprocess import element_to_prompt_text
from llm.prompts import *


def summaries_text_data() -> RunnableSerializable[Other, Other] | RunnableSerializable[Other, str]:
    """Receive summary for text or table."""

    prompt = ChatPromptTemplate.from_template(TEXT_SUMMARY_PROMPT)
    summarize_chain = {'element': element_to_prompt_text} | prompt | text_model | StrOutputParser()

    return summarize_chain


def summaries_table_data() -> RunnableSerializable[Other, Other] | RunnableSerializable[Other, str]:
    """Receive summary for text or table."""

    prompt = ChatPromptTemplate.from_template(TABLE_SUMMARY_PROMPT)
    summarize_chain = {'element': element_to_prompt_text} | prompt | text_model | StrOutputParser()

    return summarize_chain


async def summaries_images(image_el: Element) -> str:
    """Receive summary for image."""

    image_base64: str = image_el.metadata.image_base64

    model = image_model
    result: str = await model.ainvoke(
        IMAGE_SUMMARY_PROMPT,
        images=[image_base64]
    )

    return result


def rag_answer_chain() -> RunnableSerializable[Other, Other] | RunnableSerializable[Other, str]:
    """Receive the answer from the llm based on the context."""

    prompt = ChatPromptTemplate.from_template(RAG_ANSWER_PROMPT)
    return prompt | answer_model | StrOutputParser()
