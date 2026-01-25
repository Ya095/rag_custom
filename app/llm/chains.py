from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.base import Other
from unstructured.documents.elements import Element

from llm.models import text_model, image_model
from llm.preprocess import element_to_prompt_text
from llm.prompts import TEXT_SUMMARY_PROMPT, TABLE_SUMMARY_PROMPT, IMAGE_SUMMARY_PROMPT


def summaries_text_data() -> RunnableSerializable[Other, Other] | RunnableSerializable[Other, str]:
    """Receive summary for text or table."""

    prompt = ChatPromptTemplate.from_template(TEXT_SUMMARY_PROMPT)
    model = text_model
    summarize_chain = {'element': element_to_prompt_text} | prompt | model | StrOutputParser()

    return summarize_chain


def summaries_table_data() -> RunnableSerializable[Other, Other] | RunnableSerializable[Other, str]:
    """Receive summary for text or table."""

    prompt = ChatPromptTemplate.from_template(TABLE_SUMMARY_PROMPT)
    model = text_model
    summarize_chain = {'element': element_to_prompt_text} | prompt | model | StrOutputParser()

    return summarize_chain


def summaries_images(image_el: Element) -> str:
    """Receive summary for image."""

    image_base64: str = image_el.metadata.image_base64

    model = image_model
    result: str = model.invoke(
        IMAGE_SUMMARY_PROMPT,
        images=[image_base64]
    )

    return result
