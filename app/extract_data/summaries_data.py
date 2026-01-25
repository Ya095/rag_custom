import pickle

from langchain_ollama import OllamaLLM
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from unstructured.documents.elements import Element, CompositeElement, Table, Image
from unstructured.staging.base import elements_from_dicts

from llm.prompts import TEXT_SUMMARY_PROMPT, TABLE_SUMMARY_PROMPT, IMAGE_SUMMARY_PROMPT
from repository.storage import ChromaWork


# def element_to_prompt_text(elem: Element) -> str:
#     """Convert Element into clean text for LLM summarization."""
#
#     # CompositeElement
#     if hasattr(elem, "metadata") and getattr(elem.metadata, "orig_elements", None):
#         parts = []
#         for sub_elem in elem.metadata.orig_elements:
#             if hasattr(sub_elem, "text") and sub_elem.text:
#                 parts.append(sub_elem.text)
#         return "\n".join(parts)
#
#     # Plain text element
#     if hasattr(elem, "text") and elem.text:
#         return elem.text
#
#     return str(elem)
#
#
# def table_to_prompt_text(table: Element) -> str:
#     """Get table html format for llm summaries."""
#
#     html: str | None = getattr(table.metadata, 'text_as_html', None)
#     if not html:
#         return str(table)
#
#     return f'HTML table:\n{html}'
#
#
# def summaries_text_data() -> str:
#     """Receive summary for text or table."""
#
#     prompt = ChatPromptTemplate.from_template(TEXT_SUMMARY_PROMPT)
#     model = OllamaLLM(model='llama3.2:3b', temperature=0.4, num_predict=120)
#     summarize_chain = {'element': element_to_prompt_text} | prompt | model | StrOutputParser()
#
#     return summarize_chain
#
#
# def summaries_table_data() -> str:
#     """Receive summary for text or table."""
#
#     prompt = ChatPromptTemplate.from_template(TABLE_SUMMARY_PROMPT)
#     model = OllamaLLM(model='llama3.2:3b', temperature=0.4, num_predict=120)
#     summarize_chain = {'element': element_to_prompt_text} | prompt | model | StrOutputParser()
#
#     return summarize_chain
#
#
# def summaries_images(image_el: Element) -> str:
#     """Receive summary for image."""
#
#     image_base64: str = image_el.metadata.image_base64
#
#     model = OllamaLLM(model='llava:7b', temperature=0.4, num_predict=150)
#     result: str = model.invoke(
#         IMAGE_SUMMARY_PROMPT,
#         images=[image_base64]
#     )
#
#     return result


def print_element(el: Element, idx: int) -> None:
    print(f"\nChunk {idx}")
    print("Type:", type(el))
    print("Category:", el.category)
    print("Page:", el.metadata.page_number)

    # Composite text block
    if isinstance(el, CompositeElement):
        print("Composite elements:")
        for sub in el.metadata.orig_elements or []:
            print(" -", sub.category, sub.metadata.page_number)
            if hasattr(sub, "text") and sub.text:
                print(sub.text)
            elif hasattr(sub.metadata, "text_as_html"):
                print(sub.metadata.text_as_html)

    # Table
    elif isinstance(el, Table):
        print("Table:")
        if el.metadata.text_as_html:
            print(el.metadata.text_as_html)
        else:
            print("[Table without HTML representation]")

    # Image
    elif isinstance(el, Image):
        print("Image:")
        print(" - format:", el.metadata.image_mime_type)
        print(" - page:", el.metadata.page_number)

    # Plain text element
    elif hasattr(el, "text") and el.text:
        print("Text:")
        print(el.text[:500], '...')

    else:
        print("[Unknown or empty element]")


# if __name__ == '__main__':
#     # chunks_: list[Element] = doc_chunks
#     # tables, texts, images = extract_tables_texts_images(chunks_)
#     #
#     # print('Старт обработки текста')
#     # summarize_chain_text_tables = summaries_text_and_tables_data()
#     # text_summaries = summarize_chain_text_tables.batch(texts, {'max_concurrency': 3})
#     #
#     # print('Старт обработки таблиц')
#     # table_inputs = [table_to_prompt_text(t) for t in tables]
#     # table_summaries = summarize_chain_text_tables.batch(table_inputs, {'max_concurrency': 3})
#     #
#     # print('Старт обработки изображений')
#     # image_summaries: list[str] = [summaries_images(img) for img in images]
#
#     # work with db
#     chroma_work = ChromaWork()
#     retriever: MultiVectorRetriever = chroma_work.init_db()
#
#     # print('Добавление данных в БД')
#     # chroma_work.add_element(texts, text_summaries)
#     # chroma_work.add_element(tables, table_summaries)
#     # chroma_work.add_element(images, image_summaries)
#
#     print('Вопрос-ответ...')
#     chunks_new: list[bytes] = retriever.invoke('How we train the models?')
#
#     for num, raw in enumerate(chunks_new):
#         element_dict: dict = pickle.loads(raw)
#         element: Element = elements_from_dicts([element_dict])[0]
#
#         print_element(element, num)

