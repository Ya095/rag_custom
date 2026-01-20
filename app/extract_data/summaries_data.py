import pickle

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from unstructured.documents.elements import Element
from unstructured.staging.base import elements_from_base64_gzipped_json, elements_from_dicts, elements_from_json

from db.db import ChromaWork
from extract_data.get_doc_chunks import chunks as doc_chunks
from extract_data.sep_elems import extract_tables_texts_images



def element_to_prompt_text(el: Element) -> str:
    """Convert Element into clean text for LLM summarization."""

    # CompositeElement
    if hasattr(el, "metadata") and getattr(el.metadata, "orig_elements", None):
        parts = []
        for sub_el in el.metadata.orig_elements:
            if hasattr(sub_el, "text") and sub_el.text:
                parts.append(sub_el.text)
        return "\n".join(parts)

    # Plain text element
    if hasattr(el, "text") and el.text:
        return el.text

    return str(el)


def table_to_prompt_text(table: Element) -> str:
    html = getattr(table.metadata, "text_as_html", None)
    if not html:
        return str(table)
    return f"HTML table:\n{html}"


def summaries_text_and_tables_data():
    prompt_text = """You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.
    
    Respond only with the summary, no additional comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.
    
    Table or text chunk: {element}"""

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = OllamaLLM(model='llama3.2:3b', temperature=0.5)
    summarize_chain = {'element': element_to_prompt_text} | prompt | model | StrOutputParser()

    return summarize_chain


def summaries_images(image_el: Element) -> str:
    prompt = """Describe the image in detail.
    If it contains charts, explain axes and trends.
    If it contains diagrams, explain components and relationships.
                  
    Respond only with the summary, no additional comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is."""

    image_base64 = image_el.metadata.image_base64

    model = OllamaLLM(model='llava:7b', temperature=0.5)
    result = model.invoke(
        prompt,
        images=[image_base64]
    )

    return result


if __name__ == '__main__':
    # chunks_: list[Element] = doc_chunks
    # tables, texts, images = extract_tables_texts_images(chunks_)
    #
    # print('Старт обработки текста')
    # summarize_chain_text_tables = summaries_text_and_tables_data()
    # text_summaries = summarize_chain_text_tables.batch(texts, {'max_concurrency': 3})
    #
    # print('Старт обработки таблиц')
    # table_inputs = [table_to_prompt_text(t) for t in tables]
    # table_summaries = summarize_chain_text_tables.batch(table_inputs, {'max_concurrency': 3})
    #
    # print('Старт обработки изображений')
    # image_summaries: list[str] = [summaries_images(img) for img in images]

    # work with db
    print('Старт работы с БД')
    chroma_work = ChromaWork()
    retriever = chroma_work.init_db()

    # print('Добавление данных в БД')
    # chroma_work.add_texts(texts, text_summaries)
    # chroma_work.add_tables(tables, table_summaries)
    # chroma_work.add_images(images, image_summaries)

    print('Вопрос-ответ...')
    chunks_new = retriever.invoke('what is the Transformer model architecture?')

    # for num, chunk in enumerate(chunks_new):
    #     print(type(chunk))
    #     if 'CompositeElement' in str(type(chunk)):
    #         print('\n\nChunk', num)
    #         for doc in chunk.metadata.orig_elements:
    #             print(doc.to_dict()['type'], doc.metadata.page_number)

    for num, raw in enumerate(chunks_new):
        element_dict = pickle.loads(raw)
        element = elements_from_dicts([element_dict])[0]

        print("\nChunk", num)
        print("Type:", type(element))
        print("Category:", element.category)
        print("Page:", element.metadata.page_number)

        if element.metadata.orig_elements:
            for el in element.metadata.orig_elements:
                print(" -", el.category, el.metadata.page_number)
