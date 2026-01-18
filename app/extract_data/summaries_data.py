from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from unstructured.documents.elements import Element

from db.db import ChromaWork
from extract_data.get_doc_chunks import chunks as doc_chunks
from extract_data.sep_elems import tables_and_texts, get_images


def summaries_text_and_tables_data():
    prompt_text = """You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.
    
    Respond only with the summary, no additional comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.
    
    Table or text chunk: {element}"""

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = OllamaLLM(model='llama3.2:3b', temperature=0.5)
    summarize_chain = {'element': lambda x: x} | prompt | model | StrOutputParser()

    return summarize_chain


def summaries_images(image_base64: str) -> str:
    prompt = """Describe the image in detail. For context,
                  the image is part of a research paper explaining the transformers
                  architecture. Be specific about graphs, such as bar plots.
                  
                  Respond only with the summary, no additional comment.
                  Do not start your message by saying "Here is a summary" or anything like that.
                  Just give the summary as it is."""

    model = OllamaLLM(model='llava:7b', temperature=0.5)
    result = model.invoke(
        prompt,
        images=[image_base64]
    )

    return result


if __name__ == '__main__':
    chunks_: list[Element] = doc_chunks
    tables, texts = tables_and_texts(chunks_)
    images = get_images(chunks_)

    print('Старт обработки текста')
    summarize_chain_text_tables = summaries_text_and_tables_data()
    text_summaries = summarize_chain_text_tables.batch(texts, {'max_concurrency': 3})
    # print(text_summaries)
    # print('-'*10)

    print('Старт обработки таблиц')
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain_text_tables.batch(tables_html, {'max_concurrency': 3})
    # print(table_summaries)

    print('Старт обработки изображений')
    image_summaries: list[str] = [summaries_images(img) for img in images]
    # print(image_summaries)

    # saving data to db
    print('Старт работы с БД')
    chroma_work = ChromaWork()
    retriever = chroma_work.init_db()

    print('Добавление данных в БД')
    chroma_work.add_text(texts, text_summaries)
    chroma_work.add_tables(tables, table_summaries)
    chroma_work.add_images(images, image_summaries)

    print('Вопрос-ответ...')
    chunks_new = retriever.invoke('what is multihead attention?')
    for num, chunk in enumerate(chunks_new):
        print(type(chunk))
        if 'CompositeElement' in str(type(chunk)):
            print('\n\nChunk', num)
            for doc in chunk.metadata.orig_elements:
                print(doc.to_dict()['type'], doc.metadata.page_number)






