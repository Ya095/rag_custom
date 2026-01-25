import pickle

from langchain_classic.retrievers import MultiVectorRetriever
from unstructured.documents.elements import Element
from unstructured.staging.base import elements_from_dicts

from extract_data.summaries_data import print_element
from ingestion.extract_elements import extract_tables_texts_images
from ingestion.parser import parse_input_document
from llm.chains import summaries_text_data, summaries_table_data, summaries_images
from llm.preprocess import table_to_prompt_text
from repository.storage import ChromaWork


def run():
    # print('Разбивка документа')
    # chunks_: list[Element] = parse_input_document()
    # tables, texts, images = extract_tables_texts_images(chunks_)
    #
    # print('Старт обработки текста')
    # summarize_chain_text = summaries_text_data()
    # text_summaries = summarize_chain_text.batch(texts, {'max_concurrency': 3})
    #
    # print('Старт обработки таблиц')
    # summarize_chain_table = summaries_table_data()
    # table_inputs = [table_to_prompt_text(t) for t in tables]
    # table_summaries = summarize_chain_table.batch(table_inputs, {'max_concurrency': 3})
    #
    # print('Старт обработки изображений')
    # image_summaries: list[str] = [summaries_images(img) for img in images]

    # work with db
    chroma_work = ChromaWork()
    retriever: MultiVectorRetriever = chroma_work.init_db()

    # print('Добавление данных в БД')
    # chroma_work.add_element(texts, text_summaries)
    # chroma_work.add_element(tables, table_summaries)
    # chroma_work.add_element(images, image_summaries)

    print('Вопрос-ответ...')
    chunks_new: list[bytes] = retriever.invoke('How we train the models?')

    for num, raw in enumerate(chunks_new):
        element_dict: dict = pickle.loads(raw)
        element: Element = elements_from_dicts([element_dict])[0]

        print_element(element, num)


if __name__ == '__main__':
    run()
