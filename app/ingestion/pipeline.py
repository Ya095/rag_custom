import asyncio
import pickle

from langchain_classic.retrievers import MultiVectorRetriever
from unstructured.documents.elements import Element, Image
from unstructured.staging.base import elements_from_dicts

from extract_data.summaries_data import print_element
from ingestion.extract_elements import extract_tables_texts_images
from ingestion.parser import async_parse_input_document
from llm.chains import summaries_text_data, summaries_table_data, summaries_images, rag_answer_chain
from llm.preprocess import table_to_prompt_text
from repository.storage import ChromaWork
from retrieval.get_elements import build_context


async def run():
    # print('Разбивка документа')
    # chunks_: list[Element] = await async_parse_input_document()
    # tables, texts, images = await extract_tables_texts_images(chunks_)
    #
    # print('Старт обработки текста')
    # summarize_chain_text = summaries_text_data()
    # text_summaries = await summarize_chain_text.abatch(texts, {'max_concurrency': 3})
    #
    # print('Старт обработки таблиц')
    # summarize_chain_table = summaries_table_data()
    # table_inputs = [await table_to_prompt_text(t) for t in tables]
    # table_summaries = await summarize_chain_table.abatch(table_inputs, {'max_concurrency': 3})
    #
    # print('Старт обработки изображений')
    # image_summaries: list[str] = [await summaries_images(img) for img in images]

    # work with db
    chroma_work = ChromaWork()
    retriever: MultiVectorRetriever = await chroma_work.init_db()

    # print('Добавление данных в БД')
    # chroma_work.add_element(texts, text_summaries)
    # chroma_work.add_element(tables, table_summaries)
    # chroma_work.add_element(images, image_summaries)

    print('Вопрос-ответ...')
    question: str = 'What is the "Scaled Dot-Product Attention"?'
    chunks_new: list[bytes] = await retriever.ainvoke(question)

    result_chunks: list[Element] = []
    image_chunks: list[Element] = []
    for num, raw in enumerate(chunks_new):
        element_dict: dict = pickle.loads(raw)
        element_list: list[Element] = await asyncio.to_thread(elements_from_dicts, [element_dict])

        # only for debug
        # await print_element(element_list[0], num)
        # result_chunks.extend(element_list)

        if isinstance(element_list[0], Image):
            image_chunks.append(element_list[0])
        else:
            result_chunks.append(element_list[0])

    context: str = await build_context(result_chunks)

    print(context)
    exit()

    chain = rag_answer_chain()
    answer: str = await chain.ainvoke({'context': context, 'question': question})

    print('\n\n', answer, '\n\n')

    for img in image_chunks:
        print(f"<img src='data:image/png;base64,{img.metadata.image_base64}'/>")


if __name__ == '__main__':
    asyncio.run(run())
