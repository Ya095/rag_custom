import asyncio
import pickle

from langchain_classic.retrievers import MultiVectorRetriever
from unstructured.documents.elements import Element, Image
from unstructured.staging.base import elements_from_dicts

from ingestion.extract_elements import extract_tables_texts_images
from ingestion.parser import async_parse_input_document
from llm.chains import summaries_text_data, summaries_table_data, summaries_images, rag_answer_chain
from llm.preprocess import table_to_prompt_text
from repository.storage import ChromaWork
from retrieval.get_elements import build_context


async def run():
    # print('Разбивка документа')
    # chunks_, source_doc_id = await async_parse_input_document()
    # extracted_els: dict[str, list[Element]] = await extract_tables_texts_images(chunks_, source_doc_id)
    #
    # tables: list[Element] = extracted_els['tables']
    # plain_text: list[Element] = extracted_els['texts']
    # images_for_description: list[Element] = extracted_els['images_for_description']
    # images_from_text: list[Element] = extracted_els['images_for_file_storage_add']
    #
    # print('Старт обработки текста')
    # summarize_chain_text = summaries_text_data()
    # text_summaries = await summarize_chain_text.abatch(plain_text, {'max_concurrency': 3})
    #
    # print('Старт обработки таблиц')
    # summarize_chain_table = summaries_table_data()
    # table_inputs = [await table_to_prompt_text(t) for t in tables]
    # table_summaries = await summarize_chain_table.abatch(table_inputs, {'max_concurrency': 3})
    #
    # print('Старт обработки изображений')
    # image_summaries: list[str] = [await summaries_images(img) for img in images_for_description]

    # work with db
    chroma_work = ChromaWork()
    retriever: MultiVectorRetriever = await chroma_work.init_db()

    # print('Добавление данных в БД')
    # await chroma_work.async_add_elements(plain_text, text_summaries, source_doc_id)
    # await chroma_work.async_add_elements(tables, table_summaries, source_doc_id)
    # await chroma_work.async_add_elements(images_for_description, image_summaries, source_doc_id)
    # await chroma_work.async_add_elements_only_to_storage(images_from_text, source_doc_id)

    print('Вопрос-ответ...')
    question: str = 'What is the Multi-Head Attention?'
    chunks_new: list[bytes] = await retriever.ainvoke(question)

    retrieved: list[Element] = []

    for raw in chunks_new:
        element_dict = pickle.loads(raw)
        element_list: list[Element] = await asyncio.to_thread(elements_from_dicts, [element_dict])

        el: Element = element_list[0]
        retrieved.append(el)

    context: str = await build_context(retrieved)

    # todo указать модели, что бы она старалась в ответе вставлять изображения
    #  внутри контекста а не после него, если это возможно. Что бы сохранить оригинальный порядок и контекст.
    chain = rag_answer_chain()
    answer: str = await chain.ainvoke({'context': context, 'question': question})

    print('\n\n', answer, '\n\n')

    # todo доделать замену тега на base64 обратно
    # for img_uid, b64 in img_map.items():
    # for
    #     final_answer = final_answer.replace(
    #         f"[[IMG:{img_uid}]]",
    #         f"<img src='data:image/png;base64,{b64}'/>"
    #     )

    print('\n-----------------------------------------\n', answer)


if __name__ == '__main__':
    asyncio.run(run())
