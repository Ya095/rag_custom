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
    question: str = 'What is the Encoder and Decoder Stacks?'
    chunks_new: list[bytes] = await retriever.ainvoke(question)

    retrieved: list[Element] = []
    img_uids: list[str] = []

    for raw in chunks_new:
        element_dict = pickle.loads(raw)
        element_list: list[Element] = await asyncio.to_thread(elements_from_dicts, [element_dict])

        el: Element = element_list[0]
        retrieved.append(el)

        for sub_el in el.metadata.orig_elements:
            if getattr(sub_el.metadata, 'img_uid', None) is not None:
                img_uids.append(sub_el.metadata.img_uid)

    context: str = await build_context(retrieved)

    chain = rag_answer_chain()
    answer_with_image_uid: str = await chain.ainvoke({'context': context, 'question': question})

    print('\n\n', answer_with_image_uid, '\n\n')

    llm_answer: str = answer_with_image_uid

    for img_uid in img_uids:
        img_b64_raw: bytes | None = await chroma_work.get_content_from_storage(img_uid)

        if img_b64_raw is not None:
            element_dict: dict = pickle.loads(img_b64_raw)
            element_list: list[Element] = await asyncio.to_thread(elements_from_dicts, [element_dict])
            img_b64: str = element_list[0].metadata.image_base64

            llm_answer = llm_answer.replace(
                f"[[IMG:{img_uid}]]",
                f"<img src='data:image/png;base64,{img_b64}'/>"
            )

    print('\n-----------------------------------------\n', llm_answer)


if __name__ == '__main__':
    asyncio.run(run())
