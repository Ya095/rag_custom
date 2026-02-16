import asyncio
import pickle

from unstructured.documents.elements import Element
from unstructured.staging.base import elements_from_dicts

from llm.chains import rag_answer_chain
from repository.storage import ChromaWork
from retrieval.get_elements import build_context


class ProcessQuestion:
    def __init__(self):
        self.chroma_work = ChromaWork()

    def process_pipeline_sync(self, question: str) -> dict:
        return asyncio.run(self.process_pipeline(question))

    async def process_pipeline(self, question: str) -> dict:
        """Processing user question and return an answer."""

        if self.chroma_work.retriever is None:
            await self.chroma_work.init_db()

        img_uids: list[str] = []
        data: dict = await self.get_context_and_source_metadata(question, img_uids)

        print('Waiting answer from model...')
        chain = rag_answer_chain()
        answer_with_image_tags: str = await chain.ainvoke({'context': data['context'], 'question': question})

        print('\n\n', answer_with_image_tags, '\n\n')

        llm_answer: str = await self.get_answer_with_images(
            img_uids,
            answer_with_image_tags,
        )

        return {
            'llm_answer': llm_answer,
            'sources': data['source_data'],
        }

    async def get_context_and_source_metadata(self, question: str, img_uids: list[str]) -> dict:
        """Getting data from the database as list of Elements."""

        print('Start getting data.')
        chunks_new: list[bytes] = await self.chroma_work.retriever.ainvoke(question)
        retrieved: list[Element] = []
        source_data: list[dict[str, int | str]] = []

        for raw in chunks_new:
            element_dict: dict = pickle.loads(raw)
            element_list: list[Element] = await asyncio.to_thread(elements_from_dicts, [element_dict])

            el: Element = element_list[0]
            retrieved.append(el)

            source_data.append(
                {
                    'page': el.metadata.page_number,
                    'document_name': el.metadata.filename,
                }
            )

            for sub_el in el.metadata.orig_elements:
                if getattr(sub_el.metadata, 'img_uid', None) is not None:
                    img_uids.append(sub_el.metadata.img_uid)

        print('Start build context.')
        context: str = await build_context(retrieved)

        print('------------- CONTEXT ------------\n', context, '\n')

        return {
            'context': context,
            'source_data': source_data,
        }

    async def get_answer_with_images(self, img_uids: list[str], llm_answer_with_image_tag) -> str:
        """Replaces the tag of images in the LLM response with specific images in base64 format."""

        print('Start replacing tags to images (base64)...')
        llm_answer_with_base64_imgs: str = llm_answer_with_image_tag

        for img_uid in img_uids:
            img_b64_raw: bytes | None = await self.chroma_work.get_content_from_storage(img_uid)

            if img_b64_raw is not None:
                element_dict: dict = pickle.loads(img_b64_raw)
                element_list: list[Element] = await asyncio.to_thread(elements_from_dicts, [element_dict])
                img_b64: str = element_list[0].metadata.image_base64

                new_img_element: str = (
                    f"<img src='data:image/png;base64,{img_b64}'"
                    f" style='max-width:80%; height:auto;'/>"
                )
                llm_answer_with_base64_imgs = llm_answer_with_base64_imgs.replace(
                    f"[[IMG:{img_uid}]]", new_img_element
                )

        print('\n-----------------------------------------\n', llm_answer_with_base64_imgs)

        return llm_answer_with_base64_imgs


async def main():
    obj = ProcessQuestion()
    await obj.process_pipeline('What is the Encoder and Decoder Stacks?')

if __name__ == '__main__':
    asyncio.run(main())
