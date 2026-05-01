import asyncio
import logging
import pickle
from collections import defaultdict

from unstructured.documents.elements import Element
from unstructured.staging.base import elements_from_dicts

from exceptions.exceptions import RetrievalError
from repository.storage import ChromaWork
from retrieval.get_elements import build_context
from .interfaces import IRetrievalService


logger = logging.getLogger(__name__)


class RetrievalService(IRetrievalService):
    def __init__(self, chroma_work: ChromaWork):
        self.chroma_work = chroma_work

    async def retrieve_context(self, question: str) -> tuple[str, dict, list[str]]:
        logger.info('Start getting data.')

        if self.chroma_work.retriever is None:
            await self.chroma_work.init_db()

        img_uids: list[str] = []
        chunks_new: list[bytes] = await self.chroma_work.retriever.ainvoke(question)

        retrieved: list[Element] = []
        source_data: dict[str, set[int]] = defaultdict(set)

        for raw in chunks_new:
            try:
                element_dict: dict = pickle.loads(raw)
                element_list: list[Element] = await asyncio.to_thread(
                    elements_from_dicts, [element_dict]
                )

                el: Element = element_list[0]
                retrieved.append(el)

                filename = el.metadata.filename
                page_num = el.metadata.page_number
                if filename:
                    source_data[filename].add(page_num if page_num else 0)

                orig_elements = el.metadata.orig_elements
                if orig_elements:
                    for sub_el in orig_elements:
                        img_uid = getattr(sub_el.metadata, 'img_uid', None)
                        if img_uid:
                            img_uids.append(img_uid)

            except Exception as err:
                logger.error('Error processing chunk: %s', str(err))
                raise RetrievalError(f'Failed to process retrieved chunk: {err}') from err

        logger.info('Start build context.')
        context: str = await build_context(retrieved)

        return context, source_data, img_uids
