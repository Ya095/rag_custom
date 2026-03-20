import asyncio
import logging
import uuid
from copy import copy
from io import BytesIO

from unstructured.documents.elements import (
    Element,
    Table,
    CompositeElement,
    Image,
    Footer,
    Text,
    PageNumber,
    Header,
    PageBreak,
    Address,
    EmailAddress,
)
from unstructured.partition.pdf import partition_pdf

from ingestion.interfaces import IProcessDocument
from llm.chains import summaries_text_data, summaries_table_data, summaries_images
from llm.preprocess import table_to_prompt_text
from repository.storage import ChromaWork


logger = logging.getLogger(__name__)
SKIP_TYPES = (
    Footer,
    Header,
    PageNumber,
    PageBreak,
    Address,
    EmailAddress,
)


class ProcessDocumentPDF(IProcessDocument):
    def __init__(self):
        self.chroma_work = ChromaWork()
        self.source_doc_id: str = str(uuid.uuid4())
        self.filename: str = ''

    async def process_document(self, input_file: BytesIO, filename: str) -> str:
        """Processes the incoming document.

        It splits the data into chunks and writes it to the database and file storage.
        """

        self.filename = filename
        if self.chroma_work.retriever is None:
            await self.chroma_work.init_db()

        chunks_: list[Element] = await self.parse_input_document(input_file)
        extracted_elements: dict[str, list[Element]] = await self._extract_tables_texts_images(chunks_)
        data_for_save: dict[str, tuple | list[Element]] = await self._process_extracted_elements(extracted_elements)
        await self._save_data_to_storage(data_for_save)

    async def parse_input_document(self, input_file: BytesIO) -> list[Element]:
        return await asyncio.to_thread(self._parse_input_document, input_file)

    async def _extract_tables_texts_images(self, chunks: list[Element]) -> dict[str, list[Element]]:
        """Extract data from chunks."""

        result: dict[str, list[Element]] = {
            'tables': [],
            'texts': [],
            'images_for_description': [],
            'images_for_file_storage_add': [],
        }

        for chunk in chunks:
            chunk.metadata.filename = self.filename

            # top-level table
            if isinstance(chunk, Table):
                result['tables'].append(chunk)
                continue

            # top-level image
            if isinstance(chunk, Image):
                result['images_for_description'].append(chunk)
                continue

            if isinstance(chunk, CompositeElement):
                orig_elements: list[Element] = chunk.metadata.orig_elements or []
                text_elements: list[Element] = []

                for el in orig_elements:
                    if isinstance(el, Image):
                        img_uid: str = f'{self.source_doc_id}_{el.id}'
                        el.metadata.img_uid = img_uid
                        result['images_for_file_storage_add'].append(el)

                        placeholder = Text(
                            text=f'[[IMG:{img_uid}]]',
                            metadata=copy(el.metadata),
                        )
                        text_elements.append(placeholder)
                    elif isinstance(el, SKIP_TYPES):
                        continue
                    else:
                        if len(el.text.strip()) > 20:
                            text_elements.append(el)

                # copy of CompositeElement, with images like a tag
                if text_elements:
                    chunk_copy: CompositeElement = copy(chunk)
                    chunk_copy.metadata = copy(chunk.metadata)
                    chunk_copy.metadata.orig_elements = text_elements
                    result['texts'].append(chunk_copy)

        return result

    async def _process_extracted_elements(
        self,
        extracted_elements: dict[str, list[Element]],
    ) -> dict[str, tuple[list[Element], str] | list[Element]]:
        """Processing of the elements that were identified in the document."""

        tables: list[Element] = extracted_elements['tables']
        plain_text: list[Element] = extracted_elements['texts']
        images_for_description: list[Element] = extracted_elements['images_for_description']
        images_from_text: list[Element] = extracted_elements['images_for_file_storage_add']

        logger.info('Start text processing.')
        summarize_chain_text = summaries_text_data()
        text_summaries: str = await summarize_chain_text.abatch(
            plain_text,
            {
                'rpm': 30,
                'max_concurrency': 2,
            },
        )

        logger.info('Start tables processing.')
        summarize_chain_table = summaries_table_data()
        table_inputs: list[str] = [await table_to_prompt_text(t) for t in tables]
        table_summaries: str = await summarize_chain_table.abatch(
            table_inputs,
            {
                'rpm': 30,
                'max_concurrency': 2,
            },
        )

        logger.info('Start imgs processing.')
        image_summaries: list[str] = [await summaries_images(img) for img in images_for_description]

        return {
            'text_data': (plain_text, text_summaries),
            'table_data': (tables, table_summaries),
            'images_data': (images_for_description, image_summaries),
            'only_for_storage_data': images_from_text,
        }

    async def _save_data_to_storage(self, data_for_save: dict[str, tuple]) -> None:
        """Saving the processed data to the database and storage."""

        text_data: tuple = data_for_save['text_data']
        table_data: tuple = data_for_save['table_data']
        images_data: tuple = data_for_save['images_data']
        only_for_storage_data: list[Element] = data_for_save['only_for_storage_data']

        await self.chroma_work.async_add_elements(text_data[0], text_data[1], self.source_doc_id)
        await self.chroma_work.async_add_elements(table_data[0], table_data[1], self.source_doc_id)
        await self.chroma_work.async_add_elements(images_data[0], images_data[1], self.source_doc_id)
        await self.chroma_work.async_add_elements_only_to_storage(only_for_storage_data, self.source_doc_id)

    def _parse_input_document(self, input_file: BytesIO) -> list[Element]:
        """Parse intput document from docs."""

        chunks: list[Element] = partition_pdf(
            file=input_file,
            languages=['eng'],
            infer_table_structure=True,  # extract tables
            strategy='hi_res',  # mandatory to infer tables
            extract_image_block_types=['Image'],  # Add 'Table' to list to extract image of tables
            extract_image_block_to_payload=True,  # if true, will extract base64 for API usage
            chunking_strategy='by_title',
            max_characters=2000,
            new_after_n_chars=1500,
            combine_text_under_n_chars=400,
        )

        return chunks


async def main():
    from config import config

    doc_path = config.APP_PATH / 'documents'
    with open(doc_path / 'attention.pdf', 'rb') as f:
        file_ = BytesIO(f.read())

    obj = ProcessDocumentPDF()
    s_d_id = await obj.process_document(file_, 'attention.pdf')

    print(s_d_id)
    print('DONE')


if __name__ == '__main__':
    asyncio.run(main())
