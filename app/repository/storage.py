import asyncio
import pickle
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from unstructured.documents.elements import Element

import core.config as config
from repository.embeddings import SentenceTransformerEmbeddings
from utils.singleton import SingletonMeta


class ChromaWork(metaclass=SingletonMeta):
    """Chroma + LocalFileStore + MultiVectorRetriever. For multimodal (text / table / image) RAG."""

    def __init__(self) -> None:
        self.db_path: Path = config.APP_PATH / 'chroma.db'
        self.doc_store_path: Path = config.APP_PATH / 'docstore'

        self.vectorstore: Chroma | None = None
        self.docstore: LocalFileStore | None = None
        self.retriever: MultiVectorRetriever | None = None

        self.id_key: str = 'doc_id'

    async def init_db(self) -> MultiVectorRetriever:
        """Initialize vectorstore, docstore and retriever."""

        embeddings = SentenceTransformerEmbeddings(
            model_name='multi-qa-mpnet-base-dot-v1',
            device=config.DEVICE,
        )

        self.vectorstore = Chroma(
            collection_name='multi_modal_rag',
            collection_metadata={'description': 'Multimodal RAG: text, tables and images'},
            embedding_function=embeddings,
            persist_directory=self.db_path.as_posix(),
        )

        self.docstore = LocalFileStore(root_path=self.doc_store_path.as_posix())

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key,
            search_kwargs={'k': 3},
        )

        return self.retriever

    async def async_add_elements_only_to_storage(
        self,
        elements: list[Element],
        source_doc_id: str | None = None,
    ) -> None:
        """Add elements only to file storage."""

        if not elements:
            print('Nothing to add: empty elements.')
            return

        if self.vectorstore is None or self.docstore is None:
            raise RuntimeError('Database is not initialized. Call init_db() first.')

        serialized_elements: list[tuple[str, bytes]] = []

        for idx, element in enumerate(elements):
            doc_id: str = f'{source_doc_id}_{element.id}'
            element.metadata.doc_id = doc_id
            serialized_elements.append((doc_id, await self._async_serialize_element(element)))

        await asyncio.to_thread(self.docstore.mset, serialized_elements)

    async def async_add_elements(
        self,
        elements: list[Element],
        summaries: list[str],
        source_doc_id: str | None = None,
    ) -> None:
        """Add elements with their summaries to vectorstore and originals to docstore.

        Each summary is embedded.
        Each original Element is serialized and stored in docstore.
        """

        if not elements or not summaries:
            print('Nothing to add: empty elements or summaries.')
            return

        if len(elements) != len(summaries):
            raise ValueError('Elements and summaries length mismatch.')

        if self.vectorstore is None or self.docstore is None:
            raise RuntimeError('Database is not initialized. Call init_db() first.')

        summary_docs: list[Document] = []
        serialized_elements: list[tuple[str, bytes]] = []

        for idx, element in enumerate(elements):
            doc_id: str = f'{source_doc_id}_{element.id}'
            element.metadata.doc_id = doc_id

            summary_docs.append(
                Document(
                    page_content=summaries[idx],
                    metadata={
                        self.id_key: doc_id,
                        'category': element.category,
                        'source_doc_id': source_doc_id,
                        'img_uid': getattr(element.metadata, 'img_uid', None),
                    },
                )
            )

            serialized_elements.append((doc_id, await self._async_serialize_element(element)))

        await self.vectorstore.aadd_documents(summary_docs)
        await asyncio.to_thread(self.docstore.mset, serialized_elements)

    async def get_content_from_storage(self, key: str) -> bytes | None:
        """Возвращает bytes файл по ключу."""

        file_path = self.doc_store_path / key
        if file_path.exists():
            return file_path.read_bytes()

        print('File not found by key:', key)

    @staticmethod
    async def _async_serialize_element(element: Element) -> bytes:
        """Serialize Element with all metadata preserved."""

        return pickle.dumps(element.to_dict())
