import base64
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_classic.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

from config import APP_PATH
from embeddings.adapter_embedding import SentenceTransformerEmbeddings


class ChromaWork:
    def __init__(self):
        self.db_path: Path = APP_PATH / 'chroma.db'
        self.vectorstore: Chroma | None = None
        self.store: InMemoryStore | None = None
        self.retriever: MultiVectorRetriever | None = None
        self.id_key: str = 'doc_id'

    @lru_cache
    def init_db(self) -> MultiVectorRetriever:
        """Initializing: Chroma + InMemoryStore + MultiVectorRetriever."""

        embeddings = SentenceTransformerEmbeddings(model_name='multi-qa-mpnet-base-dot-v1')
        self.vectorstore = Chroma(
            collection_name='multi_modal_rag',
            collection_metadata={'description': 'Attention for transformers.'},
            embedding_function=embeddings,
            persist_directory=self.db_path.absolute().as_posix(),
        )

        self.store = InMemoryStore()

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key=self.id_key,
        )

        return self.retriever

    def add_text(self, texts: list[str], summaries: list[str]) -> None:
        """Adding texts and it`s summaries."""

        if not texts or not summaries:
            print('No texts or summaries.')
            return

        doc_ids: list[str] = self.__generate_uuids(texts)
        summary_docs: list[Document] = [
            Document(page_content=summaries[num], metadata={self.id_key: doc_ids[num]})
            for num in range(len(texts))
        ]

        self.vectorstore.add_documents(summary_docs)
        self.store.mset(list(zip(doc_ids, texts)))

    def add_tables(self, tables: list[str], table_summaries: list[str]) -> None:
        """Adding tables and it`s summaries."""

        if not tables or not table_summaries:
            print('No tables or table_summaries.')
            return

        table_ids: list[str] = self.__generate_uuids(tables)
        summary_docs: list[Document] = [
            Document(page_content=table_summaries[num], metadata={self.id_key: table_ids[num]})
            for num in range(len(tables))
        ]

        self.vectorstore.add_documents(summary_docs)
        self.store.mset(list(zip(table_ids, tables)))

    def add_images(self, images: list[base64], image_summaries: list[str]) -> None:
        """Adding images and it`s summaries."""

        if not images or not image_summaries:
            print('No images or image_summaries.')
            return

        img_ids: list[str] = self.__generate_uuids(images)
        summary_docs: list[Document] = [
            Document(page_content=image_summaries[num], metadata={self.id_key: img_ids[num]})
            for num in range(len(images))
        ]

        self.vectorstore.add_documents(summary_docs)
        self.store.mset(list(zip(img_ids, images)))

    @staticmethod
    def __generate_uuids(seq: list[Any]) -> list[str]:
        """Generate some of uuid4 for sequence."""

        return [str(uuid.uuid4()) for _ in seq]
