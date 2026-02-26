from abc import ABC, abstractmethod
from io import BytesIO

from unstructured.documents.elements import Element


class IProcessDocument(ABC):
    """An interface for document processing classes such as .pdf, .docx, and so on."""

    @abstractmethod
    async def process_document(self, input_file: BytesIO) -> str:
        pass

    @abstractmethod
    async def parse_input_document(self, input_file: BytesIO) -> list[Element]:
        pass

    @abstractmethod
    async def _process_extracted_elements(self, *args, **kwargs):
        pass

    @abstractmethod
    async def _save_data_to_storage(self, *args, **kwargs):
        pass
