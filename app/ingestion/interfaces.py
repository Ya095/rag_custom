from abc import ABC, abstractmethod
from io import BytesIO


class IProcessDocument(ABC):
    """An interface for document processing classes such as .pdf, .docx, and so on."""

    @abstractmethod
    async def process_document(self, input_file: BytesIO, filename: str) -> str:
        pass
