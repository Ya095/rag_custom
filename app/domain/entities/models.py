from dataclasses import dataclass


@dataclass
class Source:
    document_name: str
    page: int


@dataclass
class Answer:
    text: str
    sources: list[Source]
    images: list[str] | None = None
