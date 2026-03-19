from dataclasses import dataclass

from repository.storage import ChromaWork
from services import (
    RetrievalService,
    LLMService,
    ImageService,
    QuestionService,
    IRetrievalService,
    ILLMService,
    IImageService,
    IQuestionService,
)


@dataclass
class Container:
    _chroma_work: ChromaWork | None = None
    _retrieval_service: IRetrievalService | None = None
    _llm_service: ILLMService | None = None
    _image_service: IImageService | None = None
    _question_service: IQuestionService | None = None

    @property
    def chroma_work(self) -> ChromaWork:
        if self._chroma_work is None:
            self._chroma_work = ChromaWork()
        return self._chroma_work

    @property
    def retrieval_service(self) -> IRetrievalService:
        if self._retrieval_service is None:
            self._retrieval_service = RetrievalService(self.chroma_work)
        return self._retrieval_service

    @property
    def llm_service(self) -> ILLMService:
        if self._llm_service is None:
            self._llm_service = LLMService()
        return self._llm_service

    @property
    def image_service(self) -> IImageService:
        if self._image_service is None:
            self._image_service = ImageService(self.chroma_work)
        return self._image_service

    @property
    def question_service(self) -> IQuestionService:
        if self._question_service is None:
            self._question_service = QuestionService(
                self.retrieval_service,
                self.llm_service,
                self.image_service,
            )
        return self._question_service


container = Container()
