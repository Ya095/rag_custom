from pydantic import BaseModel


class QuestionChatModel(BaseModel):
    question: str


class SourcesModel(BaseModel):
    page: int
    document_name: str


class AnswerChatModel(BaseModel):
    llm_answer: str
    sources: list[SourcesModel]
