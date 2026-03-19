from typing import Annotated

from fastapi import APIRouter, Depends, status

from api.v1.schemas import QuestionChatModel
from containers.container import container
from services.interfaces import IQuestionService


router = APIRouter(prefix="/answer", tags=["Answers"])


def get_question_service() -> IQuestionService:
    return container.question_service


@router.post(
    "/one_answer_only",
    status_code=status.HTTP_202_ACCEPTED,
)
async def answer_one_only(
    service: Annotated[IQuestionService, Depends(get_question_service)],
    input_data: QuestionChatModel,
) -> dict:
    """Process only one user question and return an answer."""

    result = await service.get_answer(input_data.question)

    return {
        "llm_answer": result.text,
        "sources": [
            {"document_name": s.document_name, "page": s.page} for s in result.sources
        ],
    }
