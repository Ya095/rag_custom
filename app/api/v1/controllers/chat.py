from typing import Annotated

from fastapi import APIRouter, Depends, status

from api.v1.schemas import QuestionChatModel
from retrieval.interfaces import IProcesUserQuestion
from retrieval.process_question import ProcessQuestion


router = APIRouter(prefix='/answer', tags=['Answers'])


@router.post(
    '/one_answer_only',
    status_code=status.HTTP_202_ACCEPTED,
)
async def answer_one_only(
    service: Annotated[IProcesUserQuestion, Depends(ProcessQuestion)],
    input_data: QuestionChatModel,
) -> dict:
    """Process only one user question and return an answer."""

    result: dict = await service.get_answer(input_data.question)

    return result
