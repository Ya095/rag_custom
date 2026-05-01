from typing import Annotated, AsyncGenerator
import json

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from api.v1.schemas import QuestionChatModel
from containers.container import container
from services.interfaces import IQuestionService


router = APIRouter(prefix="/answer", tags=["Answers"])


def get_question_service() -> IQuestionService:
    return container.question_service


async def sse_generator(service: IQuestionService, question: str) -> AsyncGenerator[bytes, None]:
    async for event in service.stream_answer(question):
        event_type = event['type']
        data = event['data']

        if event_type == 'done':
            yield f'event: done\ndata:\n\n'.encode()
        else:
            yield f'event: {event_type}\ndata: {json.dumps(data)}\n\n'.encode()


@router.post(
    '/one_answer_only',
    status_code=status.HTTP_202_ACCEPTED,
)
async def answer_one_only(
    service: Annotated[IQuestionService, Depends(get_question_service)],
    input_data: QuestionChatModel,
) -> dict:
    """Process only one user question and return an answer."""

    result = await service.get_answer(input_data.question)

    return {
        'llm_answer': result.text,
        'sources': [
            {'document_name': s.document_name, 'page': s.page} for s in result.sources
        ],
    }


@router.post(
    '/stream',
    status_code=status.HTTP_200_OK,
)
async def stream_answer(
    service: Annotated[IQuestionService, Depends(get_question_service)],
    input_data: QuestionChatModel,
) -> StreamingResponse:
    return StreamingResponse(
        sse_generator(service, input_data.question),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )
