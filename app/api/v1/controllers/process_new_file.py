from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, Depends, status, UploadFile, BackgroundTasks

from ingestion.interfaces import IProcessDocument
from ingestion.pipeline import ProcessDocumentPDF


router = APIRouter(prefix='/add_new_file', tags=['Process new documents'])


@router.post(
    '/pdf',
    status_code=status.HTTP_202_ACCEPTED,
)
async def add_new_pdf_document(
    background_tasks: BackgroundTasks,
    service: Annotated[IProcessDocument, Depends(ProcessDocumentPDF)],
    input_file: UploadFile,
) -> dict[str, str]:
    """Uploads and processes a new document."""

    pdf_file: bytes = await input_file.read()
    background_tasks.add_task(
        service.process_document,
        BytesIO(pdf_file),
        input_file.filename,
    )

    return {'message': 'Document accepted for processing.'}
