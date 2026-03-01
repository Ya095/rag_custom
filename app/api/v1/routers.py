from fastapi import APIRouter
from .controllers.process_new_file import router as process_new_document_router
from .controllers.chat import router as chat_router


router = APIRouter(prefix='/api/v1')

router.include_router(process_new_document_router)
router.include_router(chat_router)
