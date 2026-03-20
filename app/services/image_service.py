import asyncio
import logging
import pickle

from unstructured.documents.elements import Element
from unstructured.staging.base import elements_from_dicts

from exceptions.exceptions import ImageProcessingError
from repository.storage import ChromaWork
from services import IImageService


logger = logging.getLogger(__name__)


class ImageService(IImageService):
    def __init__(self, chroma_work: ChromaWork):
        self._chroma_work = chroma_work

    async def process_images(self, answer: str, img_uids: list[str]) -> str:
        logger.info('Start replacing tags to images (base64)...')

        result = answer

        for img_uid in img_uids:
            try:
                img_b64_raw: bytes | None = await self._chroma_work.get_content_from_storage(img_uid)

                if img_b64_raw is None:
                    logger.warning('Image not found in storage: %s', img_uid)
                    continue

                element_dict: dict = pickle.loads(img_b64_raw)
                element_list: list[Element] = await asyncio.to_thread(
                    elements_from_dicts, [element_dict]
                )
                img_b64: str | None = element_list[0].metadata.image_base64

                if img_b64 is None:
                    logger.warning('Image base64 not found for uid: %s', img_uid)
                    continue

                new_img_element = (
                    f"<img id='{img_uid}' src='data:image/png;base64,{img_b64}'"
                    f" style='max-width:80%; height:auto;'/>"
                )
                result = result.replace(f'[[IMG:{img_uid}]]', new_img_element)

            except Exception as err:
                logger.error('Error processing image %s: %s', img_uid, str(err))
                raise ImageProcessingError(f'Failed to process image {img_uid}: {err}') from err

        logger.debug('Final answer: %s', result)
        return result
