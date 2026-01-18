import base64
from io import BytesIO
from unstructured.documents.elements import Element

from extract_data.get_doc_chunks import chunks as doc_chunks
from PIL import Image


def tables_and_texts(chunks: list[Element]) -> tuple[list, list]:
    tables = []
    texts = []

    for chunk in chunks:
        if 'Table' in str(type(chunk)):
            tables.append(chunk)

        if 'CompositeElement' in str(type(chunk)):
            texts.append(chunk)

    return tables, texts


def get_images(chunks: list[Element]) -> list[base64]:
    """Get the images from the CompositeElement objects."""

    images = []
    for chunk in chunks:
        if 'CompositeElement' in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if 'Image' in str(type(el)):
                    images.append(el.metadata.image_base64)

    return images


def display_base64_images(base64_code) -> None:
    image_data = base64.b64decode(base64_code)
    image_file = BytesIO(image_data)
    image = Image.open(image_file)

    image.show()


if __name__ == '__main__':
    chunks_: list[Element] = doc_chunks
    images_list = get_images(chunks_)
    display_base64_images(images_list[0])

