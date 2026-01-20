import copy

from unstructured.documents.elements import Element, Table, CompositeElement, Image


def extract_tables_texts_images(chunks: list[Element]) -> tuple[list[Element], list[Element], list[Element]]:
    tables: list[Element] = []
    texts: list[Element] = []
    images: list[Element] = []

    for chunk in chunks:
        # top-level table
        if isinstance(chunk, Table):
            tables.append(chunk)
            continue

        if isinstance(chunk, CompositeElement):
            orig_elements = chunk.metadata.orig_elements or []

            text_elements = []

            for el in orig_elements:
                if isinstance(el, Table):
                    tables.append(el)
                elif isinstance(el, Image):
                    images.append(el)
                else:
                    text_elements.append(el)

            # создаём КОПИЮ CompositeElement без таблиц и картинок
            if text_elements:
                chunk_copy = copy.copy(chunk)
                chunk_copy.metadata = copy.copy(chunk.metadata)
                chunk_copy.metadata.orig_elements = text_elements
                texts.append(chunk_copy)

    return tables, texts, images


# def tables_and_texts(chunks: list[Element]) -> tuple[list[Element], list[Element]]:
#     tables = []
#     texts = []
#
#     for chunk in chunks:
#         if chunk.category == 'Table':
#             tables.append(chunk)
#         elif chunk.category == 'CompositeElement':
#             texts.append(chunk)
#
#     return tables, texts
#
#
# def get_images(chunks: list[Element]) -> list[str]:
#     """Get the images from the CompositeElement objects."""
#
#     images = []
#     for chunk in chunks:
#         if chunk.category == 'CompositeElement':
#             chunk_els = chunk.metadata.orig_elements
#             for el in chunk_els:
#                 if el.category == 'Image':
#                     images.append(el.metadata.image_base64)
#
#     return images


# def extract_tables_images_texts(chunks: list[Element]) -> dict[str, list[Element]]:
#     """Extract tables, images and texts from original chunks."""
#
#     tables: list[Element] = []
#     texts: list[Element] = []
#     images: list[Element] = []
#
#     for chunk in chunks:
#         if chunk.category == 'CompositeElement':
#             chunk_text_els: list[Element] = []
#             chunk_els: list[Element] = chunk.metadata.orig_elements
#
#             for el in chunk_els:
#                 if el.category == 'Table':
#                     tables.append(el)
#                 elif el.category == 'Image':
#                     images.append(el)
#                 else:
#                     chunk_text_els.append(el)
#
#             chunk.metadata.orig_elements = chunk_text_els
#             if chunk_text_els:
#                 texts.append(chunk)
#
#         else:
#             if chunk.category == 'Table':
#                 tables.append(chunk)
#             elif chunk.category == 'Image':
#                 images.append(chunk)
#             else:
#                 texts.append(chunk)
#
#     return {
#         'images': images,
#         'texts': texts,
#         'tables': tables,
#     }
