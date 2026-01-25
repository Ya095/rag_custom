import copy

from unstructured.documents.elements import Element, Table, CompositeElement, Image, FigureCaption, Footer


async def extract_tables_texts_images(chunks: list[Element]) -> tuple[list[Element], list[Element], list[Element]]:
    """Extract data from chunks."""

    tables: list[Element] = []
    texts: list[Element] = []
    images: list[Element] = []

    for chunk in chunks:
        # top-level table
        if isinstance(chunk, Table):
            tables.append(chunk)
            continue

        # top level image
        if isinstance(chunk, Image):
            images.append(chunk)
            continue

        if isinstance(chunk, CompositeElement):
            orig_elements: list[Element] = chunk.metadata.orig_elements or []
            text_elements: list[Element] = []

            for el in orig_elements:
                if isinstance(el, Table):
                    tables.append(el)
                elif isinstance(el, Image):
                    images.append(el)
                elif isinstance(el, (FigureCaption, Footer)):
                    continue
                else:
                    if len(el.text) > 3:
                        text_elements.append(el)

            # copy of CompositeElement, without tables and images
            if text_elements:
                chunk_copy: CompositeElement = copy.copy(chunk)
                chunk_copy.metadata = copy.copy(chunk.metadata)
                chunk_copy.metadata.orig_elements = text_elements
                texts.append(chunk_copy)

    return tables, texts, images
