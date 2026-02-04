from copy import copy

from unstructured.documents.elements import (
    Element,
    Table,
    CompositeElement,
    Image,
    Footer,
    Text,
)


async def extract_tables_texts_images(
    chunks: list[Element],
    source_doc_id: str | None = None,
) -> dict[str, list[Element]]:
    """Extract data from chunks."""

    result: dict[str, list[Element]] = {
        'tables': [],
        'texts': [],
        'images_for_description': [],
        'images_for_file_storage_add': [],
    }

    for chunk in chunks:
        # top-level table
        if isinstance(chunk, Table):
            result['tables'].append(chunk)
            continue

        # top-level image
        if isinstance(chunk, Image):
            result['images_for_description'].append(chunk)
            continue

        if isinstance(chunk, CompositeElement):
            orig_elements: list[Element] = chunk.metadata.orig_elements or []
            text_elements: list[Element] = []

            for el in orig_elements:
                if isinstance(el, Image):
                    el.metadata.img_uid = el.id
                    result['images_for_file_storage_add'].append(el)

                    placeholder = Text(
                        text=f"[[IMG:{source_doc_id}_{el.id}]]",
                        metadata=copy(el.metadata),
                    )
                    text_elements.append(placeholder)
                elif isinstance(el, Footer):
                    continue
                else:
                    if len(el.text.strip()) > 5:
                        text_elements.append(el)

            # copy of CompositeElement, with images like a tag
            if text_elements:
                chunk_copy: CompositeElement = copy(chunk)
                chunk_copy.metadata = copy(chunk.metadata)
                chunk_copy.metadata.orig_elements = text_elements
                result['texts'].append(chunk_copy)

    return result
