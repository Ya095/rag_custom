from unstructured.documents.elements import Element, Table, Image, CompositeElement


async def element_to_context_html(el: Element) -> str:
    """Convert element to string suitable for model context."""

    if isinstance(el, Table):
        if getattr(el.metadata, 'text_as_html', None):
            return el.metadata.text_as_html
        else:
            return ''

    elif isinstance(el, Image):
        return f'[[IMG:{el.metadata.img_uid}]]'

    if isinstance(el, CompositeElement):
        parts: list[str] = []

        for sub_el in el.metadata.orig_elements or []:
            if isinstance(sub_el, Table):
                parts.append(sub_el.metadata.text_as_html)
                continue

            if sub_el.text:
                parts.append(sub_el.text)

        return '\n'.join(parts)

    return getattr(el, 'text', '')


async def build_context(chunks: list[Element]) -> str:
    """Building context from chunks for llm model like string."""

    parts: list[str] = [await element_to_context_html(el) for el in chunks]
    return '\n\n'.join(parts)
