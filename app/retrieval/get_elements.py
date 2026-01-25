from unstructured.documents.elements import Element, Table, Image


async def element_to_context_html(el: Element) -> str:
    """Convert element to string suitable for model context."""

    if isinstance(el, Table):
        html_context: str | None = getattr(el.metadata, 'text_as_html', None)
        if html_context:
            return el.metadata.text_as_html
        else:
            return ''

    elif isinstance(el, Image):
        # img_base64 = getattr(el.metadata, 'image_base64', None)
        # if img_base64:
        #     return f'<img src="data:image/png;base64,{img_base64}"/>'
        # else:
        return ''

    return getattr(el, 'text', str(el)).replace('\n\n', '\n')


async def build_context(chunks: list[Element]) -> str:
    parts: list[str] = [await element_to_context_html(el) for el in chunks]
    return '\n\n'.join(parts)
