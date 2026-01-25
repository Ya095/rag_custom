from unstructured.documents.elements import Element


def element_to_prompt_text(elem: Element) -> str:
    """Convert Element into clean text for LLM summarization."""

    # CompositeElement
    if hasattr(elem, 'metadata') and getattr(elem.metadata, 'orig_elements', None):
        parts = []
        for sub_elem in elem.metadata.orig_elements:
            if hasattr(sub_elem, 'text') and sub_elem.text:
                parts.append(sub_elem.text)
        return "\n".join(parts)

    # Plain text element
    if hasattr(elem, 'text') and elem.text:
        return elem.text

    return str(elem)


async def table_to_prompt_text(table: Element) -> str:
    """Get table html format for llm summaries."""

    html: str | None = getattr(table.metadata, 'text_as_html', None)
    if not html:
        return str(table)

    return f'HTML table:\n{html}'
