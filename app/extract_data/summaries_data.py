from unstructured.documents.elements import Element, CompositeElement, Table, Image


async def print_element(el: Element, idx: int) -> None:
    print(f"\nChunk {idx}")
    print("Category:", el.category)
    print("Page:", el.metadata.page_number)

    # Composite text block
    if isinstance(el, CompositeElement):
        print("Composite elements:")
        for sub in el.metadata.orig_elements or []:
            print(sub.to_dict())
            # print(" -", sub.category, sub.metadata.page_number)
            # if hasattr(sub, "text") and sub.text:
            #     print(sub.text)
            # elif hasattr(sub.metadata, "text_as_html"):
            #     print(sub.metadata.text_as_html)

    # Table
    elif isinstance(el, Table):
        print("Table:")
        if el.metadata.text_as_html:
            # print(el.metadata.text_as_html)
            print(el.to_dict())
        else:
            print("[Table without HTML representation]")

    # Image
    elif isinstance(el, Image):
        print("Image:")
        print(el.to_dict())
        # print(" - format:", el.metadata.image_mime_type)
        # print(" - page:", el.metadata.page_number)

    # Plain text element
    elif hasattr(el, "text") and el.text:
        print("Text:")
        # print(el.text[:500], '...')
        el.to_dict()

    else:
        print("[Unknown or empty element]")
