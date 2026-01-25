from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element

from core.config import APP_PATH


file_name = 'attention.pdf'
file_name_path = APP_PATH / 'documents' / file_name


def parse_input_document() -> list[Element]:
    """Parse intput document from docs."""

    chunks: list[Element] = partition_pdf(
        filename=file_name_path.absolute().as_posix(),
        languages=['eng'],
        infer_table_structure=True,  # extract tables
        strategy="hi_res",  # mandatory to infer tables
        extract_image_block_types=["Image"],  # Add 'Table' to list to extract image of tables
        extract_image_block_to_payload=True,  # if true, will extract base64 for API usage
        chunking_strategy="by_title",
        max_characters=5000,
        combine_text_under_n_chars=1500,
        new_after_n_chars=3000,
    )

    return chunks
