from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element

from config import APP_PATH


file_name = 'attention.pdf'
doc_folder_path = APP_PATH / 'documents'
file_name_path = doc_folder_path / file_name


chunks: list[Element] = partition_pdf(
    filename=file_name_path.absolute().as_posix(),
    languages=['eng'],
    infer_table_structure=True,  # extract tables
    strategy="hi_res",  # mandatory to infer tables
    extract_image_block_types=["Image"],  # Add 'Table' to list to extract image of tables
    extract_image_block_to_payload=True,  # if true, will extract base64 for API usage
    chunking_strategy="by_title",  # or 'basic'
    max_characters=7000,  # defaults to 500
    combine_text_under_n_chars=1500,
    new_after_n_chars=5000,
    overlap=800,
)


# for el in chunks:
#     print(el.metadata.orig_elements)
#     print('\n\n')

# elements = chunks[2].metadata.orig_elements
# image_chunks = [el for el in elements if 'Image' in str(type(el))]
# print(image_chunks[0].to_dict())
