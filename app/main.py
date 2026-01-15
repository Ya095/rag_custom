from collections import Counter
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown
from sentence_transformers import SentenceTransformer

from db.db import ChromaWork
from utils.download_file import file_path


md = MarkItDown()

result = md.convert(file_path)
content: str = result.text_content


# ############## Работа с документами ##############
processed_document: dict[str, str | Path] = {
    'source': file_path,
    'content': content
}

documents: list[dict[str, str | Path]] = [processed_document]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,
    separators=['\n\n', '\n', '. ', ' ', ''],
)


def process_document(
        doc: dict[str, str | Path],
        text_splitter_obj: RecursiveCharacterTextSplitter
) -> list[dict[str, str | Path]]:
    """Process a single document into chunks."""

    doc_chunks = text_splitter_obj.split_text(doc['content'])
    return [{'content': chunk, 'source': doc['source']} for chunk in doc_chunks]


all_chunks: list[dict[str, str | Path]] = []
for document in documents:
    chunks: list[dict[str, str | Path]] = process_document(document, text_splitter)
    all_chunks.extend(chunks)

# Результат разбиения
source_counts = Counter(chunk['source'] for chunk in all_chunks)
chunk_lengths = Counter(len(chunk['content']) for chunk in all_chunks)

print(f"Total chunks created: {len(all_chunks)}")
print(f"Chunk length: {min(chunk_lengths)}-{max(chunk_lengths)} characters")
print(f"Source document: {Path(documents[0]['source']).name}")


############## Создание поисковых эмбеддингов с SentenceTransformers ##############
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

documents_content = [chunk['content'] for chunk in all_chunks]
embeddings = model.encode(documents_content)

print(f"Embedding generation results:")
print(f"  - Embeddings shape: {embeddings.shape}")
print(f"  - Vector dimensions: {embeddings.shape[1]}")

############## Тест семантического сходства ##############
# query = "How do you define functions in Python?"
# document_chunks = [
#     "Variables store data values that can be used later in your program.",
#     "A function is a block of code that performs a specific task when called.",
#     "Loops allow you to repeat code multiple times efficiently.",
#     "Functions can accept parameters and return values to the calling code."
# ]
#
# query_embedding = model.encode(query)
# doc_embeddings = model.encode(document_chunks)
#
# similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
# ranked_results = sorted(
#     zip(document_chunks, similarities),
#     key=lambda x: x[1],
#     reverse=True,
# )
#
# print(f'Query: {query}')
# print("Document chunks ranked by relevance:")
# for i, (chunk, score) in enumerate(ranked_results, start=1):
#     print(f'{i}. ({score:.3f}): {chunk!r}')

############## Сохранение в ChromaDB ##############
chroma_collection = ChromaWork().init_db()

metadatas = [{"document": Path(chunk["source"]).name} for chunk in all_chunks]
chroma_collection.add(
    ids=[f'doc_{i}' for i in range(len(documents_content))],
    documents=documents_content,
    embeddings=embeddings,
    metadatas=metadatas,
)

print(f"Collection count: {chroma_collection.count()}")



