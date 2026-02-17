from pathlib import Path


APP_PATH: Path = Path(__file__).parent

# llm module
DEVICE: str = 'cpu'
TEXT_MODEL: str = 'llama3.2:3b'
IMAGE_DESCRIPTION_MODEL: str = 'llava:7b'

# Chroma module
EMBEDDING_MODEL_NAME: str = 'multi-qa-mpnet-base-dot-v1'
COLLECTION_NAME: str = 'multi_modal_rag'
COLLECTION_METADATA: dict = {
    'description': 'Multimodal RAG: text, tables and images',
    'hnsw:space': 'ip',
}
RETRIEVER_SEARCH_KWARGS: dict = {
    'k': 5,
    'fetch_k': 20,
    'lambda_mult': 0.7,
}
