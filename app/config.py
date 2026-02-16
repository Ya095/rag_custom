from pathlib import Path


APP_PATH: Path = Path(__file__).parent

# llm module
DEVICE: str = 'cpu'
TEXT_MODEL: str = 'llama3.2:3b'
IMAGE_DESCRIPTION_MODEL: str = 'llava:7b'

# Chroma module
EMBEDDING_MODEL_NAME: str = 'multi-qa-mpnet-base-dot-v1'
COLLECTION_NAME: str = 'multi_modal_rag'
COLLECTION_METADATA_DESCRIPTION: str = 'Multimodal RAG: text, tables and images'
RETRIEVER_SEARCH_KWARGS: dict = {'k': 3}
