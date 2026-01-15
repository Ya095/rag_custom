
import chromadb
from chromadb import Collection
from chromadb.api import ClientAPI
from functools import lru_cache

from utils.download_file import APP_PATH


class ChromaWork:
    def __init__(self):
        self.db_path = APP_PATH / 'chroma.db'

    @lru_cache
    def init_db(self):
        client: ClientAPI = chromadb.PersistentClient(path=self.db_path)

        collection: Collection = client.get_or_create_collection(
            name='python_guide',
            metadata={"description": "Python programming guide"},
        )

        return collection
