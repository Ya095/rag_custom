class RAGBaseException(Exception):
    pass


class DatabaseNotInitializedError(RAGBaseException):
    pass


class RetrievalError(RAGBaseException):
    pass


class LLMError(RAGBaseException):
    pass


class ImageProcessingError(RAGBaseException):
    pass


class DocumentProcessingError(RAGBaseException):
    pass
