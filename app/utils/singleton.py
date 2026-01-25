class SingletonMeta(type):
    """Singleton metaclass."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            print('ChromaDB created object.')
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
