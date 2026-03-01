import chromadb

_chroma_client = None
_collection = None


def init_chroma(app):
    global _chroma_client, _collection
    _chroma_client = chromadb.PersistentClient(path=app.config['CHROMA_DB_PATH'])
    _collection = _chroma_client.get_or_create_collection(
        name="nasa_reports",
        metadata={"hnsw:space": "cosine"},
    )


def get_collection():
    return _collection
