import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-fallback-key')
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
    OLLAMA_CHAT_MODEL = os.environ.get('OLLAMA_CHAT_MODEL', 'llama3.2:3b')
    OLLAMA_EMBED_MODEL = os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')
    CHROMA_DB_PATH = os.environ.get(
        'CHROMA_DB_PATH',
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'chroma_db'),
    )
    RAG_TOP_K = int(os.environ.get('RAG_TOP_K', '5'))
    RAG_TEMPERATURE = float(os.environ.get('RAG_TEMPERATURE', '0.3'))


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    TESTING = True
    DEBUG = True


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig,
}
