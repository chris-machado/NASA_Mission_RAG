import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-fallback-key')
    # Defensive cookie flags — inert today (no sessions) but correct if any are added.
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    # Rate-limit storage. "memory://" is correct for the single-worker serving
    # model; point at redis://... when scaling to multiple workers/hosts.
    RATELIMIT_STORAGE_URI = os.environ.get('RATELIMIT_STORAGE_URI', 'memory://')
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
    OLLAMA_CHAT_MODEL = os.environ.get('OLLAMA_CHAT_MODEL', 'gemma4:12b')
    OLLAMA_EMBED_MODEL = os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')
    OLLAMA_TIMEOUT = float(os.environ.get('OLLAMA_TIMEOUT', '120'))
    # Context window passed to the chat model. Safe for gemma4:12b and :26b;
    # 12b can go to 32k, 26b should stay <= 8k alongside flash-attn + q8 KV cache.
    OLLAMA_NUM_CTX = int(os.environ.get('OLLAMA_NUM_CTX', '8192'))
    # Gemma 4 "thinks" (chain-of-thought) before answering by default. Disable it
    # so grounded RAG answers stream immediately instead of after a reasoning pause.
    OLLAMA_THINK = os.environ.get('OLLAMA_THINK', 'false').lower() in ('1', 'true', 'yes', 'on')
    CHROMA_DB_PATH = os.environ.get(
        'CHROMA_DB_PATH',
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'chroma_db'),
    )
    RAG_TOP_K = int(os.environ.get('RAG_TOP_K', '6'))
    RAG_TEMPERATURE = float(os.environ.get('RAG_TEMPERATURE', '0.3'))
    # Max chunks kept from any single mission page, to keep retrieved context
    # diverse instead of 5 near-duplicate overlapping chunks from one page.
    RAG_MAX_PER_SOURCE = int(os.environ.get('RAG_MAX_PER_SOURCE', '2'))


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True


class TestingConfig(Config):
    TESTING = True
    DEBUG = True


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig,
}
