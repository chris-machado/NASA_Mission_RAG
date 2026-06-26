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
    OLLAMA_CHAT_MODEL = os.environ.get('OLLAMA_CHAT_MODEL', 'gemma4:26b')
    OLLAMA_EMBED_MODEL = os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')
    OLLAMA_TIMEOUT = float(os.environ.get('OLLAMA_TIMEOUT', '120'))
    # Context window passed to the chat model. 4096 keeps gemma4:26b's KV cache
    # small enough to co-reside with the GPU reranker on a 24 GB card; this RAG
    # only feeds ~1.7k tokens of context. Raise if you free GPU memory.
    OLLAMA_NUM_CTX = int(os.environ.get('OLLAMA_NUM_CTX', '4096'))
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
    # Retrieve-then-rerank: pull a deep candidate pool, then a cross-encoder
    # reorders it by true relevance before we keep RAG_TOP_K. The 4090 has
    # headroom for this and we trade ~1s of latency for sharper context.
    RAG_CANDIDATE_K = int(os.environ.get('RAG_CANDIDATE_K', '40'))
    RAG_RERANK_ENABLED = os.environ.get(
        'RAG_RERANK_ENABLED', 'true').lower() in ('1', 'true', 'yes', 'on')
    # bge-reranker-v2-m3 grounds the query entity far better than -base here
    # (-base drifted to other missions' failure stories for "what went wrong on
    # Apollo 13"); the 4090 has room for it alongside the 26B chat model.
    RAG_RERANK_MODEL = os.environ.get('RAG_RERANK_MODEL', 'BAAI/bge-reranker-v2-m3')
    RAG_RERANK_DEVICE = os.environ.get('RAG_RERANK_DEVICE', 'auto')  # auto|cuda|cpu
    # Weight of the cross-encoder rank when fused (RRF) with the first-stage rank.
    # <1 keeps first-stage retrieval a strong co-signal so a confident-but-wrong
    # reranker can't bury good lexical/vector hits (entity drift); 0 disables fusion.
    RAG_RERANK_WEIGHT = float(os.environ.get('RAG_RERANK_WEIGHT', '0.5'))
    # Per-chunk relevance gate for citing sources. Vector search always returns
    # nearest chunks, so off-topic questions cite bogus pages and focused ones
    # cite weak/related leftovers. A citable chunk must be BOTH cross-encoder
    # relevant (>= MIN_RERANK) AND a close vector match (<= MAX_DISTANCE): the
    # rerank score alone can't tell the subject from a related mission (Gemini XII
    # scores 0.95 for "Apollo 13" via a shared astronaut) but that page is a far
    # vector match, so distance excludes it. Temporal queries cite by year
    # instead — the reranker is fooled by year mentions.
    RAG_RELEVANCE_MIN_RERANK = float(os.environ.get('RAG_RELEVANCE_MIN_RERANK', '0.4'))
    RAG_RELEVANCE_MAX_DISTANCE = float(os.environ.get('RAG_RELEVANCE_MAX_DISTANCE', '0.31'))
    # Load the reranker at server start (vs. lazily on first query). Set in the
    # service env so the first user query isn't slowed by a model load.
    RAG_RERANK_PRELOAD = os.environ.get(
        'RAG_RERANK_PRELOAD', 'false').lower() in ('1', 'true', 'yes', 'on')


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    # Keep tests free of the heavy cross-encoder dependency/model load; rerank
    # logic is exercised separately with a mocked model.
    RAG_RERANK_ENABLED = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig,
}
