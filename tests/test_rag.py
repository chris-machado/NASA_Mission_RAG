"""Tests for retrieval logic: keyword extraction, RRF fusion, diversity cap."""

import app.chat.rag as rag
from app.chat.rag import QUERY_PREFIX, _extract_keywords


class FakeCollection:
    """Returns canned vector hits; lexical (where_document) arms return nothing."""

    def __init__(self, hits):
        self.hits = hits  # list of (doc, meta, dist)

    def query(self, query_embeddings, n_results, include, where_document=None):
        if where_document is not None:
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        docs = [h[0] for h in self.hits][:n_results]
        metas = [h[1] for h in self.hits][:n_results]
        dists = [h[2] for h in self.hits][:n_results]
        return {'documents': [docs], 'metadatas': [metas], 'distances': [dists]}


class FakeEmbedClient:
    def __init__(self):
        self.last_input = None

    def embed(self, model, input):
        self.last_input = input
        return {'embeddings': [[0.1, 0.2, 0.3]]}


def test_extract_keywords_orders_proper_nouns_first():
    kws = _extract_keywords('What did Voyager discover at Jupiter?')
    assert kws[:2] == ['Voyager', 'Jupiter']     # proper nouns first
    assert 'discover' in kws                      # non-stopword kept


def test_extract_keywords_case_insensitive():
    upper = _extract_keywords('What did Voyager discover at Jupiter?')
    lower = _extract_keywords('what did voyager discover at jupiter')
    assert {w.lower() for w in upper} == {w.lower() for w in lower}


def test_extract_keywords_drops_stopwords():
    assert _extract_keywords('tell me about the nasa mission') == []


def test_query_uses_search_query_prefix(app, monkeypatch):
    embed_client = FakeEmbedClient()
    monkeypatch.setattr(rag, 'get_client', lambda: embed_client)
    monkeypatch.setattr(rag, 'get_collection', lambda: FakeCollection([
        ('doc', {'source': 'A', 'chunk_index': 0, 'mission': 'MA'}, 0.1),
    ]))
    with app.app_context():
        rag.retrieve('Apollo', n_results=1)
    assert embed_client.last_input == QUERY_PREFIX + 'Apollo'


def test_retrieve_caps_chunks_per_mission(app, monkeypatch):
    # 3 chunks from source A (best distances) then 1 from B.
    hits = [
        ('A0', {'source': 'A', 'chunk_index': 0, 'mission': 'MA'}, 0.10),
        ('A1', {'source': 'A', 'chunk_index': 1, 'mission': 'MA'}, 0.20),
        ('A2', {'source': 'A', 'chunk_index': 2, 'mission': 'MA'}, 0.30),
        ('B0', {'source': 'B', 'chunk_index': 0, 'mission': 'MB'}, 0.40),
    ]
    monkeypatch.setattr(rag, 'get_client', lambda: FakeEmbedClient())
    monkeypatch.setattr(rag, 'get_collection', lambda: FakeCollection(hits))
    with app.app_context():
        results = rag.retrieve('Voyager Jupiter', n_results=3)

    texts = [r['text'] for r in results]
    # RAG_MAX_PER_SOURCE defaults to 2, so A2 is dropped and B0 surfaces.
    assert texts == ['A0', 'A1', 'B0']
    assert 'A2' not in texts


def test_retrieve_empty_collection_returns_empty(app, monkeypatch):
    monkeypatch.setattr(rag, 'get_client', lambda: FakeEmbedClient())
    monkeypatch.setattr(rag, 'get_collection', lambda: FakeCollection([]))
    with app.app_context():
        assert rag.retrieve('anything', n_results=5) == []


def test_build_sources_dedupes_by_url():
    chunks = [
        {'text': 'x', 'metadata': {'source': 'urlA', 'mission': 'A'}},
        {'text': 'y', 'metadata': {'source': 'urlA', 'mission': 'A'}},
        {'text': 'z', 'metadata': {'source': 'urlB', 'mission': 'B'}},
    ]
    sources = rag._build_sources(chunks)
    assert [s['url'] for s in sources] == ['urlA', 'urlB']
