"""Tests for retrieval logic: keyword extraction, RRF fusion, diversity cap,
temporal gating."""

from datetime import date

import app.chat.rag as rag
from app.chat.rag import QUERY_PREFIX, _extract_keywords, _temporal_targets


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


def test_citable_chunks_requires_rerank_and_proximity(app):
    chunks = [
        {'metadata': {'mission': 'Apollo 13', 'year': 1970}, 'rerank_score': 0.93, 'distance': 0.23},
        {'metadata': {'mission': 'STS-36', 'year': 1990}, 'rerank_score': 0.28, 'distance': 0.30},   # weak rerank
        {'metadata': {'mission': 'Gemini XII', 'year': 1966}, 'rerank_score': 0.95, 'distance': 0.34},  # related but vector-far
    ]
    with app.app_context():
        cited = rag._citable_chunks('What went wrong on Apollo 13?', chunks)
    # Gemini XII has the highest rerank but is dropped on distance; STS-36 on rerank.
    assert [c['metadata']['mission'] for c in cited] == ['Apollo 13']


def test_citable_chunks_off_topic_returns_empty(app):
    chunks = [{'metadata': {'mission': 'Lucy', 'year': 2021}, 'rerank_score': 0.06, 'distance': 0.44}]
    with app.app_context():
        assert rag._citable_chunks('what is your name?', chunks) == []


def test_citable_chunks_keeps_top_when_relevant_but_vector_distant(app):
    # Rerank-strong but vector-distant best result still yields a citation.
    chunks = [{'metadata': {'mission': 'Deep Cut', 'year': 1980}, 'rerank_score': 0.85, 'distance': 0.38}]
    with app.app_context():
        cited = rag._citable_chunks('tell me about the deep cut mission', chunks)
    assert [c['metadata']['mission'] for c in cited] == ['Deep Cut']


def test_citable_chunks_temporal_uses_year_not_rerank(app):
    # The reranker is fooled by year mentions, so temporal queries cite by year:
    # a real 2026 mission with a low rerank score beats a high-scoring off-year page.
    chunks = [
        {'metadata': {'mission': 'NISAR', 'year': 2026}, 'rerank_score': 0.27, 'distance': 0.36},
        {'metadata': {'mission': 'Old Probe', 'year': 1999}, 'rerank_score': 0.81, 'distance': 0.30},
    ]
    with app.app_context():
        cited = rag._citable_chunks('latest missions in 2026', chunks)
    missions = [c['metadata']['mission'] for c in cited]
    assert 'NISAR' in missions and 'Old Probe' not in missions


def test_citable_chunks_distance_fallback_when_no_reranker(app):
    chunks = [
        {'metadata': {'mission': 'A', 'year': 1980}, 'distance': 0.20},
        {'metadata': {'mission': 'B', 'year': 1980}, 'distance': 0.50},
    ]
    with app.app_context():
        cited = rag._citable_chunks('tell me about mission A', chunks)
    assert [c['metadata']['mission'] for c in cited] == ['A']


def test_build_sources_dedupes_by_url():
    chunks = [
        {'text': 'x', 'metadata': {'source': 'urlA', 'mission': 'A'}},
        {'text': 'y', 'metadata': {'source': 'urlA', 'mission': 'A'}},
        {'text': 'z', 'metadata': {'source': 'urlB', 'mission': 'B'}},
    ]
    sources = rag._build_sources(chunks)
    assert [s['url'] for s in sources] == ['urlA', 'urlB']


TODAY = date(2026, 6, 26)


def test_temporal_targets_explicit_year():
    window, tokens, primary, recency = _temporal_targets('missions in 2026', today=TODAY)
    assert window == {2025, 2026, 2027}
    assert tokens == {'2026'}
    assert primary == {2026}
    assert recency is False


def test_temporal_targets_recency_words():
    window, tokens, primary, recency = _temporal_targets('the latest missions', today=TODAY)
    assert recency is True
    assert window == {2025, 2026, 2027}
    assert primary == window


def test_temporal_targets_topical_is_none():
    assert _temporal_targets('What did Voyager discover at Jupiter?',
                             today=TODAY) == (None, None, None, False)


def test_temporal_targets_ignores_ambiguous_cues():
    # "current" (cf. ocean current) and "new"/"now" must NOT trigger temporal mode.
    assert _temporal_targets('the current ocean mission', today=TODAY)[3] is False
    assert _temporal_targets('Tell me about New Horizons',
                             today=TODAY) == (None, None, None, False)


class TemporalFakeCollection:
    """Honors where={'year': {'$in': ...}} and where_document={'$contains': ...}."""

    def __init__(self, hits):
        self.hits = hits  # (doc, meta, dist)

    def query(self, query_embeddings, n_results, include, where=None, where_document=None):
        items = self.hits
        if where and 'year' in where:
            allowed = set(where['year']['$in'])
            items = [h for h in items if h[1].get('year') in allowed]
        if where_document and '$contains' in where_document:
            token = where_document['$contains']
            items = [h for h in items if token in h[0]]
        items = items[:n_results]
        return {'documents': [[h[0] for h in items]],
                'metadatas': [[h[1] for h in items]],
                'distances': [[h[2] for h in items]]}


def test_fusion_keeps_strong_retrieval_hit_over_confident_reranker(app, monkeypatch):
    # First-stage ranks MA top; a confident-but-wrong reranker maxes out MB. RRF
    # fusion (weight 0.5) must keep MA first rather than let the reranker bury it.
    hits = [
        ('strong first-stage hit', {'source': 'A', 'chunk_index': 0, 'mission': 'MA'}, 0.10),
        ('reranker favourite', {'source': 'B', 'chunk_index': 0, 'mission': 'MB'}, 0.40),
    ]
    monkeypatch.setattr(rag, 'get_client', lambda: FakeEmbedClient())
    monkeypatch.setattr(rag, 'get_collection', lambda: FakeCollection(hits))
    monkeypatch.setattr(rag.reranker, 'score',
                        lambda q, cands, model, device='auto':
                        [1.0 if c['metadata']['mission'] == 'MB' else 0.0 for c in cands])
    app.config['RAG_RERANK_ENABLED'] = True
    app.config['RAG_RERANK_WEIGHT'] = 0.5
    with app.app_context():
        results = rag.retrieve('topical query', n_results=2)
    assert results[0]['metadata']['mission'] == 'MA'


def test_retrieve_temporal_gates_out_old_missions(app, monkeypatch):
    # Apollo 13 has the *better* vector distance, but a "2026" query must gate it
    # out (wrong year, text doesn't mention 2026) and surface Artemis II.
    hits = [
        ('Apollo 13 launched in 1970',
         {'source': 'apo', 'chunk_index': 0, 'mission': 'Apollo 13', 'year': 1970}, 0.10),
        ('Artemis II will launch in 2026',
         {'source': 'art', 'chunk_index': 0, 'mission': 'Artemis II', 'year': 2026}, 0.50),
    ]
    monkeypatch.setattr(rag, 'get_client', lambda: FakeEmbedClient())
    monkeypatch.setattr(rag, 'get_collection', lambda: TemporalFakeCollection(hits))
    with app.app_context():
        results = rag.retrieve('latest NASA missions in 2026', n_results=5)
    missions = [r['metadata']['mission'] for r in results]
    assert 'Artemis II' in missions
    assert 'Apollo 13' not in missions


# --- Conversational memory: history normalization + query contextualization ----

class FakeChatClient:
    """Stands in for the Ollama client: non-streamed chat (condense) returns a
    dict; streamed chat (generation) yields message chunks. Records the last call."""

    def __init__(self, reply='', capture=None):
        self._reply = reply
        self.capture = capture

    def chat(self, model, messages, stream=False, think=None, options=None):
        if self.capture is not None:
            self.capture.update(model=model, messages=messages, options=options)
        if stream:
            def gen():
                yield {'message': {'content': self._reply}}
            return gen()
        return {'message': {'content': self._reply}}


def test_normalize_history_drops_malformed_and_caps(app):
    raw = [
        'not a dict',
        {'role': 'system', 'content': 'x'},        # bad role
        {'role': 'user', 'content': '   '},         # empty after strip
        {'role': 'user', 'content': 'Q1'},
        {'role': 'assistant', 'content': 'A1'},
        {'role': 'user', 'content': 'Q2'},
        {'role': 'assistant', 'content': 'A2'},
    ]
    with app.app_context():
        app.config['RAG_HISTORY_MAX_TURNS'] = 2
        out = rag._normalize_history(raw)
    assert out == [{'role': 'user', 'content': 'Q2'},
                   {'role': 'assistant', 'content': 'A2'}]


def test_normalize_history_trims_leading_assistant(app):
    # A history that opens on an assistant turn (truncated mid-exchange) is trimmed
    # so the replayed messages start on a user turn.
    raw = [{'role': 'assistant', 'content': 'A0'},
           {'role': 'user', 'content': 'Q1'},
           {'role': 'assistant', 'content': 'A1'}]
    with app.app_context():
        out = rag._normalize_history(raw)
    assert [t['role'] for t in out] == ['user', 'assistant']
    assert out[0]['content'] == 'Q1'


def test_normalize_history_truncates_long_turns(app):
    with app.app_context():
        app.config['RAG_HISTORY_MAX_CHARS'] = 100
        out = rag._normalize_history([{'role': 'user', 'content': 'x' * 5000}])
    assert len(out[0]['content']) == 100


def test_normalize_history_non_list_returns_empty(app):
    with app.app_context():
        assert rag._normalize_history(None) == []
        assert rag._normalize_history('nope') == []


def test_contextualize_query_no_history_skips_llm(app, monkeypatch):
    def boom():
        raise AssertionError('no client should be built without history')
    monkeypatch.setattr(rag, 'get_client', boom)
    with app.app_context():
        assert rag._contextualize_query('hello', []) == 'hello'


def test_contextualize_query_rewrites_with_history(app, monkeypatch):
    monkeypatch.setattr(rag, 'get_client',
                        lambda: FakeChatClient(reply='What did Voyager discover at Saturn?'))
    hist = [{'role': 'user', 'content': 'Tell me about Voyager'},
            {'role': 'assistant', 'content': 'Voyager 1 and 2 flew by the outer planets.'}]
    with app.app_context():
        out = rag._contextualize_query('what about Saturn?', hist)
    assert out == 'What did Voyager discover at Saturn?'


def test_contextualize_query_strips_echoed_label(app, monkeypatch):
    monkeypatch.setattr(rag, 'get_client',
                        lambda: FakeChatClient(reply='Standalone question: What did Voyager find at Saturn?'))
    hist = [{'role': 'user', 'content': 'Tell me about Voyager'}]
    with app.app_context():
        out = rag._contextualize_query('what about Saturn?', hist)
    assert out == 'What did Voyager find at Saturn?'


def test_contextualize_query_falls_back_on_error(app, monkeypatch):
    class BoomClient:
        def chat(self, *a, **k):
            raise RuntimeError('ollama down')
    monkeypatch.setattr(rag, 'get_client', lambda: BoomClient())
    hist = [{'role': 'user', 'content': 'Tell me about Voyager'}]
    with app.app_context():
        assert rag._contextualize_query('what about Saturn?', hist) == 'what about Saturn?'


def test_contextualize_query_disabled_returns_raw(app, monkeypatch):
    monkeypatch.setattr(rag, 'get_client',
                        lambda: FakeChatClient(reply='REWRITTEN'))
    hist = [{'role': 'user', 'content': 'Tell me about Voyager'}]
    with app.app_context():
        app.config['RAG_CONTEXTUALIZE'] = False
        assert rag._contextualize_query('what about Saturn?', hist) == 'what about Saturn?'


def test_generate_response_threads_history_into_messages(app, monkeypatch):
    cap = {}
    monkeypatch.setattr(rag, '_contextualize_query', lambda q, h: 'STANDALONE QUERY')
    seen = {}

    def fake_retrieve(query, n_results=None):
        seen['query'] = query
        return [{'text': 'ctx', 'rerank_score': 0.9, 'distance': 0.2,
                 'metadata': {'mission': 'Voyager 1', 'source': 'u', 'year': 1977}}]

    monkeypatch.setattr(rag, 'retrieve', fake_retrieve)
    monkeypatch.setattr(rag, 'get_client', lambda: FakeChatClient(reply='answer', capture=cap))
    hist = [{'role': 'user', 'content': 'Tell me about Voyager'},
            {'role': 'assistant', 'content': 'Voyager 1 and 2 ...'}]
    with app.app_context():
        gen, _sources = rag.generate_response('what about Saturn?', hist)
        list(gen)  # drain the stream so the chat call fires

    # Retrieval ran on the contextualized (standalone) query, not the raw follow-up.
    assert seen['query'] == 'STANDALONE QUERY'
    # History is replayed between the system prompt and the final user turn.
    assert [m['role'] for m in cap['messages']] == ['system', 'user', 'assistant', 'user']
    # Only the final user turn carries retrieved context + the original question.
    assert 'ctx' in cap['messages'][-1]['content']
    assert 'what about Saturn?' in cap['messages'][-1]['content']
