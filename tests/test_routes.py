"""Tests for HTTP endpoints: validation, SSE streaming, deep health check."""

import app.chat.routes as routes
from tests.conftest import FakeCollection, FakeListClient


def test_chat_requires_question(client):
    assert client.post('/api/chat', json={}).status_code == 400
    assert client.post('/api/chat', json={'question': '   '}).status_code == 400


def test_chat_rejects_overlong_question(client):
    assert client.post('/api/chat', json={'question': 'x' * 501}).status_code == 400


def test_chat_streams_tokens_and_sources(client, monkeypatch):
    def fake_generate(question, history=None):
        def gen():
            yield 'Hello '
            yield 'world'
        return gen(), [{'title': 'Apollo', 'url': 'http://x', 'mission': 'Apollo'}]

    monkeypatch.setattr(routes, 'generate_response', fake_generate)
    resp = client.post('/api/chat', json={'question': 'hi'})
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'data: ' in body
    assert 'Hello ' in body and 'world' in body
    assert '"sources"' in body
    assert '"done": true' in body


def test_chat_forwards_history_to_backend(client, monkeypatch):
    seen = {}

    def fake_generate(question, history=None):
        seen['question'] = question
        seen['history'] = history
        return iter(['ok']), []

    monkeypatch.setattr(routes, 'generate_response', fake_generate)
    hist = [{'role': 'user', 'content': 'Tell me about Voyager'},
            {'role': 'assistant', 'content': 'Voyager 1 and 2 ...'}]
    resp = client.post('/api/chat',
                       json={'question': 'what did it find at Saturn?', 'history': hist})
    assert resp.status_code == 200
    assert seen['question'] == 'what did it find at Saturn?'
    assert seen['history'] == hist


def test_chat_handles_backend_failure(client, monkeypatch):
    def boom(question, history=None):
        raise RuntimeError('ollama down')

    monkeypatch.setattr(routes, 'generate_response', boom)
    resp = client.post('/api/chat', json={'question': 'hi'})
    assert resp.status_code == 503


def test_health_all_green(client, monkeypatch):
    chat = client.application.config['OLLAMA_CHAT_MODEL']
    monkeypatch.setattr(routes, 'get_client',
                        lambda: FakeListClient([chat, 'nomic-embed-text:latest']))
    monkeypatch.setattr(routes, 'get_collection', lambda: FakeCollection(count=42))
    resp = client.get('/api/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'ok'
    assert data['components']['chat_model'] == 'ready'
    assert data['components']['embed_model'] == 'ready'
    assert data['components']['corpus_chunks'] == 42


def test_health_missing_chat_model_is_degraded(client, monkeypatch):
    monkeypatch.setattr(routes, 'get_client',
                        lambda: FakeListClient(['nomic-embed-text:latest']))
    monkeypatch.setattr(routes, 'get_collection', lambda: FakeCollection(count=42))
    resp = client.get('/api/health')
    assert resp.status_code == 503
    assert resp.get_json()['components']['chat_model'] == 'missing'


def test_health_empty_corpus_is_degraded(client, monkeypatch):
    chat = client.application.config['OLLAMA_CHAT_MODEL']
    monkeypatch.setattr(routes, 'get_client',
                        lambda: FakeListClient([chat, 'nomic-embed-text:latest']))
    monkeypatch.setattr(routes, 'get_collection', lambda: FakeCollection(count=0))
    resp = client.get('/api/health')
    assert resp.status_code == 503
    assert resp.get_json()['components']['corpus_chunks'] == 0


def test_health_nomic_bare_name_matches_latest_tag(client, monkeypatch):
    # embed model configured as bare "nomic-embed-text" must match the
    # installed "nomic-embed-text:latest".
    chat = client.application.config['OLLAMA_CHAT_MODEL']
    monkeypatch.setattr(routes, 'get_client',
                        lambda: FakeListClient([chat, 'nomic-embed-text:latest']))
    monkeypatch.setattr(routes, 'get_collection', lambda: FakeCollection(count=1))
    assert client.get('/api/health').get_json()['components']['embed_model'] == 'ready'
