"""Shared pytest fixtures. Ollama and ChromaDB are mocked everywhere, so the
suite runs with no model server or vector DB present."""

import pytest

from app import create_app


class FakeListClient:
    """Stands in for ollama.Client in health checks."""

    def __init__(self, models):
        self._models = models

    def list(self):
        return {'models': [{'model': m} for m in self._models]}


class FakeCollection:
    def __init__(self, count=0):
        self._count = count

    def count(self):
        return self._count


@pytest.fixture
def app(monkeypatch):
    # Don't open a real ChromaDB on disk during app creation.
    import app.extensions as ext
    monkeypatch.setattr(ext, 'init_chroma', lambda application: None)
    return create_app('testing')


@pytest.fixture
def client(app):
    return app.test_client()
