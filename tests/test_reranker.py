"""Tests for the cross-encoder reranking stage (model fully mocked)."""

import app.chat.reranker as rr


def test_score_empty_returns_empty():
    assert rr.score('q', [], 'model') == []


def test_score_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(rr, 'get_reranker', lambda *a, **k: None)
    assert rr.score('q', [{'text': 'a'}, {'text': 'b'}], 'model') == [None, None]


def test_score_aligns_to_input_order(monkeypatch):
    class FakeCrossEncoder:
        def predict(self, pairs):
            return [len(passage) for _, passage in pairs]

    monkeypatch.setattr(rr, 'get_reranker', lambda *a, **k: FakeCrossEncoder())
    assert rr.score('q', [{'text': 'a'}, {'text': 'bbb'}], 'model') == [1.0, 3.0]


def test_rerank_empty_returns_empty():
    assert rr.rerank('q', [], 'model') == []


def test_rerank_falls_back_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(rr, 'get_reranker', lambda *a, **k: None)
    cands = [{'text': 'a'}, {'text': 'b'}]
    # Unchanged order, returned as-is (a new list).
    assert [c['text'] for c in rr.rerank('q', cands, 'model')] == ['a', 'b']


def test_rerank_orders_by_score_desc(monkeypatch):
    class FakeCrossEncoder:
        def predict(self, pairs):
            # Score by passage length so 'bbb' > 'cc' > 'a'.
            return [len(passage) for _, passage in pairs]

    monkeypatch.setattr(rr, 'get_reranker', lambda *a, **k: FakeCrossEncoder())
    out = rr.rerank('q', [{'text': 'a'}, {'text': 'bbb'}, {'text': 'cc'}], 'model')
    assert [c['text'] for c in out] == ['bbb', 'cc', 'a']
    assert out[0]['rerank_score'] == 3.0


def test_rerank_predict_error_falls_back(monkeypatch):
    class BoomCrossEncoder:
        def predict(self, pairs):
            raise RuntimeError('boom')

    monkeypatch.setattr(rr, 'get_reranker', lambda *a, **k: BoomCrossEncoder())
    cands = [{'text': 'a'}, {'text': 'b'}]
    assert [c['text'] for c in rr.rerank('q', cands, 'model')] == ['a', 'b']
