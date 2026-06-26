"""Cross-encoder reranking stage.

Dense retrieval (and RRF over keyword arms) is good at *recall* but mediocre at
*precision* of the final ordering: it scores query/passage similarity in a
shared embedding space, so it can't tell that a vividly-worded but off-topic
passage is worse than a drier on-topic one. A cross-encoder reads the (query,
passage) pair jointly and scores true relevance — much sharper, at the cost of
a forward pass per candidate. We can afford that here: a dedicated RTX 4090 and
a tolerance for ~1s of extra latency in exchange for better answers.

The model is loaded once (lazily) and kept resident. If sentence-transformers or
the weights are unavailable, rerank() degrades gracefully to the input order so
retrieval still works.
"""
import logging
import threading

logger = logging.getLogger(__name__)

_model = None
_load_failed = False
_lock = threading.Lock()


def _resolve_device(device):
    if device and device != 'auto':
        return device
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def get_reranker(model_name, device='auto'):
    """Return a loaded CrossEncoder, or None if it can't be loaded (cached)."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    with _lock:
        if _model is not None or _load_failed:
            return _model
        try:
            from sentence_transformers import CrossEncoder
            resolved = _resolve_device(device)
            _model = CrossEncoder(model_name, device=resolved, max_length=512)
            if resolved == 'cuda':
                # Half precision ~halves VRAM (so it co-resides with gemma4:26b on
                # a 24 GB card) and is faster on tensor cores; ranking is unaffected.
                try:
                    _model.model.half()
                except Exception:
                    logger.warning('fp16 cast failed; keeping fp32 reranker')
            logger.info('Loaded reranker %s on %s', model_name, resolved)
        except Exception:
            logger.exception('Could not load reranker %s; using fusion order', model_name)
            _load_failed = True
    return _model


def warm_up(model_name, device='auto'):
    """Eagerly load the model (e.g. at server start) so the first query is fast."""
    get_reranker(model_name, device)


def score(query, candidates, model_name, device='auto'):
    """Return a relevance score per candidate, aligned to input order.

    Each candidate is a dict with a 'text' key. Returns ``[None, ...]`` if the
    reranker is unavailable or errors, so callers can fall back to fusion order.
    """
    if not candidates:
        return []
    model = get_reranker(model_name, device)
    if model is None:
        return [None] * len(candidates)
    try:
        raw = model.predict([(query, c['text']) for c in candidates])
        return [float(s) for s in raw]
    except Exception:
        logger.exception('Reranker.predict failed; using fusion order')
        return [None] * len(candidates)


def rerank(query, candidates, model_name, device='auto'):
    """Reorder candidate chunk dicts by cross-encoder relevance to ``query``.

    Returns a new list ordered best first, annotated with 'rerank_score'. Falls
    back to the input order if the reranker is unavailable or errors. (retrieve()
    uses score() directly so it can *fuse* rerank with the first-stage ranking;
    this helper is kept for simple standalone reranking.)
    """
    if not candidates:
        return []
    scores = score(query, candidates, model_name, device)
    if all(s is None for s in scores):
        return list(candidates)
    order = sorted(range(len(candidates)),
                   key=lambda i: (scores[i] is not None, scores[i] or 0.0),
                   reverse=True)
    ranked = []
    for i in order:
        c = dict(candidates[i])
        c['rerank_score'] = scores[i]
        ranked.append(c)
    return ranked
