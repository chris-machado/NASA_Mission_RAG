import json
import logging

from flask import (Blueprint, Response, current_app, render_template, request,
                   stream_with_context)

from app.chat.ollama_client import get_client
from app.chat.rag import generate_response
from app.extensions import get_collection

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@chat_bp.route('/api/chat', methods=['POST'])
def chat():
    """Streaming chat endpoint using Server-Sent Events."""
    data = request.get_json(silent=True)
    if not data or not data.get('question'):
        return {'error': 'No question provided'}, 400

    question = data['question'].strip()
    if not question:
        return {'error': 'No question provided'}, 400
    if len(question) > 500:
        return {'error': 'Question too long (max 500 characters)'}, 400

    # Prior conversation turns for follow-up context; generate_response validates
    # and bounds the shape, so a malformed history degrades to a stateless answer.
    history = data.get('history')

    # Retrieval + client setup happen before streaming, so a down Ollama or empty
    # corpus surfaces as a clean error instead of a half-open SSE stream.
    try:
        token_gen, sources = generate_response(question, history)
    except Exception as e:
        logger.error('RAG init error: %s', e)
        return {'error': 'The assistant is temporarily unavailable. Please try again.'}, 503

    def event_stream():
        try:
            for token in token_gen:
                yield f"data: {json.dumps({'token': token})}\n\n"
            if sources:
                yield f"data: {json.dumps({'sources': sources})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.error('RAG pipeline error: %s', e)
            yield f"data: {json.dumps({'error': 'An error occurred. Please try again.'})}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


def _model_present(want, have):
    """True if model spec ``want`` is satisfied by the set of installed ``have``,
    treating a bare name as equivalent to its ":latest" tag."""
    if want in have:
        return True
    if ':' not in want and f'{want}:latest' in have:
        return True
    if want.endswith(':latest') and want[: -len(':latest')] in have:
        return True
    return False


def _installed_model_names(client):
    """Return the set of installed model names, tolerant of ollama-python's
    dict-vs-object response shapes across versions."""
    listed = client.list()
    models = getattr(listed, 'models', None)
    if models is None and isinstance(listed, dict):
        models = listed.get('models', [])
    names = set()
    for m in models or []:
        name = getattr(m, 'model', None)
        if name is None and isinstance(m, dict):
            name = m.get('model') or m.get('name')
        if name:
            names.add(name)
    return names


@chat_bp.route('/api/health', methods=['GET'])
def health():
    """Deep health check: Ollama reachable, required models pulled, corpus loaded."""
    components = {}
    ok = True

    chat_model = current_app.config['OLLAMA_CHAT_MODEL']
    embed_model = current_app.config['OLLAMA_EMBED_MODEL']
    try:
        names = _installed_model_names(get_client())
        components['ollama'] = 'connected'
        components['chat_model'] = 'ready' if _model_present(chat_model, names) else 'missing'
        components['embed_model'] = 'ready' if _model_present(embed_model, names) else 'missing'
        if 'missing' in (components['chat_model'], components['embed_model']):
            ok = False
    except Exception:
        components.update(ollama='disconnected', chat_model='unknown', embed_model='unknown')
        ok = False

    try:
        count = get_collection().count()
        components['corpus_chunks'] = count
        if not count:
            ok = False
    except Exception:
        components['corpus_chunks'] = 'error'
        ok = False

    status = {'status': 'ok' if ok else 'degraded', 'components': components}
    return (status, 200) if ok else (status, 503)
