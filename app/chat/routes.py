import json
import logging

from flask import Blueprint, request, render_template, Response, stream_with_context

from app.chat.rag import generate_response

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@chat_bp.route('/api/chat', methods=['POST'])
def chat():
    """Streaming chat endpoint using Server-Sent Events."""
    data = request.get_json()
    if not data or not data.get('question'):
        return {'error': 'No question provided'}, 400

    question = data['question'].strip()
    if len(question) > 500:
        return {'error': 'Question too long (max 500 characters)'}, 400

    token_gen, sources = generate_response(question)

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


@chat_bp.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        import ollama
        ollama.list()
        return {'status': 'ok', 'ollama': 'connected'}
    except Exception:
        return {'status': 'degraded', 'ollama': 'disconnected'}, 503
