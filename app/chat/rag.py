import logging

import ollama
from flask import current_app

from app.extensions import get_collection
from app.chat.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def retrieve(query, n_results=None):
    """Embed query and retrieve top-k similar chunks from ChromaDB."""
    if n_results is None:
        n_results = current_app.config['RAG_TOP_K']

    collection = get_collection()

    query_embedding = ollama.embed(
        model=current_app.config['OLLAMA_EMBED_MODEL'],
        input=query,
    )['embeddings'][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances'],
    )

    return [
        {
            'text': doc,
            'metadata': meta,
            'distance': dist,
        }
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
        )
    ]


def generate_response(question):
    """Retrieve context and stream LLM response tokens."""
    chunks = retrieve(question)

    if not chunks:
        yield "I don't have any NASA mission report data loaded yet. Please check back later."
        return

    context = '\n\n'.join(
        f"[Source: {c['metadata'].get('mission', 'Unknown')} — {c['metadata'].get('source', 'Unknown')}, "
        f"p.{c['metadata'].get('page', '?')}]:\n{c['text']}"
        for c in chunks
    )

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': USER_PROMPT_TEMPLATE.format(
            context=context, question=question,
        )},
    ]

    stream = ollama.chat(
        model=current_app.config['OLLAMA_CHAT_MODEL'],
        messages=messages,
        stream=True,
        options={
            'temperature': current_app.config['RAG_TEMPERATURE'],
            'num_ctx': 4096,
        },
    )

    for chunk in stream:
        yield chunk['message']['content']
