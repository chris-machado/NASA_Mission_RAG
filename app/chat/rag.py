import logging
import re
from collections import OrderedDict
from datetime import date

import ollama
from flask import current_app

from app.extensions import get_collection
from app.chat.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def _extract_keywords(query):
    """Extract meaningful keywords from a query for keyword-boosted search.

    Prioritizes proper nouns (capitalized words like mission names) over generic
    descriptive words, which avoids boosting irrelevant documents that happen to
    contain common terms like "goals" or "history".
    """
    stop_words = {
        'tell', 'me', 'about', 'what', 'is', 'the', 'a', 'an', 'of', 'and',
        'or', 'in', 'on', 'for', 'to', 'with', 'how', 'did', 'does', 'was',
        'were', 'can', 'you', 'do', 'know', 'any', 'some', 'its', 'it',
        'this', 'that', 'from', 'by', 'at', 'be', 'has', 'had', 'have',
        'are', 'been', 'would', 'could', 'should', 'which', 'who', 'when',
        'where', 'why', 'there', 'their', 'they', 'i', 'my', 'we', 'our',
        'nasa', 'mission', 'missions', 'spacecraft', 'space', 'program',
    }
    words = re.findall(r'[a-zA-Z0-9]+', query)
    filtered = [w for w in words if w.lower() not in stop_words and len(w) > 1]

    # Prefer capitalized words (proper nouns — likely mission/topic names)
    proper_nouns = [w for w in filtered if w[0].isupper()]
    return proper_nouns if proper_nouns else filtered


def retrieve(query, n_results=None):
    """Hybrid retrieve: combine vector search with keyword-filtered search."""
    if n_results is None:
        n_results = current_app.config['RAG_TOP_K']

    collection = get_collection()

    query_embedding = ollama.embed(
        model=current_app.config['OLLAMA_EMBED_MODEL'],
        input=query,
    )['embeddings'][0]

    # 1. Pure vector search
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances'],
    )

    # 2. Keyword-filtered vector search for each meaningful keyword
    keyword_hits = {}  # chunk_key -> set of keywords matched
    keyword_results_list = []
    keywords = _extract_keywords(query)
    for keyword in keywords[:3]:  # Limit to top 3 keywords
        try:
            kw_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where_document={'$contains': keyword},
                include=['documents', 'metadatas', 'distances'],
            )
            keyword_results_list.append(kw_results)
            # Track which chunks matched which keywords
            for meta in kw_results['metadatas'][0]:
                key = f"{meta.get('source', '')}_{meta.get('chunk_index', '')}"
                keyword_hits.setdefault(key, set()).add(keyword)
        except Exception:
            pass  # Keyword filter may match nothing

    # 3. Combined keyword search — find chunks matching ALL keywords at once.
    #    This ensures chunks relevant to multi-topic queries (e.g. "Voyager" +
    #    "Jupiter") surface even if they rank poorly in individual searches.
    if len(keywords) >= 2:
        try:
            combined_filter = {'$and': [{'$contains': kw} for kw in keywords[:3]]}
            combined_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where_document=combined_filter,
                include=['documents', 'metadatas', 'distances'],
            )
            keyword_results_list.append(combined_results)
            for meta in combined_results['metadatas'][0]:
                key = f"{meta.get('source', '')}_{meta.get('chunk_index', '')}"
                for kw in keywords[:3]:
                    keyword_hits.setdefault(key, set()).add(kw)
        except Exception:
            pass

    # Merge and deduplicate, keeping best distance per chunk
    seen = {}  # id -> chunk dict
    for results in [vector_results] + keyword_results_list:
        if not results['documents'][0]:
            continue
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
        ):
            chunk_key = f"{meta.get('source', '')}_{meta.get('chunk_index', '')}"
            if chunk_key not in seen or dist < seen[chunk_key]['distance']:
                seen[chunk_key] = {
                    'text': doc,
                    'metadata': meta,
                    'distance': dist,
                }

    # Apply keyword boost: reduce distance for chunks that matched keywords.
    keyword_boost = 0.20
    for chunk_key, chunk in seen.items():
        n_keywords = len(keyword_hits.get(chunk_key, set()))
        if n_keywords > 0:
            chunk['distance'] = max(0.0, chunk['distance'] - n_keywords * keyword_boost)

    # Sort by (boosted) distance and return top n_results
    merged = sorted(seen.values(), key=lambda x: x['distance'])
    return merged[:n_results]


def _build_sources(chunks):
    """Deduplicate chunk metadata into a sources list."""
    seen = OrderedDict()
    for c in chunks:
        source = c['metadata'].get('source', '')
        if source not in seen:
            seen[source] = {
                'title': c['metadata'].get('mission', source),
                'url': source,
                'mission': c['metadata'].get('mission', 'Unknown'),
            }
    return list(seen.values())[:3]


def generate_response(question):
    """Retrieve context and return (token_generator, sources_list)."""
    chunks = retrieve(question)

    if not chunks:
        def empty_gen():
            yield "I don't have any NASA mission data loaded yet. Please check back later."
        return empty_gen(), []

    sources = _build_sources(chunks)

    def _format_chunk_source(c):
        mission = c['metadata'].get('mission', 'Unknown')
        source = c['metadata'].get('source', 'Unknown')
        return f"[Source: {mission} — {source}]:\n{c['text']}"

    context = '\n\n'.join(_format_chunk_source(c) for c in chunks)

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT.format(today=date.today().strftime('%B %d, %Y'))},
        {'role': 'user', 'content': USER_PROMPT_TEMPLATE.format(
            context=context, question=question,
        )},
    ]

    def token_generator():
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

    return token_generator(), sources
