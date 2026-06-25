import logging
import re
from collections import OrderedDict, defaultdict
from datetime import date

from flask import current_app

from app.chat.ollama_client import get_client
from app.chat.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from app.extensions import get_collection

logger = logging.getLogger(__name__)

# nomic-embed-text is trained with task-instruction prefixes, and its Ollama
# template ({{ .Prompt }}) does not add them — so we must. Queries use
# "search_query:"; stored passages use "search_document:" (added at ingest time).
QUERY_PREFIX = 'search_query: '

# Reciprocal Rank Fusion damping constant (standard default).
RRF_K = 60


def _extract_keywords(query):
    """Extract meaningful keywords for the lexical (keyword-filtered) arm.

    Proper nouns (capitalized mission/topic names) are ordered first, but
    inclusion no longer depends on casing — so "voyager jupiter" and
    "Voyager Jupiter" extract the same keyword set.
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

    # Prefer capitalized words (likely proper nouns) in ordering, but keep all.
    proper_nouns = [w for w in filtered if w[0].isupper()]
    others = [w for w in filtered if not w[0].isupper()]

    ordered = []
    seen = set()
    for w in proper_nouns + others:
        if w.lower() not in seen:
            seen.add(w.lower())
            ordered.append(w)
    return ordered


def _chunk_key(meta):
    return f"{meta.get('source', '')}_{meta.get('chunk_index', '')}"


def retrieve(query, n_results=None):
    """Hybrid retrieve: fuse a vector arm with keyword-filtered arms via RRF."""
    if n_results is None:
        n_results = current_app.config['RAG_TOP_K']
    max_per_source = current_app.config.get('RAG_MAX_PER_SOURCE', 0)

    collection = get_collection()
    client = get_client()

    query_embedding = client.embed(
        model=current_app.config['OLLAMA_EMBED_MODEL'],
        input=QUERY_PREFIX + query,
    )['embeddings'][0]

    # Over-fetch per arm so fusion has enough candidates to work with.
    fetch_n = max(n_results * 4, 20)
    ranked_lists = []       # list of ranked [chunk_key, ...] — one per arm
    chunk_data = {}         # chunk_key -> {text, metadata, distance}

    def _record(results):
        if not results['documents'] or not results['documents'][0]:
            return
        order = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
        ):
            key = _chunk_key(meta)
            order.append(key)
            if key not in chunk_data or dist < chunk_data[key]['distance']:
                chunk_data[key] = {'text': doc, 'metadata': meta, 'distance': dist}
        ranked_lists.append(order)

    # Arm 1: pure semantic (vector) search.
    _record(collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_n,
        include=['documents', 'metadatas', 'distances'],
    ))

    # Arm 2..N: lexical arm — same query vector, filtered to chunks containing a
    # keyword. Chroma's $contains is case-sensitive, so proper-noun ordering helps.
    keywords = _extract_keywords(query)
    for keyword in keywords[:3]:
        try:
            _record(collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_n,
                where_document={'$contains': keyword},
                include=['documents', 'metadatas', 'distances'],
            ))
        except Exception:
            pass  # Keyword filter may match nothing.

    # Combined lexical arm: chunks containing ALL top keywords (multi-topic queries
    # like "Voyager" + "Jupiter"), which can rank poorly in individual arms.
    if len(keywords) >= 2:
        try:
            _record(collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_n,
                where_document={'$and': [{'$contains': kw} for kw in keywords[:3]]},
                include=['documents', 'metadatas', 'distances'],
            ))
        except Exception:
            pass

    if not chunk_data:
        return []

    # Reciprocal Rank Fusion: a chunk surfaced highly by several arms outranks one
    # that's strong in a single arm — without arithmetic on the cosine distances.
    rrf = defaultdict(float)
    for order in ranked_lists:
        for rank, key in enumerate(order):
            rrf[key] += 1.0 / (RRF_K + rank)
    ranked_keys = sorted(rrf, key=lambda k: rrf[k], reverse=True)

    # Take the top chunks, capping how many come from any single mission page so
    # the context isn't dominated by near-duplicate overlapping chunks.
    selected, selected_keys = [], set()
    per_source = defaultdict(int)
    for key in ranked_keys:
        source = chunk_data[key]['metadata'].get('source', '')
        if max_per_source and per_source[source] >= max_per_source:
            continue
        selected.append(chunk_data[key])
        selected_keys.add(key)
        per_source[source] += 1
        if len(selected) >= n_results:
            break

    # Backfill if the diversity cap left us short (few sources matched the query).
    if len(selected) < n_results:
        for key in ranked_keys:
            if key in selected_keys:
                continue
            selected.append(chunk_data[key])
            if len(selected) >= n_results:
                break

    return selected


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

    # Read all config + build the client while we're definitely in app context;
    # the generator below is consumed later during SSE streaming.
    client = get_client()
    model = current_app.config['OLLAMA_CHAT_MODEL']
    think = current_app.config['OLLAMA_THINK']
    options = {
        'temperature': current_app.config['RAG_TEMPERATURE'],
        'num_ctx': current_app.config['OLLAMA_NUM_CTX'],
    }

    def token_generator():
        stream = client.chat(model=model, messages=messages, stream=True,
                             think=think, options=options)
        for chunk in stream:
            # During any "thinking" phase content is empty; only answer text streams.
            yield chunk['message']['content'] or ''

    return token_generator(), sources
