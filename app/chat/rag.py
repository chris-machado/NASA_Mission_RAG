import logging
import re
from collections import OrderedDict, defaultdict
from datetime import date

from flask import current_app

from app.chat import reranker
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

# Clearly time-scoped cue words. Deliberately excludes ambiguous ones — "current"
# (cf. "ocean current"), "new"/"now" (collide with mission names and filler) —
# because a false positive narrows the pool to recent years and can hide the
# real answer to a non-temporal question.
RECENCY_TERMS = [
    'latest', 'most recent', 'recent', 'recently', 'newest',
    'upcoming', 'nowadays', 'this year', 'these days',
]
_RECENCY_RE = re.compile(
    '|'.join(r'\b' + re.escape(t) + r'\b' for t in RECENCY_TERMS), re.I)
_YEAR_RE = re.compile(r'\b(?:19|20)\d{2}\b')
_YEAR_LO, _YEAR_FUTURE_PAD = 1958, 10


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


def _temporal_targets(query, today=None):
    """Classify a time-scoped query.

    Returns ``(year_window, year_tokens, primary_years, is_recency)``:
    ``year_window`` is the metadata years to allow, ``year_tokens`` the year
    strings to match in passage text, ``primary_years`` the strongest-match years
    used to re-sort after reranking. ``(None, None, None, False)`` means an
    ordinary topical query.

    - Explicit year(s) → those years ±1 (launch-year tagging is approximate, so a
      2026 mission may be tagged 2025); the exact years are "primary".
    - Recency words → the band around today (last year .. next year).
    """
    current = (today or date.today()).year
    hi = current + _YEAR_FUTURE_PAD
    explicit = sorted({int(y) for y in _YEAR_RE.findall(query)})
    if explicit:
        window = set()
        for y in explicit:
            window.update((y - 1, y, y + 1))
        window = {y for y in window if _YEAR_LO <= y <= hi}
        primary = {y for y in explicit if _YEAR_LO <= y <= hi}
        return window, {str(y) for y in explicit}, primary, False
    if _RECENCY_RE.search(query):
        band = {current - 1, current, current + 1}
        return band, {str(current - 1), str(current)}, band, True
    return None, None, None, False


def _apply_source_cap(ordered_chunks, max_per_source, n_results):
    """Take the best chunks in order, capping how many come from one mission page
    so the context isn't dominated by near-duplicate overlapping chunks."""
    selected = []
    per_source = defaultdict(int)
    for c in ordered_chunks:
        source = c['metadata'].get('source', '')
        if max_per_source and per_source[source] >= max_per_source:
            continue
        selected.append(c)
        per_source[source] += 1
        if len(selected) >= n_results:
            break
    return selected


def retrieve(query, n_results=None):
    """Two-stage retrieval: build a candidate pool, then cross-encoder rerank.

    Topical queries fuse a vector arm with keyword arms via RRF. Time-scoped
    queries ("latest missions in 2026") instead gate the pool to year-matching
    chunks first — otherwise the reranker, which scores topical relevance and is
    blind to recency, happily floats a vivid old mission to the top. The reranker
    then orders whichever pool we built; a per-source cap keeps the result diverse.
    """
    if n_results is None:
        n_results = current_app.config['RAG_TOP_K']
    max_per_source = current_app.config.get('RAG_MAX_PER_SOURCE', 0)
    candidate_k = current_app.config.get('RAG_CANDIDATE_K', 40)
    rerank_enabled = current_app.config.get('RAG_RERANK_ENABLED', True)

    collection = get_collection()
    client = get_client()
    query_embedding = client.embed(
        model=current_app.config['OLLAMA_EMBED_MODEL'],
        input=QUERY_PREFIX + query,
    )['embeddings'][0]

    fetch_n = max(candidate_k, 20)
    chunk_data = {}  # chunk_key -> {text, metadata, distance}

    def _record(results):
        if not results['documents'] or not results['documents'][0]:
            return []
        order = []
        for doc, meta, dist in zip(results['documents'][0],
                                   results['metadatas'][0],
                                   results['distances'][0]):
            key = _chunk_key(meta)
            order.append(key)
            if key not in chunk_data or dist < chunk_data[key]['distance']:
                chunk_data[key] = {'text': doc, 'metadata': meta, 'distance': dist}
        return order

    year_window, year_tokens, primary_years, is_recency = _temporal_targets(query)

    if year_window:
        # Temporal pool: chunks whose metadata year is in the window, plus chunks
        # whose text literally names a target year (catches year==0 mistags).
        try:
            _record(collection.query(
                query_embeddings=[query_embedding], n_results=fetch_n,
                where={'year': {'$in': sorted(year_window)}},
                include=['documents', 'metadatas', 'distances']))
        except Exception:
            logger.exception('temporal metadata arm failed')
        for token in year_tokens:
            try:
                _record(collection.query(
                    query_embeddings=[query_embedding], n_results=fetch_n,
                    where_document={'$contains': token},
                    include=['documents', 'metadatas', 'distances']))
            except Exception:
                pass
        pool_keys = sorted(chunk_data, key=lambda k: chunk_data[k]['distance'])[:candidate_k]
    else:
        # Topical pool: vector arm + keyword-filtered arms, fused via RRF.
        ranked_lists = [_record(collection.query(
            query_embeddings=[query_embedding], n_results=fetch_n,
            include=['documents', 'metadatas', 'distances']))]
        keywords = _extract_keywords(query)
        for keyword in keywords[:3]:
            try:
                ranked_lists.append(_record(collection.query(
                    query_embeddings=[query_embedding], n_results=fetch_n,
                    where_document={'$contains': keyword},
                    include=['documents', 'metadatas', 'distances'])))
            except Exception:
                pass
        if len(keywords) >= 2:
            try:
                ranked_lists.append(_record(collection.query(
                    query_embeddings=[query_embedding], n_results=fetch_n,
                    where_document={'$and': [{'$contains': kw} for kw in keywords[:3]]},
                    include=['documents', 'metadatas', 'distances'])))
            except Exception:
                pass
        rrf = defaultdict(float)
        for order in ranked_lists:
            for rank, key in enumerate(order):
                rrf[key] += 1.0 / (RRF_K + rank)
        pool_keys = sorted(rrf, key=lambda k: rrf[k], reverse=True)[:candidate_k]

    # Fallback: a temporal query that matched nothing (e.g. a year we don't cover)
    # still gets a plain vector pool rather than an empty answer.
    if not pool_keys:
        order = _record(collection.query(
            query_embeddings=[query_embedding], n_results=fetch_n,
            include=['documents', 'metadatas', 'distances']))
        pool_keys = order[:candidate_k]
    if not pool_keys:
        return []

    pool = [chunk_data[k] for k in pool_keys]

    # Each candidate's first-stage rank (its position in the pool) seeds a fusion
    # score we may refine with the reranker.
    for rank, c in enumerate(pool):
        c['_retrieval_rank'] = rank
        c['score'] = 1.0 / (RRF_K + rank)

    # Precision stage: a cross-encoder scores true query/passage relevance, but we
    # *fuse* its ranking with the first-stage ranking via weighted RRF rather than
    # letting it override. A confident-but-wrong reranker otherwise buries strong
    # retrieval hits — e.g. for "what went wrong on Apollo 13" bge-reranker floats
    # other missions' failure stories above Apollo 13 itself.
    if rerank_enabled:
        scores = reranker.score(
            query, pool,
            current_app.config['RAG_RERANK_MODEL'],
            current_app.config.get('RAG_RERANK_DEVICE', 'auto'))
        if any(s is not None for s in scores):
            rerank_weight = current_app.config.get('RAG_RERANK_WEIGHT', 1.0)
            rerank_order = sorted(
                range(len(pool)),
                key=lambda i: (scores[i] is not None, scores[i] or 0.0),
                reverse=True)
            for rerank_rank, i in enumerate(rerank_order):
                c = pool[i]
                c['rerank_score'] = scores[i]
                c['score'] = (1.0 / (RRF_K + c['_retrieval_rank'])
                              + rerank_weight / (RRF_K + rerank_rank))

    # Order by fused score. For temporal queries the year tier is a hard constraint
    # the reranker is blind to, so it dominates; the fused score breaks ties.
    if year_window:
        def _order_key(c):
            y = c['metadata'].get('year') or 0
            return (y in primary_years, y in year_window, c['score'])
    else:
        def _order_key(c):
            return c['score']
    pool.sort(key=_order_key, reverse=True)

    selected = _apply_source_cap(pool, max_per_source, n_results)
    if len(selected) < n_results:  # diversity cap left us short — backfill in order
        chosen = {id(c) for c in selected}
        for c in pool:
            if id(c) not in chosen:
                selected.append(c)
                if len(selected) >= n_results:
                    break
    return selected[:n_results]


def _citable_chunks(question, chunks):
    """Filter retrieved chunks down to the ones relevant enough to cite.

    Vector search always returns nearest-neighbour chunks, so off-topic questions
    ("what is your name?") and focused ones alike pull weakly-related pages the
    answer never really uses. We cite only the relevant ones — which drops both
    bogus citations and dubious leftovers (Apollo 13 -> STS-36, Apollo 1).

    Topical queries judge relevance with the cross-encoder (clean: on-topic >0.5,
    weak <0.35). Temporal queries judge by year instead — the reranker is fooled
    by year mentions (it rates an old mission whose page says "2026" above a real
    2026 launch), so a chunk is citable iff its launch year is in the window.
    """
    year_window, _, _, _ = _temporal_targets(question)
    if year_window:
        cited = [c for c in chunks if (c['metadata'].get('year') or 0) in year_window]
        return cited or chunks  # nothing in-window (sparse year) -> don't over-filter

    min_rr = current_app.config.get('RAG_RELEVANCE_MIN_RERANK', 0.4)
    max_d = current_app.config.get('RAG_RELEVANCE_MAX_DISTANCE', 0.31)
    scored = [c for c in chunks if c.get('rerank_score') is not None]
    if scored:
        # A citable source is BOTH cross-encoder-relevant AND a close vector
        # match — this excludes "related but not the subject" pages (Gemini XII
        # scores 0.95 for an Apollo 13 question via a shared astronaut, but is a
        # far vector match at 0.33, so distance drops it).
        cited = [c for c in scored
                 if c['rerank_score'] >= min_rr and c['distance'] <= max_d]
        # Safety net: a clearly-relevant query whose best chunk is rerank-strong
        # but vector-distant still gets its top source, never zero citations.
        if not cited and scored[0]['rerank_score'] >= min_rr:
            return [scored[0]]
        return cited
    # Reranker disabled: cosine distance alone.
    return [c for c in chunks if c.get('distance') is not None and c['distance'] <= max_d]


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

    # Cite only the chunks actually relevant to the question, so off-topic answers
    # ("I don't have a name") carry no citations and focused answers don't tack on
    # weakly-related leftovers.
    sources = _build_sources(_citable_chunks(question, chunks))

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
