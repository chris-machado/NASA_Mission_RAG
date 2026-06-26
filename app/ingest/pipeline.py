"""Document ingestion pipeline: web pages → chunks → embeddings → ChromaDB."""

import hashlib
import logging
import re
import time
import unicodedata
from datetime import date

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chat.ollama_client import get_client

logger = logging.getLogger(__name__)

# nomic-embed-text task prefix for stored passages (queries use "search_query:").
DOCUMENT_PREFIX = 'search_document: '


def clean_text(text):
    """Clean extracted web text."""
    # Normalize unicode (smart quotes, em dashes → ASCII equivalents)
    text = unicodedata.normalize('NFKD', text)
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('–', '-').replace('—', '-')
    # Strip any residual HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove bracketed alt-text leakage (lines that are just [some alt text])
    text = re.sub(r'^\[.*?\]$', '', text, flags=re.MULTILINE)
    # Collapse multiple whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


_MONTHS = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)[a-z]*\.?'
_YEAR = r'(?:19|20)\d{2}'
# A year sitting right after a launch/liftoff verb (optionally with a date).
_LAUNCH_YEAR = re.compile(
    r'\b(?:launch(?:ed|es|ing)?|lift[\s-]?off|liftoff)\b[^.\d]{0,30}?'
    r'(?:' + _MONTHS + r'\s+\d{1,2}(?:st|nd|rd|th)?,?\s+)?(' + _YEAR + r')\b', re.I)
# A full "Month Day, Year" calendar date.
_FULL_DATE = re.compile(
    _MONTHS + r'\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(' + _YEAR + r')\b', re.I)
# A "Month Year" date with no day.
_MONTH_YEAR = re.compile(_MONTHS + r'\s+(' + _YEAR + r')\b', re.I)


def extract_year(text, today=None):
    """Best-effort launch/start year for a mission page.

    Tiered, most-reliable-signal-first: (1) a year next to a launch/liftoff verb,
    (2) the earliest full "Month Day, Year" date, (3) the earliest "Month Year".
    Returns 0 when none is present — deliberately, rather than guessing from the
    earliest bare 4-digit number, which previously mistagged pages with the
    NASA-founding year (1958) or a stray copyright year. "Plausible" = 1958 ..
    current year + 10 (to allow announced future launches).
    """
    current_year = (today or date.today()).year
    lo, hi = 1958, current_year + 10

    def earliest(regex):
        years = [int(y) for y in regex.findall(text)]
        years = [y for y in years if lo <= y <= hi]
        return min(years) if years else None

    return (earliest(_LAUNCH_YEAR)
            or earliest(_FULL_DATE)
            or earliest(_MONTH_YEAR)
            or 0)


def _get_with_retry(url, *, timeout=60, retries=3, backoff=2.0, headers=None):
    """HTTP GET with simple exponential backoff for transient failures."""
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


def extract_text_from_web_page(url):
    """Fetch a web page and extract its main textual content."""
    resp = _get_with_retry(url, timeout=60, headers={
        'User-Agent': 'NASA-Mission-RAG/1.0 (Educational Research)',
    })

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Remove non-content elements
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header',
                               'aside', 'form', 'iframe', 'noscript']):
        tag.decompose()

    # Remove boilerplate sections by class name
    for section in soup.find_all(['section', 'div'], class_=re.compile(
            r'related|latest|news-list|card-grid|sidebar|share|social|comment|'
            r'newsletter|breadcrumb|pagination|menu|toolbar|banner|cookie|modal|popup',
            re.I)):
        section.decompose()

    # Try to find the main content area
    main = (soup.find('main')
            or soup.find('article')
            or soup.find('div', {'role': 'main'})
            or soup.find('div', class_=re.compile(r'content|entry|article|post', re.I)))

    target = main if main else soup.body if soup.body else soup

    # Extract text from meaningful elements
    blocks = []
    seen_texts = set()
    for el in target.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'td',
                                'blockquote', 'figcaption', 'dd', 'dt']):
        # Skip elements that are inside link-heavy containers (nav-like lists)
        links = el.find_all('a')
        text = el.get_text(separator=' ', strip=True)
        if len(text) < 20:
            continue
        # Skip if the element is mostly links (navigation/related content)
        link_text_len = sum(len(a.get_text(strip=True)) for a in links)
        if link_text_len > len(text) * 0.7 and len(links) > 1:
            continue
        # Deduplicate
        if text in seen_texts:
            continue
        seen_texts.add(text)
        blocks.append(text)

    full_text = '\n\n'.join(blocks)

    # Remove common boilerplate lines
    boilerplate_patterns = re.compile(
        r'^\s*(Share|Read More|Follow NASA|Credits:|Tags:|Last Updated)\b.*$',
        re.MULTILINE | re.IGNORECASE,
    )
    full_text = boilerplate_patterns.sub('', full_text)

    # Safety net: strip any residual HTML tags
    full_text = re.sub(r'<[^>]+>', '', full_text)

    full_text = clean_text(full_text)
    return full_text


def chunk_text(text, chunk_size=800, chunk_overlap=200):
    """Split a plain text string into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', '. ', ' ', ''],
    )
    raw_chunks = splitter.split_text(text)
    return [
        {'text': t, 'chunk_index': i}
        for i, t in enumerate(raw_chunks)
    ]


def embed_chunks(chunks, model='nomic-embed-text', batch_size=50):
    """Generate embeddings for text chunks using Ollama.

    Each passage is prefixed with "search_document:" (nomic-embed-text's expected
    document instruction); the stored chunk text stays clean.
    """
    client = get_client()
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [DOCUMENT_PREFIX + c['text'] for c in batch]
        response = client.embed(model=model, input=texts)
        embeddings.extend(response['embeddings'])
        logger.info('Embedded batch %d/%d', i // batch_size + 1,
                     (len(chunks) + batch_size - 1) // batch_size)
    return embeddings


def ingest_web_page(collection, url, mission_name, embed_model='nomic-embed-text'):
    """Ingest a single web page into ChromaDB (idempotent per source URL)."""
    logger.info('Processing web page: %s ...', mission_name)

    text = extract_text_from_web_page(url)
    if len(text.strip()) < 100:
        logger.warning('  Insufficient content from %s, skipping', url)
        return 0

    chunks = chunk_text(text)
    year = extract_year(text)
    logger.info('  %d characters → %d chunks (year=%s)', len(text), len(chunks), year)

    embeddings = embed_chunks(chunks, model=embed_model)

    # Stable, collision-free id prefix from a hash of the URL.
    source_hash = hashlib.sha1(url.encode('utf-8')).hexdigest()[:16]
    ids = [f"web_{source_hash}_{i}" for i in range(len(chunks))]
    documents = [c['text'] for c in chunks]
    metadatas = [
        {
            'source': url,
            'mission': mission_name,
            'year': year,
            'chunk_index': c['chunk_index'],
        }
        for c in chunks
    ]

    # Idempotent re-ingest: drop any prior chunks for this URL first, so a page
    # that now yields fewer chunks can't leave orphaned high-index chunks behind.
    try:
        collection.delete(where={'source': url})
    except Exception:
        logger.exception('  Failed to clear existing chunks for %s', url)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info('  Upserted %d chunks for %s', len(chunks), mission_name)
    return len(chunks)
