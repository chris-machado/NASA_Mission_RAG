"""Document ingestion pipeline: web pages → chunks → embeddings → ChromaDB."""

import logging
import re
import unicodedata

import ollama
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def clean_text(text):
    """Clean extracted web text."""
    # Normalize unicode (smart quotes, em dashes → ASCII equivalents)
    text = unicodedata.normalize('NFKD', text)
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    # Strip any residual HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove bracketed alt-text leakage (lines that are just [some alt text])
    text = re.sub(r'^\[.*?\]$', '', text, flags=re.MULTILINE)
    # Collapse multiple whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_text_from_web_page(url):
    """Fetch a web page and extract its main textual content."""
    resp = requests.get(url, timeout=60, headers={
        'User-Agent': 'NASA-Mission-RAG/1.0 (Educational Research)',
    })
    resp.raise_for_status()

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
        {'text': t, 'page': 1, 'chunk_index': i}
        for i, t in enumerate(raw_chunks)
    ]


def embed_chunks(chunks, model='nomic-embed-text', batch_size=50):
    """Generate embeddings for text chunks using Ollama."""
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c['text'] for c in batch]
        response = ollama.embed(model=model, input=texts)
        embeddings.extend(response['embeddings'])
        logger.info('Embedded batch %d/%d', i // batch_size + 1,
                     (len(chunks) + batch_size - 1) // batch_size)
    return embeddings


def ingest_web_page(collection, url, mission_name, embed_model='nomic-embed-text'):
    """Ingest a single web page into ChromaDB."""
    logger.info('Processing web page: %s ...', mission_name)

    text = extract_text_from_web_page(url)
    if len(text.strip()) < 100:
        logger.warning('  Insufficient content from %s, skipping', url)
        return 0

    chunks = chunk_text(text)
    logger.info('  %d characters → %d chunks', len(text), len(chunks))

    embeddings = embed_chunks(chunks, model=embed_model)

    # Create a stable source id from the URL
    source_id = re.sub(r'[^a-zA-Z0-9]', '_', url.split('nasa.gov')[-1])[:80]
    ids = [f"web_{source_id}_{i}" for i in range(len(chunks))]
    documents = [c['text'] for c in chunks]
    metadatas = [
        {
            'source': url,
            'mission': mission_name,
            'year': 0,
            'page': 1,
            'chunk_index': c['chunk_index'],
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info('  Upserted %d chunks for %s', len(chunks), mission_name)
    return len(chunks)
