"""Document ingestion pipeline: PDF → chunks → embeddings → ChromaDB."""

import logging
import os
import re

import ollama
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF, page by page."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ''
        text = clean_text(text)
        if len(text.strip()) > 50:  # Skip near-empty pages (figures, diagrams)
            pages.append({'text': text, 'page': i + 1})
    return pages


def clean_text(text):
    """Clean extracted PDF text."""
    # Collapse multiple whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove common header/footer patterns
    text = re.sub(r'NASA-\w+-\d+', '', text)
    return text.strip()


def chunk_pages(pages, chunk_size=800, chunk_overlap=200):
    """Split page texts into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', '. ', ' ', ''],
    )

    chunks = []
    for page_data in pages:
        page_chunks = splitter.split_text(page_data['text'])
        for i, chunk_text in enumerate(page_chunks):
            chunks.append({
                'text': chunk_text,
                'page': page_data['page'],
                'chunk_index': i,
            })
    return chunks


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


def ingest_pdf(collection, pdf_path, doc_metadata, embed_model='nomic-embed-text'):
    """Ingest a single PDF into ChromaDB."""
    filename = os.path.basename(pdf_path)
    logger.info('Processing %s...', filename)

    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        logger.warning('No text extracted from %s, skipping', filename)
        return 0

    chunks = chunk_pages(pages)
    logger.info('  %d pages → %d chunks', len(pages), len(chunks))

    embeddings = embed_chunks(chunks, model=embed_model)

    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    documents = [c['text'] for c in chunks]
    metadatas = [
        {
            'source': filename,
            'mission': doc_metadata.get('mission', 'Unknown'),
            'year': doc_metadata.get('year', 0),
            'page': c['page'],
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

    logger.info('  Upserted %d chunks for %s', len(chunks), doc_metadata.get('mission', filename))
    return len(chunks)
