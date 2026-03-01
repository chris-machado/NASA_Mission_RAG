#!/usr/bin/env python3
"""Run the document ingestion pipeline for all downloaded NASA PDFs."""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from app.ingest.sources import NASA_DOCUMENTS
from app.ingest.pipeline import ingest_pdf

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PDF_DIR = os.path.join(PROJECT_ROOT, 'data', 'pdfs')
CHROMA_DIR = os.path.join(PROJECT_ROOT, 'data', 'chroma_db')
EMBED_MODEL = os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')


def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name='nasa_reports',
        metadata={'hnsw:space': 'cosine'},
    )

    total_chunks = 0
    processed = 0
    skipped = 0

    for doc in NASA_DOCUMENTS:
        pdf_path = os.path.join(PDF_DIR, doc['filename'])
        if not os.path.exists(pdf_path):
            logger.warning('PDF not found: %s — run download_docs.py first', doc['filename'])
            skipped += 1
            continue

        count = ingest_pdf(collection, pdf_path, doc, embed_model=EMBED_MODEL)
        total_chunks += count
        processed += 1

    logger.info('--- Ingestion complete ---')
    logger.info('Processed: %d documents', processed)
    logger.info('Skipped: %d documents (not downloaded)', skipped)
    logger.info('Total chunks in collection: %d', collection.count())


if __name__ == '__main__':
    main()
