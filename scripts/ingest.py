#!/usr/bin/env python3
"""Ingest NASA mission pages into ChromaDB from the A-to-Z missions index."""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from app.ingest.web_sources import fetch_mission_urls
from app.ingest.pipeline import ingest_web_page

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CHROMA_DIR = os.path.join(PROJECT_ROOT, 'data', 'chroma_db')
EMBED_MODEL = os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')


def main():
    parser = argparse.ArgumentParser(description='Ingest NASA mission web pages')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max number of pages to ingest (0 = all)')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Seconds to wait between page fetches')
    parser.add_argument('--filter', type=str, default='',
                        help='Only ingest missions whose name contains this string (case-insensitive)')
    args = parser.parse_args()

    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name='nasa_reports',
        metadata={'hnsw:space': 'cosine'},
    )

    missions = fetch_mission_urls()

    if args.filter:
        missions = [m for m in missions if args.filter.lower() in m['name'].lower()]
        logger.info('Filtered to %d missions matching "%s"', len(missions), args.filter)

    if args.limit:
        missions = missions[:args.limit]
        logger.info('Limited to %d missions', len(missions))

    total_chunks = 0
    processed = 0
    skipped = 0

    for i, mission in enumerate(missions):
        logger.info('[%d/%d] %s', i + 1, len(missions), mission['name'])
        try:
            count = ingest_web_page(
                collection, mission['url'], mission['name'],
                embed_model=EMBED_MODEL,
            )
            if count > 0:
                total_chunks += count
                processed += 1
            else:
                skipped += 1
        except Exception:
            logger.exception('  Failed to ingest %s', mission['url'])
            skipped += 1

        # Be polite to NASA servers
        if i < len(missions) - 1:
            time.sleep(args.delay)

    logger.info('--- Web ingestion complete ---')
    logger.info('Processed: %d pages', processed)
    logger.info('Skipped: %d pages', skipped)
    logger.info('New chunks added: %d', total_chunks)
    logger.info('Total chunks in collection: %d', collection.count())


if __name__ == '__main__':
    main()
