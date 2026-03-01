#!/usr/bin/env python3
"""Download NASA mission report PDFs from NTRS."""

import os
import sys
import time

import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.ingest.sources import NASA_DOCUMENTS

PDF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pdfs')


def download_documents():
    os.makedirs(PDF_DIR, exist_ok=True)

    for doc in NASA_DOCUMENTS:
        filepath = os.path.join(PDF_DIR, doc['filename'])

        if os.path.exists(filepath):
            print(f"  [skip] {doc['filename']} already exists")
            continue

        print(f"  [download] {doc['title']}...")
        try:
            resp = requests.get(doc['url'], timeout=120, stream=True)
            resp.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"             Saved {doc['filename']} ({size_mb:.1f} MB)")

            # Be polite to NASA servers
            time.sleep(2)

        except requests.RequestException as e:
            print(f"  [error] Failed to download {doc['title']}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)

    print(f"\nDone. PDFs saved to {PDF_DIR}")


if __name__ == '__main__':
    download_documents()
