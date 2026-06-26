"""Repair ChromaDB metadata in place — no re-scrape, no re-embed.

Recomputes a consistent per-source launch ``year`` from the stored chunk text
using the improved extractor, and deletes a small set of unambiguous scraper
noise (news-feed 'missions', pure-navigation chunks).

Dry-run by default; pass --apply to write. Stop the web server first if your
ChromaDB build locks the store for concurrent writers.

    python scripts/repair_metadata.py            # preview
    python scripts/repair_metadata.py --apply     # write
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from app import create_app
from app.extensions import get_collection
from app.ingest.pipeline import extract_year

# News-feed cards scraped as "missions".
_NAME_JUNK = re.compile(
    r'\bmin read\b|\b\d+\s*(?:hours?|days?|minutes?|weeks?|months?)\s+ago\b'
    r'|article\d', re.I)
# Navigation/boilerplate phrases; a chunk that is almost nothing but these is junk.
_BOILER = re.compile(
    r"Play this trivia game and test your knowledge!?|Control Center"
    r"|360-Degree Tour|Take a virtual tour[^.]*|Discover More Topics From NASA"
    r"|Additional Resources|Download high-resolution[^.]*|Read More", re.I)


def _is_pure_boilerplate(doc):
    residual = _BOILER.sub('', doc)
    return sum(c.isalpha() for c in residual) < 80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='write changes (default: dry run)')
    args = ap.parse_args()

    app = create_app('development')
    with app.app_context():
        col = get_collection()
        data = col.get(include=['documents', 'metadatas'])
        ids, docs, metas = data['ids'], data['documents'], data['metadatas']
        print(f"Loaded {len(ids)} chunks.")

        # 1. Decide deletions.
        del_ids, del_reasons = set(), Counter()
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            if _NAME_JUNK.search(meta.get('mission', '')):
                del_ids.add(ids[i]); del_reasons['junk-mission-name'] += 1
            elif _BOILER.search(doc) and _is_pure_boilerplate(doc):
                del_ids.add(ids[i]); del_reasons['pure-boilerplate'] += 1

        # 2. Recompute per-source year from the surviving chunk text.
        by_source = defaultdict(list)
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            if ids[i] in del_ids:
                continue
            by_source[meta.get('source', '')].append((meta.get('chunk_index', 0), doc))
        new_year = {
            src: extract_year(' '.join(t for _, t in sorted(chunks)))
            for src, chunks in by_source.items()
        }

        # 3. Build metadata updates for surviving chunks.
        upd_ids, upd_metas, changed = [], [], 0
        old_dist, new_dist = Counter(), Counter()
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            if ids[i] in del_ids:
                continue
            ny = new_year.get(meta.get('source', ''), 0)
            old_dist[meta.get('year', 0)] += 1
            new_dist[ny] += 1
            if ny != meta.get('year'):
                changed += 1
            nm = dict(meta); nm['year'] = ny
            upd_ids.append(ids[i]); upd_metas.append(nm)

        # Report.
        print(f"\nDeletions: {len(del_ids)} chunks  {dict(del_reasons)}")
        print(f"Year updates: {changed} chunks change year "
              f"(of {len(upd_ids)} surviving)")
        unknown_old = old_dist.get(0, 0)
        unknown_new = new_dist.get(0, 0)
        print(f"year==0 (unknown): {unknown_old} -> {unknown_new}")
        print("Noisiest OLD years:", old_dist.most_common(4))
        recent = sorted((y for y in new_dist if y and y >= 2023))
        print("NEW recent-year chunk counts:",
              {y: new_dist[y] for y in recent})

        if not args.apply:
            print("\nDRY RUN — re-run with --apply to write.")
            return

        # 4. Apply: delete first, then batched metadata updates.
        if del_ids:
            col.delete(ids=list(del_ids))
            print(f"Deleted {len(del_ids)} chunks.")
        B = 500
        for i in range(0, len(upd_ids), B):
            col.update(ids=upd_ids[i:i + B], metadatas=upd_metas[i:i + B])
        print(f"Updated metadata for {len(upd_ids)} chunks. Final count: {col.count()}")


if __name__ == '__main__':
    main()
