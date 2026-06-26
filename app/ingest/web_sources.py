"""Scrape NASA's A-to-Z missions index for mission page URLs."""

import logging
import re

from bs4 import BeautifulSoup

from app.ingest.pipeline import _get_with_retry

logger = logging.getLogger(__name__)

A_TO_Z_URL = 'https://www.nasa.gov/a-to-z-of-nasa-missions/'

# Some /mission/ links are news-feed cards, not missions; their link text reads
# like "3 min read<title>article6 hours ago". Reject names that look like that.
_NAME_JUNK = re.compile(
    r'\bmin read\b|\b\d+\s*(?:hours?|days?|minutes?|weeks?|months?)\s+ago\b'
    r'|article\d', re.I)


def _is_valid_mission_name(name):
    return bool(name) and len(name) <= 90 and not _NAME_JUNK.search(name)


def fetch_mission_urls():
    """Scrape the A-to-Z index and return a list of mission dicts.

    Returns:
        List of dicts with keys: name, url
    """
    logger.info('Fetching NASA A-to-Z missions index...')
    resp = _get_with_retry(A_TO_Z_URL, timeout=60)

    soup = BeautifulSoup(resp.text, 'html.parser')
    missions = []
    seen_urls = set()

    for link in soup.find_all('a', href=True):
        href = link['href']
        name = link.get_text(strip=True)

        # Mission links point to /mission/ paths on nasa.gov or science.nasa.gov
        if not re.search(r'(science\.)?nasa\.gov/mission/', href):
            continue

        # Normalize URL
        if href.startswith('//'):
            href = 'https:' + href
        elif href.startswith('/'):
            href = 'https://www.nasa.gov' + href
        if not href.startswith('https://'):
            href = 'https://' + href

        # Ensure trailing slash for consistency
        if not href.endswith('/'):
            href += '/'

        if href in seen_urls or not _is_valid_mission_name(name):
            continue

        seen_urls.add(href)
        missions.append({
            'name': name,
            'url': href,
        })

    if not missions:
        raise RuntimeError(
            'No mission URLs found on the A-to-Z index — the page structure may '
            'have changed. Aborting so an empty scrape cannot replace the corpus.'
        )

    logger.info('Found %d mission pages', len(missions))
    return missions
