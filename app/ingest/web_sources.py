"""Scrape NASA's A-to-Z missions index for mission page URLs."""

import logging
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

A_TO_Z_URL = 'https://www.nasa.gov/a-to-z-of-nasa-missions/'


def fetch_mission_urls():
    """Scrape the A-to-Z index and return a list of mission dicts.

    Returns:
        List of dicts with keys: name, url
    """
    logger.info('Fetching NASA A-to-Z missions index...')
    resp = requests.get(A_TO_Z_URL, timeout=60)
    resp.raise_for_status()

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

        if href in seen_urls or not name:
            continue

        seen_urls.add(href)
        missions.append({
            'name': name,
            'url': href,
        })

    logger.info('Found %d mission pages', len(missions))
    return missions
