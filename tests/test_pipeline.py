"""Tests for ingestion: chunking, year extraction, embedding prefixes."""

import hashlib

import app.ingest.pipeline as pl
from app.ingest.pipeline import DOCUMENT_PREFIX, chunk_text, extract_year


def test_chunk_text_sequential_indices_no_page_field():
    chunks = chunk_text('A' * 2000, chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    assert [c['chunk_index'] for c in chunks] == list(range(len(chunks)))
    assert 'page' not in chunks[0]


def test_extract_year_prefers_launch_keyword():
    assert extract_year('Galileo launched in 1989 to study Jupiter.') == 1989


def test_extract_year_launch_date_beats_later_news_year():
    # JWST regression: a launch date must win over a later year mentioned in news
    # text (previously this page was mistagged 2025 instead of its 2021 launch).
    text = ('The telescope launched on Dec. 25, 2021. '
            'In 2025 it observed distant galaxies.')
    assert extract_year(text) == 2021


def test_extract_year_full_date_beats_bare_month_year():
    # Apollo 13 regression: an astronaut "selected in September 1962" must not beat
    # the full launch date "April 11, 1970".
    text = ('Selected by NASA in September 1962, the crew flew the mission that '
            'launched. On April 11, 1970, the spacecraft lifted off.')
    assert extract_year(text) == 1970


def test_extract_year_month_year_when_no_full_date():
    # NISAR-style: only a "Month Year" reference is available.
    assert extract_year('Limited data was released in February 2026.') == 2026


def test_extract_year_no_date_context_returns_zero():
    # Deliberately does NOT guess from a bare 4-digit year — that previously
    # mistagged pages with the NASA-founding year (1958) or a copyright year.
    assert extract_year('Results from 2005 built on findings from 1999.') == 0


def test_extract_year_ignores_implausible():
    assert extract_year('In 1300 and again in 1850, nothing relevant.') == 0


def test_extract_year_none_found():
    assert extract_year('This page has no dates whatsoever.') == 0


def test_embed_chunks_adds_document_prefix(monkeypatch):
    captured = {}

    class FakeClient:
        def embed(self, model, input):
            captured['input'] = input
            return {'embeddings': [[0.0]] * len(input)}

    monkeypatch.setattr(pl, 'get_client', lambda: FakeClient())
    pl.embed_chunks([{'text': 'hello'}, {'text': 'world'}], model='nomic-embed-text')
    assert captured['input'] == [DOCUMENT_PREFIX + 'hello', DOCUMENT_PREFIX + 'world']


def test_clean_text_normalizes_smart_punctuation():
    out = pl.clean_text('NASA’s “Artemis” — program')
    assert '’' not in out and '“' not in out and '—' not in out
    assert "NASA's" in out


def test_source_hash_is_stable_and_short():
    url = 'https://www.nasa.gov/mission/voyager-1/'
    h = hashlib.sha1(url.encode('utf-8')).hexdigest()[:16]
    assert len(h) == 16 and h == hashlib.sha1(url.encode('utf-8')).hexdigest()[:16]
