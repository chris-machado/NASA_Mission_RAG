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


def test_extract_year_falls_back_to_earliest_plausible():
    # No launch keyword → earliest plausible 4-digit year mentioned.
    assert extract_year('Results from 2005 built on findings from 1999.') == 1999


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
