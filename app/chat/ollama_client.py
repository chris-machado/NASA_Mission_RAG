"""Shared Ollama client factory.

The top-level ``ollama.embed`` / ``ollama.chat`` helpers ignore ``OLLAMA_BASE_URL``
and have no request timeout, so a stalled Ollama would hang a worker indefinitely.
Routing every call through a configured ``ollama.Client`` fixes both.
"""

import os

import ollama

DEFAULT_BASE_URL = 'http://127.0.0.1:11434'
DEFAULT_TIMEOUT = 120.0


def make_client(base_url=None, timeout=DEFAULT_TIMEOUT):
    """Build an Ollama client bound to ``base_url`` with a read/connect timeout.

    The timeout is per-read, so streaming generations are unaffected as long as
    tokens keep arriving; it only trips when Ollama is genuinely unresponsive.
    """
    base_url = base_url or os.environ.get('OLLAMA_BASE_URL', DEFAULT_BASE_URL)
    return ollama.Client(host=base_url, timeout=timeout)


def get_client():
    """Return a client configured from the active Flask app, or from the
    environment when there is no app context (e.g. the ingest script)."""
    try:
        from flask import current_app

        return make_client(
            current_app.config.get('OLLAMA_BASE_URL'),
            current_app.config.get('OLLAMA_TIMEOUT', DEFAULT_TIMEOUT),
        )
    except (RuntimeError, ImportError):
        return make_client()
