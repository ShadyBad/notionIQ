"""Tests for cache-key invalidation on ``last_edited_time`` and ``prompt_version``.

The exact-content-hash cache must invalidate when a page's ``last_edited_time``
changes (stale analysis of an edited page) and when ``settings.prompt_version``
changes (a taxonomy/prompt edit must invalidate every cached analysis).
"""

from types import SimpleNamespace

import pytest

from ai_analyzer import UniversalAIAnalyzer


@pytest.fixture
def analyzer():
    """Build an analyzer without __init__ side effects (no network/client setup)."""
    obj = UniversalAIAnalyzer.__new__(UniversalAIAnalyzer)
    obj.settings = SimpleNamespace(prompt_version="v1")
    return obj


def test_cache_key_changes_with_last_edited_time(analyzer):
    base = {
        "title": "X",
        "content": "hello",
        "properties": {},
        "last_edited_time": "2026-01-01T00:00:00Z",
    }
    edited = {**base, "last_edited_time": "2026-06-01T00:00:00Z"}
    assert analyzer._generate_cache_key(base) != analyzer._generate_cache_key(edited)


def test_cache_key_changes_with_prompt_version(analyzer, monkeypatch):
    page = {
        "title": "X",
        "content": "hello",
        "properties": {},
        "last_edited_time": "2026-01-01T00:00:00Z",
    }
    k1 = analyzer._generate_cache_key(page)
    monkeypatch.setattr(analyzer.settings, "prompt_version", "v2")
    assert analyzer._generate_cache_key(page) != k1
