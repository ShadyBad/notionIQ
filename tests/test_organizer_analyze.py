"""Regression tests for the organizer's page-analysis paths.

These paths previously called a non-existent `self.api_optimizer`, passed two
positional args to a one-arg `optimize_page_content`, and `await`-ed the SYNC
`analyze_page` — so a real `notioniq` run (default workspace mode) crashed.
The fix: pass the full `get_page_content` dict straight to `analyze_page`
(which optimizes internally), called synchronously.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from notion_organizer import NotionOrganizer


def _make_organizer(content_dict):
    org = NotionOrganizer.__new__(NotionOrganizer)
    org.analysis_results = []
    org.verbose = False
    org.workspace_analysis = None
    org.notion = SimpleNamespace(
        client=SimpleNamespace(
            pages=SimpleNamespace(retrieve=lambda pid: {"id": pid})
        ),
        get_page_content=lambda pid: content_dict,
        get_page_blocks=lambda pid: [],
    )
    # analyze_page is SYNC and returns a dict (not a coroutine).
    analyze = MagicMock(return_value={"classification": {"primary_type": "Note"}})
    org.ai_analyzer = SimpleNamespace(analyze_page=analyze)
    return org, analyze


def test_hierarchy_passes_full_content_dict_to_analyze_page():
    content = {"id": "p1", "title": "T", "content": "hello body", "properties": {}}
    org, analyze = _make_organizer(content)

    # Must not raise AttributeError (self.api_optimizer) or TypeError (await dict).
    asyncio.run(org._process_page_hierarchy("p1"))

    analyze.assert_called_once_with(content)
    assert org.analysis_results == [{"classification": {"primary_type": "Note"}}]


def test_hierarchy_skips_pages_with_no_content():
    content = {"id": "p2", "title": "Empty", "content": "", "properties": {}}
    org, analyze = _make_organizer(content)

    asyncio.run(org._process_page_hierarchy("p2"))

    analyze.assert_not_called()
    assert org.analysis_results == []
