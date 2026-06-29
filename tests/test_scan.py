"""Regression test for the workspace-scan double-fetch (Task 10).

Before the fix, ``scan_workspace`` paginated full ``databases.query`` payloads
purely to *count* pages, then ``_process_all_databases`` re-issued
``databases.query`` for the same databases to actually list the pages — a
redundant network round-trip per database.

The fix retains the page payloads fetched during the scan and exposes them via
``get_scanned_pages`` so the analysis path reuses them instead of re-querying.
These tests stub the Notion client and *count* ``databases.query`` calls to
prove each database's page list is fetched at most once across scan + analyze.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from notion_wrapper import NotionAdvancedClient


class CountingNotionClient:
    """Minimal stub of ``notion_client.Client`` that counts calls.

    Models a fixed two-database workspace, each with its own pages. Tracks how
    many times each page is listed (via ``databases.query``) and retrieved
    (via ``pages.retrieve`` / ``blocks.children.list``).
    """

    def __init__(self):
        # Fixture workspace: two databases, three pages total.
        self._db_pages = {
            "db1": [
                {"id": "p1", "properties": {}, "last_edited_time": "2026-01-01T00:00:00Z"},
                {"id": "p2", "properties": {}, "last_edited_time": "2026-01-02T00:00:00Z"},
            ],
            "db2": [
                {"id": "p3", "properties": {}, "last_edited_time": "2026-01-03T00:00:00Z"},
            ],
        }
        # Call counters keyed by entity id.
        self.query_count = {}  # db_id -> times listed via databases.query
        self.retrieve_count = {}  # page_id -> times pages.retrieve called
        self.blocks_count = {}  # page_id -> times blocks.children.list called

        outer = self

        class _Databases:
            def query(self, database_id, **kwargs):
                outer.query_count[database_id] = (
                    outer.query_count.get(database_id, 0) + 1
                )
                return {
                    "results": list(outer._db_pages.get(database_id, [])),
                    "has_more": False,
                    "next_cursor": None,
                }

        class _PagesRetrieve:
            def retrieve(self, page_id):
                outer.retrieve_count[page_id] = (
                    outer.retrieve_count.get(page_id, 0) + 1
                )
                return {
                    "id": page_id,
                    "url": "",
                    "created_time": "2026-01-01T00:00:00Z",
                    "last_edited_time": "2026-01-01T00:00:00Z",
                    "properties": {},
                    "parent": {},
                    "archived": False,
                }

        class _BlocksChildren:
            def list(self, **kwargs):
                page_id = kwargs.get("block_id")
                outer.blocks_count[page_id] = outer.blocks_count.get(page_id, 0) + 1
                return {"results": [], "has_more": False, "next_cursor": None}

        class _Blocks:
            children = _BlocksChildren()

        self.databases = _Databases()
        self.pages = _PagesRetrieve()
        self.blocks = _Blocks()

    def search(self, **kwargs):
        return {
            "results": [
                {"id": "db1", "title": [{"plain_text": "Tasks"}], "properties": {}},
                {"id": "db2", "title": [{"plain_text": "Notes"}], "properties": {}},
            ]
        }


@pytest.fixture
def client(tmp_path):
    """Build a NotionAdvancedClient without __init__ side effects (no network)."""
    obj = NotionAdvancedClient.__new__(NotionAdvancedClient)
    obj.client = CountingNotionClient()
    obj.async_client = None
    obj.rate_limit = 1000.0
    obj.last_request_time = 0
    obj.cache = {}
    obj.cache_expiry = {}
    obj.workspace_structure = None
    obj.databases = {}
    obj.page_count = 0
    obj.settings = SimpleNamespace(
        data_dir=tmp_path,
        enable_caching=False,
        cache_ttl=SimpleNamespace(total_seconds=lambda: 0),
    )
    # Avoid touching disk during scan.
    obj._save_workspace_structure = MagicMock()
    return obj


@pytest.mark.asyncio
async def test_scan_counts_pages_without_redundant_query(client):
    """The scan should report correct counts and list each database once."""
    result = await client.scan_workspace()

    assert result["total_pages"] == 3
    assert result["databases"]["db1"]["page_count"] == 2
    assert result["databases"]["db2"]["page_count"] == 1
    # Scan lists each database exactly once.
    assert client.client.query_count == {"db1": 1, "db2": 1}


@pytest.mark.asyncio
async def test_each_page_listed_at_most_once_across_scan_and_analyze(client):
    """The redundant re-query is gone: each db is listed once total.

    Scan fetches the page payloads; the analysis path reuses them via
    ``get_scanned_pages`` instead of issuing a second ``databases.query``.
    """
    await client.scan_workspace()

    # Simulate the downstream analysis consumption for each database.
    total_pages_seen = 0
    for db_id in client.workspace_structure["databases"]:
        pages = client.get_scanned_pages(db_id)
        assert pages is not None, "scan must retain page payloads for reuse"
        for page in pages:
            # Per-page content fetch (blocks) is legitimate, not the double-fetch.
            client.get_page_content(page["id"])
            total_pages_seen += 1

    assert total_pages_seen == 3
    # The key assertion: each database's page list was fetched at most ONCE
    # across scan + analyze (no second databases.query).
    assert client.client.query_count == {"db1": 1, "db2": 1}
    # Each page's content is retrieved at most once.
    assert all(c <= 1 for c in client.client.retrieve_count.values())
    assert set(client.client.retrieve_count) == {"p1", "p2", "p3"}


@pytest.mark.asyncio
async def test_get_scanned_pages_applies_limit(client):
    """Reused payloads must honor the same limit slice the old path applied."""
    await client.scan_workspace()
    limited = client.get_scanned_pages("db1", limit=1)
    assert len(limited) == 1
    assert limited[0]["id"] == "p1"
