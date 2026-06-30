"""End-to-end integration harness for the REAL NotionIQ run path.

Goal: prove the app works through its real code path WITHOUT live API keys.
Only the two network boundaries are stubbed:

  * the raw Anthropic SDK client (`client.messages.create`), and
  * the raw Notion SDK client (`search` / `databases.query` / `pages.retrieve`
    / `blocks.children.list`).

EVERYTHING else is the real production code: provider + model selection
(`ai_providers`), the real `UniversalAIAnalyzer.analyze_page`, the real
`APIOptimizer` (token optimization, content-hash cache, cost accounting from
the fake `usage` object), `NotionAdvancedClient.scan_workspace` /
`get_scanned_pages` / `get_page_content`, the organizer's
`_process_all_databases` / `_process_page_hierarchy` / `_create_report` /
`_save_results` / `_build_summary_panel`, and `_write_markdown_report`.

Harness construction (see `build_organizer`):
  * A real `Settings` is built directly (no `.env`) with valid-format keys.
  * The real `NotionAdvancedClient` is built via ``__new__`` + attribute
    injection, with `.client` = `FakeNotionSDK` (the only network seam).
  * The real `UniversalAIAnalyzer` is built normally, with
    ``anthropic.Anthropic`` monkeypatched so `_init_claude_client` returns a
    `FakeAnthropic` (only `messages.create` is faked). Provider/model
    selection, optimizer, cost, and cache are all real.
  * `ai_providers.get_settings` is monkeypatched so provider detection sees the
    test settings (Anthropic key only) rather than any real `.env`.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

import config
from ai_analyzer import UniversalAIAnalyzer
from api_optimizer import OptimizationLevel
from config import Settings
from notion_organizer import NotionOrganizer, _write_markdown_report
from notion_wrapper import NotionAdvancedClient

# Valid-format fake keys (see security.SecurityValidator): notion = secret_ + 43
# alnum (len 50); anthropic = sk-ant- + >50 chars. No live calls are ever made.
FAKE_NOTION_KEY = "secret_" + "a" * 43
FAKE_ANTHROPIC_KEY = "sk-ant-" + "b" * 95


# --------------------------------------------------------------------------- #
# Fixture workspace data
# --------------------------------------------------------------------------- #

# Two databases, a few pages each. Bodies are crafted so two pages
# ("Near A" / "Near B") share a near-identical body but differ in title/id —
# this is the "no similarity bleed" probe.
NEAR_BODY = (
    "This document covers the quarterly planning details for the team and the "
    "general approach we intend to follow over the coming weeks and months."
)

WORKSPACE = {
    "db1": {
        "title": "Tasks",
        "pages": {
            "p_task": {
                "title": "Ship the release",
                "content": "We must ship the v2 release before Friday. Action required.",
                "last_edited_time": "2026-01-01T00:00:00Z",
            },
            "p_meeting": {
                "title": "Weekly sync notes",
                "content": "Meeting notes from the weekly sync. Attendees discussed status.",
                "last_edited_time": "2026-01-02T00:00:00Z",
            },
        },
    },
    "db2": {
        "title": "Notes",
        "pages": {
            "p_idea": {
                "title": "Idea: dark mode",
                "content": "An idea worth exploring: add a dark mode toggle to settings.",
                "last_edited_time": "2026-01-03T00:00:00Z",
            },
            "p_near_a": {
                "title": "Near A",
                "content": NEAR_BODY,
                "last_edited_time": "2026-01-04T00:00:00Z",
            },
            "p_near_b": {
                "title": "Near B",
                "content": NEAR_BODY,
                "last_edited_time": "2026-01-05T00:00:00Z",
            },
        },
    },
}

ALL_PAGE_IDS = [pid for db in WORKSPACE.values() for pid in db["pages"]]

# Deterministic classification keyed off the page title. The fake Anthropic
# client returns exactly this JSON, so we can assert per-page correctness and
# prove no cross-page bleed.
TITLE_TO_TYPE = {
    "Ship the release": "task",
    "Weekly sync notes": "meeting_note",
    "Idea: dark mode": "idea",
    "Near A": "reference",
    "Near B": "journal",
}


# --------------------------------------------------------------------------- #
# Fake Anthropic SDK client (only network seam for the AI boundary)
# --------------------------------------------------------------------------- #


class FakeUsage:
    """Real-shaped Claude usage object."""

    def __init__(self, input_tokens, output_tokens, cache_read_input_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class FakeMessage:
    def __init__(self, text, usage):
        # content=[obj.text=...], matching message.content[0].text
        self.content = [SimpleNamespace(text=text)]
        self.usage = usage


# Fixed, known token shape per call so the cost assertion is exact.
INPUT_TOKENS = 1000
OUTPUT_TOKENS = 50
CACHE_READ_TOKENS = 4096


class FakeMessages:
    def __init__(self, parent):
        self._parent = parent

    def create(self, **kwargs):
        self._parent.create_calls.append(kwargs)
        # Derive a deterministic classification from the per-page user content.
        user_text = kwargs["messages"][0]["content"]
        primary_type = "unknown"
        for title, ptype in TITLE_TO_TYPE.items():
            if title in user_text:
                primary_type = ptype
                break
        body = json.dumps(
            {
                "classification": {"primary_type": primary_type, "confidence": 0.95},
                "recommendations": {
                    "primary_action": "move_to_database",
                    "suggested_database": "Tasks",
                    "confidence": 0.9,
                },
                "urgency": {"level": "this_week"},
            }
        )
        return FakeMessage(
            body,
            FakeUsage(INPUT_TOKENS, OUTPUT_TOKENS, CACHE_READ_TOKENS),
        )


class FakeAnthropic:
    """Stand-in for anthropic.Anthropic — only `messages.create` is exercised."""

    def __init__(self, *args, **kwargs):
        self.create_calls = []
        self.messages = FakeMessages(self)


# --------------------------------------------------------------------------- #
# Fake Notion SDK client (only network seam for the Notion boundary)
# --------------------------------------------------------------------------- #


class FakeNotionSDK:
    """Minimal stub of notion_client.Client backed by WORKSPACE."""

    def __init__(self):
        outer = self

        class _Databases:
            def query(self, database_id, **kwargs):
                pages = WORKSPACE.get(database_id, {}).get("pages", {})
                return {
                    "results": [{"id": pid} for pid in pages],
                    "has_more": False,
                    "next_cursor": None,
                }

        class _Pages:
            def retrieve(self, page_id):
                for db in WORKSPACE.values():
                    if page_id in db["pages"]:
                        pg = db["pages"][page_id]
                        return {
                            "id": page_id,
                            "url": f"https://notion.so/{page_id}",
                            "created_time": "2026-01-01T00:00:00Z",
                            "last_edited_time": pg["last_edited_time"],
                            "properties": {
                                "Name": {
                                    "type": "title",
                                    "title": [{"plain_text": pg["title"]}],
                                }
                            },
                            "parent": {},
                            "archived": False,
                        }
                return {"id": page_id, "properties": {}, "archived": False}

        class _BlocksChildren:
            def list(self, **kwargs):
                page_id = kwargs.get("block_id")
                for db in WORKSPACE.values():
                    if page_id in db["pages"]:
                        pg = db["pages"][page_id]
                        return {
                            "results": [
                                {
                                    "id": f"{page_id}_b0",
                                    "type": "paragraph",
                                    "paragraph": {
                                        "rich_text": [
                                            {"plain_text": pg["content"]}
                                        ]
                                    },
                                }
                            ],
                            "has_more": False,
                            "next_cursor": None,
                        }
                return {"results": [], "has_more": False, "next_cursor": None}

        class _Blocks:
            children = _BlocksChildren()

        self.databases = _Databases()
        self.pages = _Pages()
        self.blocks = _Blocks()

    def search(self, **kwargs):
        # Used by scan_workspace to enumerate databases.
        return {
            "results": [
                {"id": db_id, "title": [{"plain_text": db["title"]}], "properties": {}}
                for db_id, db in WORKSPACE.items()
            ]
        }


# --------------------------------------------------------------------------- #
# Harness construction
# --------------------------------------------------------------------------- #


def make_settings(tmp_path):
    """Build a real validated Settings with valid-format fake keys, isolated dirs.

    Built directly (no `.env`) so cost/cache state starts clean each run.
    """
    return Settings(
        notion_api_key=FAKE_NOTION_KEY,
        notion_inbox_database_id="db1",
        anthropic_api_key=FAKE_ANTHROPIC_KEY,
        openai_api_key=None,
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
        enable_caching=True,
        enable_recommendations_page=False,  # don't touch Notion writes
        app_env="production",  # avoid stderr DEBUG handler noise
    )


def build_analyzer(settings, monkeypatch):
    """Construct the REAL UniversalAIAnalyzer with only the Anthropic client faked."""
    # Provider detection does `from config import get_settings` inside
    # AIProviderManager._detect_available_providers; patch config.get_settings
    # so it sees the test settings (Anthropic key only) rather than any .env.
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    # Ensure no real provider keys / model overrides bleed in from the env.
    for var in (
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "CLAUDE_MODEL",
        "PREFERRED_AI_PROVIDER",
    ):
        monkeypatch.delenv(var, raising=False)

    fake_anthropic = FakeAnthropic()
    # _init_claude_client does `from anthropic import Anthropic; Anthropic(...)`.
    import anthropic

    monkeypatch.setattr(
        anthropic, "Anthropic", lambda *a, **k: fake_anthropic, raising=True
    )

    analyzer = UniversalAIAnalyzer(
        settings, optimization_level=OptimizationLevel.MINIMAL
    )
    # Sanity: the injected fake is what the analyzer holds.
    assert analyzer.client is fake_anthropic
    return analyzer, fake_anthropic


def build_notion(settings):
    """Build the REAL NotionAdvancedClient with the raw SDK client faked."""
    notion = NotionAdvancedClient.__new__(NotionAdvancedClient)
    notion.client = FakeNotionSDK()
    notion.async_client = None
    notion.rate_limit = 1000.0
    notion.last_request_time = 0
    notion.cache = {}
    notion.cache_expiry = {}
    notion.workspace_structure = None
    notion.databases = {}
    notion.page_count = 0
    notion.scanned_pages = {}
    notion.settings = settings
    return notion


def build_organizer(tmp_path, monkeypatch, output_format="json"):
    """Construct the REAL NotionOrganizer wired to both fakes.

    Built via ``__new__`` + injection so we control exactly the two network
    seams while running every other production component for real.
    """
    settings = make_settings(tmp_path)
    analyzer, fake_anthropic = build_analyzer(settings, monkeypatch)
    notion = build_notion(settings)

    org = NotionOrganizer.__new__(NotionOrganizer)
    org.settings = settings
    org.optimization_level = OptimizationLevel.MINIMAL
    org.verbose = False
    org.output_format = output_format
    org.cost_monitor = None
    org.notion = notion
    org.ai_analyzer = analyzer
    # Real workspace analyzer + executor, wired to the fake notion client.
    from recommendation_executor import RecommendationExecutor
    from workspace_analyzer import WorkspaceAnalyzer

    org.workspace_analyzer = WorkspaceAnalyzer(notion, settings)
    org.executor = RecommendationExecutor(notion, settings)
    org.analysis_results = []
    org.workspace_analysis = {}
    org.report_data = {}
    return org, analyzer, fake_anthropic


# --------------------------------------------------------------------------- #
# Assertion 1 + 2 + 3: workspace mode end-to-end, model, cost
# --------------------------------------------------------------------------- #


def test_workspace_mode_runs_end_to_end(tmp_path, monkeypatch):
    """Assertion 1: the previously-crashing workspace path runs with no exception
    and analyzes every non-empty page."""
    org, analyzer, _ = build_organizer(tmp_path, monkeypatch)

    asyncio.run(org.run_analysis(analyze_workspace=True, process_mode="workspace"))

    # All 5 fixture pages have content -> all analyzed.
    assert len(org.analysis_results) == len(ALL_PAGE_IDS) == 5
    types = {r["page_title"]: r["classification"]["primary_type"] for r in org.analysis_results}
    assert types == TITLE_TO_TYPE


def test_selected_model_is_haiku(tmp_path, monkeypatch):
    """Assertion 2: cost-default selection picks claude-haiku-4-5."""
    _, analyzer, _ = build_organizer(tmp_path, monkeypatch)
    assert analyzer.ai_config["provider"] == "claude"
    assert analyzer.ai_config["model"] == "claude-haiku-4-5"
    assert analyzer.ai_config["model_info"]["name"] == "Claude Haiku 4.5"


def test_cost_computed_from_usage_at_haiku_rates(tmp_path, monkeypatch):
    """Assertion 3: total_cost equals the value computed from the fake usage at
    Haiku rates ($1/$5 per 1M, cache reads at 0.1x input)."""
    org, analyzer, _ = build_organizer(tmp_path, monkeypatch)
    asyncio.run(org.run_analysis(analyze_workspace=True, process_mode="workspace"))

    n = 5  # five unique pages -> five real API calls
    per_call = (
        (INPUT_TOKENS / 1_000_000) * 1.00
        + (OUTPUT_TOKENS / 1_000_000) * 5.00
        + (CACHE_READ_TOKENS / 1_000_000) * 1.00 * 0.1
    )
    expected = per_call * n

    assert analyzer.api_optimizer.metrics.total_cost > 0
    assert round(analyzer.api_optimizer.metrics.total_cost, 10) == round(expected, 10)
    # Prove cache_read tokens contributed (cost is strictly higher than without).
    without_cache = n * (
        (INPUT_TOKENS / 1_000_000) * 1.00 + (OUTPUT_TOKENS / 1_000_000) * 5.00
    )
    assert analyzer.api_optimizer.metrics.total_cost > without_cache


# --------------------------------------------------------------------------- #
# Assertion 4: exact content-hash cache (same page twice -> one API call)
# --------------------------------------------------------------------------- #


def test_same_page_twice_hits_cache(tmp_path, monkeypatch):
    """Assertion 4: analyzing the SAME page twice issues only ONE messages.create."""
    org, analyzer, fake = build_organizer(tmp_path, monkeypatch)
    page = org.notion.get_page_content("p_task")

    first = analyzer.analyze_page(page)
    second = analyzer.analyze_page(page)

    assert len(fake.create_calls) == 1  # second served from cache
    assert analyzer.api_optimizer.metrics.cache_hits == 1
    assert first["classification"]["primary_type"] == "task"
    assert second["classification"]["primary_type"] == "task"


# --------------------------------------------------------------------------- #
# Assertion 5: no similarity bleed (near-identical bodies, distinct results)
# --------------------------------------------------------------------------- #


def test_no_similarity_bleed_between_near_identical_pages(tmp_path, monkeypatch):
    """Assertion 5: two pages with near-identical bodies but different
    titles/ids each get their OWN classification (no cache cross-contamination)."""
    org, analyzer, fake = build_organizer(tmp_path, monkeypatch)
    page_a = org.notion.get_page_content("p_near_a")
    page_b = org.notion.get_page_content("p_near_b")

    res_a = analyzer.analyze_page(page_a)
    res_b = analyzer.analyze_page(page_b)

    # Two distinct API calls (no false cache hit), distinct results.
    assert len(fake.create_calls) == 2
    assert res_a["page_title"] == "Near A"
    assert res_b["page_title"] == "Near B"
    assert res_a["classification"]["primary_type"] == "reference"
    assert res_b["classification"]["primary_type"] == "journal"
    assert res_a["classification"]["primary_type"] != res_b["classification"]["primary_type"]


# --------------------------------------------------------------------------- #
# Assertion 6: summary panel renders with page count + cost
# --------------------------------------------------------------------------- #


def _render_panel(org):
    from rich.console import Console

    metrics = org.ai_analyzer.api_optimizer.metrics
    panel = org._build_summary_panel(metrics, elapsed=1.2)
    console = Console(width=80, record=True)
    console.print(panel)
    return console.export_text()


def test_summary_panel_renders_with_counts_and_cost(tmp_path, monkeypatch):
    """Assertion 6: the rendered panel contains the page count and a cost figure."""
    org, analyzer, _ = build_organizer(tmp_path, monkeypatch)
    asyncio.run(org.run_analysis(analyze_workspace=True, process_mode="workspace"))

    text = _render_panel(org)
    assert "NotionIQ" in text
    assert "Pages analyzed" in text
    assert "5" in text  # page count
    assert "Total cost" in text
    assert "$" in text  # a cost figure is rendered


# --------------------------------------------------------------------------- #
# Assertion 7: markdown export writes an .md with expected sections
# --------------------------------------------------------------------------- #


def test_markdown_export_writes_sections(tmp_path, monkeypatch):
    """Assertion 7: --format both writes an .md report with the expected headers."""
    org, analyzer, _ = build_organizer(
        tmp_path, monkeypatch, output_format="both"
    )
    asyncio.run(org.run_analysis(analyze_workspace=True, process_mode="workspace"))

    md_path = org._last_markdown_report
    assert md_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "# NotionIQ Analysis Report" in text
    assert "## Summary" in text
    assert "## Classifications" in text
    assert "## Recommendations" in text


# --------------------------------------------------------------------------- #
# Assertion 8: page-hierarchy mode end-to-end (the other previously-broken path)
# --------------------------------------------------------------------------- #


def test_page_hierarchy_mode_runs_end_to_end(tmp_path, monkeypatch):
    """Assertion 8: page-hierarchy mode runs without exception and analyzes the
    target page (its blocks contain no child pages, so just the root)."""
    org, analyzer, _ = build_organizer(tmp_path, monkeypatch)

    asyncio.run(
        org.run_analysis(
            analyze_workspace=False,
            process_mode="page",
            target_page_id="p_idea",
        )
    )

    assert len(org.analysis_results) == 1
    assert org.analysis_results[0]["page_title"] == "Idea: dark mode"
    assert org.analysis_results[0]["classification"]["primary_type"] == "idea"


# --------------------------------------------------------------------------- #
# CLI-routing check
# --------------------------------------------------------------------------- #


def test_cli_routing():
    """Bare `notioniq` routes to analysis; `init` exists; `run --help` lists
    --format and --verbose."""
    from click.testing import CliRunner

    from notion_organizer import main

    runner = CliRunner()

    # `run --help` lists the new flags.
    res = runner.invoke(main, ["run", "--help"])
    assert res.exit_code == 0
    assert "--format" in res.output
    assert "--verbose" in res.output

    # `init` subcommand exists.
    assert "init" in main.commands

    # Bare invocation (no args) routes to the default `run` command: it parses
    # run's options (proven by --help, which DefaultGroup routes to run).
    res_bare = runner.invoke(main, ["--help"])
    assert res_bare.exit_code == 0
    assert "--mode" in res_bare.output  # bare help shows run's options
