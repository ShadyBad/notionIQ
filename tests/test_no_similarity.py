"""Regression test: the similarity cache must never serve another page's analysis.

NotionIQ previously consulted a ``SmartCache`` that returned a *similar* page's
cached classification (Jaccard similarity > 0.85 on coarse fingerprints). Two
distinct pages with near-identical body text but different titles would collide:
page B would receive page A's classification. That is a correctness bug — each
page must get its own analysis.

This test builds two pages whose body text is >= 0.85 Jaccard-similar but whose
titles differ, stubs the Claude client so the returned classification is keyed
off the page title, and asserts each page gets its OWN classification. With the
similarity cache live, page B inherits page A's result and this test fails.
"""

import json
from types import SimpleNamespace

from ai_analyzer import UniversalAIAnalyzer
from api_optimizer import APIOptimizer, OptimizationLevel


class _TitleKeyedMessages:
    """Stub ``client.messages`` whose classification is derived from the title.

    The page title is carried in ``user_text`` as ``Page Title: <title>``. The
    stub parses it out and returns a classification ``primary_type`` equal to the
    title, so a cross-page cache hit is detectable: page B's result would carry
    page A's title-derived type.
    """

    def __init__(self):
        self.create_calls = 0

    def create(self, **kwargs):
        self.create_calls += 1
        user_text = kwargs["messages"][0]["content"]
        title = "UNKNOWN"
        for line in user_text.splitlines():
            if line.startswith("Page Title:"):
                title = line.split("Page Title:", 1)[1].strip()
                break
        payload = {
            "classification": {"primary_type": title, "confidence": 1.0},
            "recommendations": {"primary_action": "review", "confidence": 1.0},
        }
        usage = SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=0
        )
        return SimpleNamespace(
            content=[SimpleNamespace(text=json.dumps(payload))],
            usage=usage,
        )


def _make_analyzer(tmp_path):
    """Build an analyzer without __init__ side effects, with a REAL APIOptimizer."""
    analyzer = UniversalAIAnalyzer.__new__(UniversalAIAnalyzer)
    # BALANCED avoids minimal-mode prompt optimization (which would mangle the
    # title in user_text) and minimal-mode template skipping.
    analyzer.optimization_level = OptimizationLevel.BALANCED
    analyzer.provider_type = "claude"
    analyzer.ai_config = {
        "model": "claude-haiku-4-5",
        "model_info": {"name": "Claude Haiku 4.5"},
    }
    analyzer.client = SimpleNamespace(messages=_TitleKeyedMessages())

    settings = SimpleNamespace(
        data_dir=tmp_path,
        enable_caching=True,
        cache_ttl_hours=24,
        max_content_length=2000,
    )
    analyzer.settings = settings
    analyzer.response_cache = {}
    analyzer.api_optimizer = APIOptimizer(
        settings,
        OptimizationLevel.BALANCED,
        input_cost_per_1m=1.00,
        output_cost_per_1m=5.00,
    )
    return analyzer


def test_similar_pages_get_their_own_classifications(tmp_path):
    analyzer = _make_analyzer(tmp_path)

    # Identical body text -> Jaccard similarity well above the 0.85 threshold
    # (only the short title/id tokens differ), but the titles differ so the two
    # pages must NOT share a classification.
    shared_body = (
        "We discussed the quarterly roadmap and the upcoming product launch in "
        "great detail. The team reviewed all open tasks, assigned clear owners, "
        "and set concrete target dates for the next planning cycle as well as "
        "the follow up review meeting scheduled afterward. Notes captured action "
        "items, blockers, dependencies, risks, and the agreed communication plan."
    )
    page_a = {
        "id": "page-a",
        "title": "Alpha",
        "content": shared_body,
        "properties": {},
        "last_edited_time": "2026-01-01T00:00:00Z",
    }
    page_b = {
        "id": "page-b",
        "title": "Bravo",
        "content": shared_body,
        "properties": {},
        "last_edited_time": "2026-01-01T00:00:00Z",
    }

    result_a = analyzer.analyze_page(page_a)
    result_b = analyzer.analyze_page(page_b)

    # Each page must receive its OWN title-keyed classification. With the
    # similarity cache active, page B inherits page A's "Alpha" type.
    assert result_a["classification"]["primary_type"] == "Alpha"
    assert result_b["classification"]["primary_type"] == "Bravo"

    # Both pages required a real (stubbed) API call; B was not served from a
    # similarity hit.
    assert analyzer.client.messages.create_calls == 2
