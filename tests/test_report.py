"""Tests for the Markdown report writer (`--format markdown`)."""

import notion_organizer as no


def _sample_report():
    """A report dict shaped like NotionOrganizer._create_report() output."""
    return {
        "metadata": {"generated_at": "2026-06-28T12:00:00+00:00", "version": "1.0.0"},
        "workspace_summary": {
            "total_databases": 3,
            "total_pages_analyzed": 7,
            "health_score": 72.5,
        },
        "classification_summary": {
            "document_types": {"note": 4, "task": 2, "meeting_notes": 1},
        },
        "recommendations": {
            "suggested_moves": [
                {
                    "title": "Q3 Planning",
                    "action": "move_to_database",
                    "confidence": 0.92,
                    "suggested_database": "Projects",
                },
                {
                    "title": "Random idea",
                    "action": "move_to_database",
                    "confidence": 0.71,
                    "suggested_database": "Ideas",
                },
            ],
            "archive_candidates": [
                {"title": "Old meeting", "action": "archive", "confidence": 0.85},
            ],
            "delete_candidates": [],
        },
        "insights": [
            {
                "type": "pattern",
                "title": "Most Common Content Type",
                "description": "Your workspace primarily contains note items (4 pages)",
            }
        ],
    }


def test_write_markdown_report_creates_file(tmp_path):
    path = no._write_markdown_report(
        _sample_report(), tmp_path, timestamp="20260628_120000"
    )
    assert path.exists()
    assert path.name == "analysis_report_20260628_120000.md"


def test_write_markdown_report_contains_expected_sections(tmp_path):
    path = no._write_markdown_report(
        _sample_report(), tmp_path, timestamp="20260628_120000"
    )
    text = path.read_text(encoding="utf-8")

    # Title + headers
    assert "# NotionIQ Analysis Report" in text
    assert "## Summary" in text
    assert "## Classifications" in text
    assert "## Recommendations" in text
    assert "## Top Actions" in text
    assert "## Key Insights" in text

    # Summary counts rendered
    assert "| Pages analyzed | 7 |" in text
    assert "| Databases | 3 |" in text
    assert "72.5/100" in text

    # Classification rows (titles cased, underscores -> spaces)
    assert "| Note | 4 |" in text
    assert "| Meeting Notes | 1 |" in text

    # Per-category recommendations table rows
    assert "| Suggested Moves | 2 |" in text
    assert "| Archive Candidates | 1 |" in text

    # Top actions: highest-confidence item first, with target + percentage
    assert "Q3 Planning" in text
    assert "(to Projects)" in text
    assert "confidence 92%" in text

    # Insight rendered
    assert "Most Common Content Type" in text


def test_write_markdown_report_creates_output_dir(tmp_path):
    nested = tmp_path / "out" / "reports"
    path = no._write_markdown_report(_sample_report(), nested)
    assert path.exists()
    assert path.parent == nested
