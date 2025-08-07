"""
Pytest configuration and shared fixtures for NotionIQ tests
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import json
from datetime import datetime, timezone
import tempfile
import os

from config import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        return Settings(
            notion_api_key="test_notion_key",
            notion_inbox_database_id="test_db_id",
            anthropic_api_key="test_anthropic_key",
            claude_model="claude-3-opus-20240229",
            app_env="development",
            log_level="DEBUG",
            output_dir=Path(tmpdir) / "output",
            data_dir=Path(tmpdir) / "data",
            batch_size=5,
            max_content_length=1000,
            enable_caching=True,
            cache_ttl_hours=1,
            rate_limit_requests_per_second=10.0,
            enable_encryption=False,
            enable_workspace_scan=True,
            enable_pattern_learning=True,
            enable_auto_organization=False,
            enable_recommendations_page=True,
            enable_analytics=True,
            track_time_saved=True,
            track_accuracy=True
        )


@pytest.fixture
def mock_notion_client(mock_settings):
    """Create mock Notion client"""
    with patch('notion_client.Client') as mock_client_class:
        with patch('notion_client.AsyncClient') as mock_async_client_class:
            from notion_client import NotionAdvancedClient
            client = NotionAdvancedClient(mock_settings)
            
            # Mock common methods
            client.get_database = MagicMock(return_value={
                "id": "test_db_id",
                "title": [{"plain_text": "Test Database"}],
                "properties": {}
            })
            
            client.get_database_pages = MagicMock(return_value=[
                {
                    "id": "page1",
                    "properties": {},
                    "created_time": "2024-01-01T00:00:00Z"
                },
                {
                    "id": "page2",
                    "properties": {},
                    "created_time": "2024-01-02T00:00:00Z"
                }
            ])
            
            client.get_page_content = MagicMock(return_value={
                "id": "page1",
                "title": "Test Page",
                "content": "Test content",
                "properties": {},
                "created_time": "2024-01-01T00:00:00Z",
                "last_edited_time": "2024-01-01T00:00:00Z",
                "url": "https://notion.so/test",
                "archived": False,
                "blocks": []
            })
            
            return client


@pytest.fixture
def mock_claude_analyzer(mock_settings):
    """Create mock Claude analyzer"""
    with patch('claude_analyzer.Anthropic') as mock_anthropic:
        from claude_analyzer import ClaudeAnalyzer
        analyzer = ClaudeAnalyzer(mock_settings)
        
        # Mock analyze_page to return structured analysis
        analyzer.analyze_page = MagicMock(return_value={
            "page_id": "page1",
            "page_title": "Test Page",
            "page_url": "https://notion.so/test",
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "classification": {
                "primary_type": "task",
                "confidence": 0.85,
                "reasoning": "Contains action items",
                "secondary_types": []
            },
            "content_analysis": {
                "summary": "Test summary",
                "key_topics": ["testing", "development"],
                "sentiment": "neutral",
                "completeness": "complete",
                "information_density": "medium",
                "actionable_items": 2
            },
            "urgency_assessment": {
                "level": "this_week",
                "confidence": 0.7,
                "detected_deadline": None,
                "reasoning": "No specific deadline mentioned"
            },
            "context_detection": {
                "primary_context": "work",
                "confidence": 0.8,
                "detected_project": None,
                "detected_people": [],
                "detected_department": None
            },
            "recommendations": {
                "primary_action": "move_to_database",
                "confidence": 0.75,
                "suggested_database": "Tasks",
                "reasoning": "Page contains actionable items",
                "additional_actions": []
            },
            "organization_suggestions": {
                "suggested_title": "Test Page",
                "suggested_tags": ["testing"],
                "suggested_properties": {},
                "suggested_relationships": []
            },
            "quality_assessment": {
                "organization_score": 75,
                "factors": {
                    "has_clear_title": True,
                    "has_meaningful_content": True,
                    "properly_tagged": False,
                    "has_clear_purpose": True,
                    "well_structured": True
                },
                "improvement_potential": "low"
            }
        })
        
        return analyzer


@pytest.fixture
def sample_page_content():
    """Sample page content for testing"""
    return {
        "id": "test-page-123",
        "title": "Q4 Planning Meeting Notes",
        "content": """
        Meeting Date: October 15, 2024
        Attendees: John, Sarah, Mike
        
        Agenda:
        1. Review Q3 results
        2. Set Q4 objectives
        3. Resource allocation
        
        Key Decisions:
        - Increase marketing budget by 20%
        - Hire 2 new engineers
        - Launch new product feature by November
        
        Action Items:
        - John: Prepare budget proposal
        - Sarah: Start recruitment process
        - Mike: Finalize feature specifications
        """,
        "properties": {
            "Status": {"type": "select", "select": {"name": "In Progress"}},
            "Priority": {"type": "select", "select": {"name": "High"}}
        },
        "created_time": "2024-10-15T10:00:00Z",
        "last_edited_time": "2024-10-15T14:30:00Z",
        "url": "https://notion.so/test-page",
        "archived": False,
        "parent": {"database_id": "test_db_id"},
        "blocks": []
    }


@pytest.fixture
def sample_workspace_structure():
    """Sample workspace structure for testing"""
    return {
        "databases": {
            "db1": {
                "title": "Tasks",
                "properties": {
                    "Status": {"type": "select", "id": "prop1"},
                    "Priority": {"type": "select", "id": "prop2"},
                    "Due Date": {"type": "date", "id": "prop3"}
                },
                "created_time": "2024-01-01T00:00:00Z",
                "last_edited_time": "2024-10-01T00:00:00Z",
                "page_count": 25
            },
            "db2": {
                "title": "Projects",
                "properties": {
                    "Status": {"type": "status", "id": "prop4"},
                    "Owner": {"type": "people", "id": "prop5"},
                    "Tasks": {"type": "relation", "id": "prop6", "database_id": "db1"}
                },
                "created_time": "2024-01-01T00:00:00Z",
                "last_edited_time": "2024-10-01T00:00:00Z",
                "page_count": 10
            },
            "db3": {
                "title": "Inbox",
                "properties": {
                    "Name": {"type": "title", "id": "prop7"}
                },
                "created_time": "2024-01-01T00:00:00Z",
                "last_edited_time": "2024-10-01T00:00:00Z",
                "page_count": 5
            }
        },
        "total_pages": 40,
        "relationships": [
            {
                "from_database": "db2",
                "from_database_title": "Projects",
                "to_database": "db1",
                "to_database_title": "Tasks",
                "property_name": "Tasks",
                "type": "relation"
            }
        ],
        "scan_timestamp": "2024-10-15T12:00:00Z"
    }


@pytest.fixture
def mock_anthropic_response():
    """Mock response from Anthropic API"""
    return {
        "classification": {
            "primary_type": "meeting_note",
            "confidence": 0.9,
            "reasoning": "Contains meeting agenda, attendees, and action items",
            "secondary_types": ["project", "task"]
        },
        "content_analysis": {
            "summary": "Q4 planning meeting discussing budget, hiring, and product launch",
            "key_topics": ["planning", "budget", "hiring", "product"],
            "sentiment": "positive",
            "completeness": "complete",
            "information_density": "high",
            "actionable_items": 3
        },
        "urgency_assessment": {
            "level": "this_month",
            "confidence": 0.8,
            "detected_deadline": "November",
            "reasoning": "Product launch deadline in November"
        },
        "recommendations": {
            "primary_action": "move_to_database",
            "confidence": 0.85,
            "suggested_database": "Meeting Notes",
            "reasoning": "Well-structured meeting notes with clear action items"
        }
    }


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()