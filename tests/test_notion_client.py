"""
Tests for Notion client module
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, call
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

from notion_client import NotionAdvancedClient


class TestNotionAdvancedClient:
    """Test NotionAdvancedClient class"""
    
    def test_initialization(self, mock_settings):
        """Test client initialization"""
        with patch('notion_client.Client') as mock_client:
            with patch('notion_client.AsyncClient') as mock_async_client:
                client = NotionAdvancedClient(mock_settings)
                
                assert client.settings == mock_settings
                assert client.rate_limit == mock_settings.rate_limit_requests_per_second
                assert client.cache == {}
                assert client.cache_expiry == {}
                mock_client.assert_called_once_with(auth=mock_settings.notion_api_key)
                mock_async_client.assert_called_once_with(auth=mock_settings.notion_api_key)
    
    def test_rate_limiting(self, mock_notion_client):
        """Test rate limiting implementation"""
        mock_notion_client.rate_limit = 10.0  # 10 requests per second
        mock_notion_client.last_request_time = 0
        
        with patch('time.time', side_effect=[1.0, 1.05, 1.1]):
            with patch('time.sleep') as mock_sleep:
                # First request - no wait
                mock_notion_client._rate_limit_wait()
                mock_sleep.assert_not_called()
                
                # Second request - should wait
                mock_notion_client.last_request_time = 1.0
                mock_notion_client._rate_limit_wait()
                mock_sleep.assert_called_once()
                
                # Calculate expected sleep time
                expected_sleep = 0.1 - 0.05  # min_interval - time_since_last
                actual_sleep = mock_sleep.call_args[0][0]
                assert abs(actual_sleep - expected_sleep) < 0.01
    
    def test_get_database_with_retry(self, mock_notion_client):
        """Test database retrieval with retry logic"""
        mock_db = {
            "id": "test_db_id",
            "title": [{"plain_text": "Test Database"}],
            "properties": {}
        }
        mock_notion_client.client.databases.retrieve.return_value = mock_db
        
        result = mock_notion_client.get_database("test_db_id")
        
        assert result == mock_db
        mock_notion_client.client.databases.retrieve.assert_called_with("test_db_id")
    
    def test_get_database_pages_with_pagination(self, mock_notion_client):
        """Test getting pages with pagination"""
        # Mock paginated responses
        page1 = {"id": "page1", "properties": {}}
        page2 = {"id": "page2", "properties": {}}
        page3 = {"id": "page3", "properties": {}}
        
        mock_notion_client.client.databases.query.side_effect = [
            {"results": [page1, page2], "has_more": True, "next_cursor": "cursor1"},
            {"results": [page3], "has_more": False, "next_cursor": None}
        ]
        
        pages = mock_notion_client.get_database_pages("test_db_id")
        
        assert len(pages) == 3
        assert pages[0] == page1
        assert pages[1] == page2
        assert pages[2] == page3
        
        # Verify pagination calls
        calls = mock_notion_client.client.databases.query.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["database_id"] == "test_db_id"
        assert calls[1][1]["start_cursor"] == "cursor1"
    
    def test_get_database_pages_with_limit(self, mock_notion_client):
        """Test getting pages with limit"""
        pages = [{"id": f"page{i}"} for i in range(10)]
        mock_notion_client.client.databases.query.return_value = {
            "results": pages,
            "has_more": False
        }
        
        result = mock_notion_client.get_database_pages("test_db_id", limit=5)
        
        assert len(result) == 5
        assert result[0]["id"] == "page0"
        assert result[4]["id"] == "page4"
    
    def test_get_page_content_with_caching(self, mock_notion_client, mock_settings):
        """Test page content retrieval with caching"""
        page_id = "test_page_id"
        mock_page = {"id": page_id, "properties": {}}
        mock_blocks = [{"type": "paragraph", "paragraph": {"rich_text": []}}]
        
        mock_notion_client.client.pages.retrieve.return_value = mock_page
        mock_notion_client.client.blocks.children.list.return_value = {
            "results": mock_blocks,
            "has_more": False
        }
        mock_notion_client.settings = mock_settings
        mock_settings.enable_caching = True
        
        # First call - should fetch from API
        with patch('time.time', return_value=1000):
            result1 = mock_notion_client.get_page_content(page_id)
        
        assert result1["id"] == page_id
        mock_notion_client.client.pages.retrieve.assert_called_once()
        
        # Second call within cache TTL - should use cache
        mock_notion_client.client.pages.retrieve.reset_mock()
        with patch('time.time', return_value=1500):  # Within 1 hour TTL
            result2 = mock_notion_client.get_page_content(page_id)
        
        assert result2 == result1
        mock_notion_client.client.pages.retrieve.assert_not_called()
        
        # Third call after cache expiry - should fetch again
        with patch('time.time', return_value=5000):  # After 1 hour TTL
            result3 = mock_notion_client.get_page_content(page_id)
        
        mock_notion_client.client.pages.retrieve.assert_called_once()
    
    def test_extract_text_from_blocks(self, mock_notion_client):
        """Test text extraction from various block types"""
        blocks = [
            {
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"plain_text": "This is a paragraph"}]
                }
            },
            {
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"plain_text": "Heading 1"}]
                }
            },
            {
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"plain_text": "Bullet point"}]
                }
            },
            {
                "type": "to_do",
                "to_do": {
                    "rich_text": [{"plain_text": "Task item"}],
                    "checked": True
                }
            },
            {
                "type": "code",
                "code": {
                    "rich_text": [{"plain_text": "print('hello')"}],
                    "language": "python"
                }
            }
        ]
        
        text = mock_notion_client.extract_text_from_blocks(blocks)
        
        assert "This is a paragraph" in text
        assert "Heading 1" in text
        assert "• Bullet point" in text
        assert "✓ Task item" in text
        assert "```python\nprint('hello')\n```" in text
    
    def test_extract_page_title(self, mock_notion_client):
        """Test page title extraction"""
        # Page with Name property
        page1 = {
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": "Test Page Title"}]
                }
            }
        }
        assert mock_notion_client.extract_page_title(page1) == "Test Page Title"
        
        # Page with Title property
        page2 = {
            "properties": {
                "Title": {
                    "type": "title",
                    "title": [{"plain_text": "Another Title"}]
                }
            }
        }
        assert mock_notion_client.extract_page_title(page2) == "Another Title"
        
        # Page with custom title property
        page3 = {
            "properties": {
                "CustomTitle": {
                    "type": "title",
                    "title": [{"plain_text": "Custom"}]
                }
            }
        }
        assert mock_notion_client.extract_page_title(page3) == "Custom"
        
        # Page without title
        page4 = {"properties": {}}
        assert mock_notion_client.extract_page_title(page4) == "Untitled"
    
    @pytest.mark.asyncio
    async def test_scan_workspace(self, mock_notion_client):
        """Test workspace scanning"""
        mock_databases = [
            {
                "id": "db1",
                "title": [{"plain_text": "Tasks"}],
                "properties": {
                    "Status": {"type": "select", "id": "prop1"}
                }
            }
        ]
        
        mock_notion_client.client.search.return_value = {
            "results": mock_databases
        }
        mock_notion_client.client.databases.query.return_value = {
            "results": [{"id": "page1"}],
            "has_more": False
        }
        
        with patch.object(mock_notion_client, '_save_workspace_structure'):
            result = await mock_notion_client.scan_workspace()
        
        assert "databases" in result
        assert "db1" in result["databases"]
        assert result["databases"]["db1"]["title"] == "Tasks"
        assert "total_pages" in result
        assert "relationships" in result
        assert "scan_timestamp" in result
    
    def test_create_page(self, mock_notion_client):
        """Test page creation"""
        mock_page = {"id": "new_page_id"}
        mock_notion_client.client.pages.create.return_value = mock_page
        
        properties = {"Name": {"title": [{"text": {"content": "New Page"}}]}}
        content = [{"object": "block", "type": "paragraph"}]
        
        result = mock_notion_client.create_page("db_id", properties, content)
        
        assert result == mock_page
        mock_notion_client.client.pages.create.assert_called_once()
        call_args = mock_notion_client.client.pages.create.call_args[1]
        assert call_args["parent"]["database_id"] == "db_id"
        assert call_args["properties"] == properties
        assert call_args["children"] == content
    
    def test_update_page(self, mock_notion_client):
        """Test page update"""
        mock_page = {"id": "page_id", "archived": True}
        mock_notion_client.client.pages.update.return_value = mock_page
        
        properties = {"Status": {"select": {"name": "Complete"}}}
        result = mock_notion_client.update_page("page_id", properties, archived=True)
        
        assert result == mock_page
        mock_notion_client.client.pages.update.assert_called_once()
        call_args = mock_notion_client.client.pages.update.call_args[1]
        assert call_args["page_id"] == "page_id"
        assert call_args["properties"] == properties
        assert call_args["archived"] is True


class TestWorkspaceAnalysis:
    """Test workspace analysis functions"""
    
    def test_detect_database_relationships(self, mock_notion_client, sample_workspace_structure):
        """Test relationship detection between databases"""
        mock_notion_client.workspace_structure = sample_workspace_structure
        
        relationships = mock_notion_client._detect_database_relationships(
            sample_workspace_structure["databases"]
        )
        
        assert len(relationships) == 1
        assert relationships[0]["from_database"] == "db2"
        assert relationships[0]["to_database"] == "db1"
        assert relationships[0]["property_name"] == "Tasks"
        assert relationships[0]["type"] == "relation"
    
    def test_analyze_database_properties(self, mock_notion_client):
        """Test database property analysis"""
        database = {
            "properties": {
                "Status": {"type": "select", "id": "prop1"},
                "Tasks": {
                    "type": "relation",
                    "id": "prop2",
                    "relation": {"database_id": "related_db"}
                },
                "Rollup": {
                    "type": "rollup",
                    "id": "prop3",
                    "rollup": {"relation_property_name": "Tasks"}
                }
            }
        }
        
        analyzed = mock_notion_client._analyze_database_properties(database)
        
        assert analyzed["Status"]["type"] == "select"
        assert analyzed["Tasks"]["type"] == "relation"
        assert analyzed["Tasks"]["database_id"] == "related_db"
        assert analyzed["Rollup"]["type"] == "rollup"
        assert analyzed["Rollup"]["relation_property"] == "Tasks"
    
    def test_save_and_load_workspace_structure(self, mock_notion_client, mock_settings, sample_workspace_structure):
        """Test saving and loading workspace structure"""
        import tempfile
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.data_dir = Path(tmpdir)
            mock_notion_client.settings = mock_settings
            mock_notion_client.workspace_structure = sample_workspace_structure
            
            # Save structure
            mock_notion_client._save_workspace_structure()
            
            # Verify file exists
            file_path = Path(tmpdir) / "workspace_structure.json"
            assert file_path.exists()
            
            # Load and verify
            mock_notion_client.workspace_structure = None
            loaded = mock_notion_client.load_workspace_structure()
            
            assert loaded == sample_workspace_structure
            assert mock_notion_client.workspace_structure == sample_workspace_structure