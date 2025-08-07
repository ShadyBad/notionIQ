"""
Enhanced Notion API client wrapper for NotionIQ
Provides advanced functionality for workspace analysis and page management
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from notion_client import Client
from notion_client.errors import APIResponseError, RequestTimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential
from logger_wrapper import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import Settings, get_settings


class NotionAdvancedClient:
    """Enhanced Notion client with advanced features"""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize Notion client"""
        self.settings = settings or get_settings()
        self.client = Client(auth=self.settings.notion_api_key)
        # AsyncClient not available in current notion-client version
        self.async_client = None
        
        # Rate limiting
        self.rate_limit = self.settings.rate_limit_requests_per_second
        self.last_request_time = 0
        
        # Caching
        self.cache = {}
        self.cache_expiry = {}
        
        # Workspace metadata
        self.workspace_structure = None
        self.databases = {}
        self.page_count = 0
        
        logger.info("NotionAdvancedClient initialized")
    
    def _rate_limit_wait(self) -> None:
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def get_database(self, database_id: str) -> Dict[str, Any]:
        """Get database metadata with retry logic"""
        self._rate_limit_wait()
        
        try:
            database = self.client.databases.retrieve(database_id)
            logger.debug(f"Retrieved database: {database.get('title', [{}])[0].get('plain_text', 'Untitled')}")
            return database
        except APIResponseError as e:
            logger.error(f"Error retrieving database {database_id}: {e}")
            raise
    
    def get_database_pages(
        self,
        database_id: str,
        filter_dict: Optional[Dict] = None,
        sorts: Optional[List] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all pages from a database with optional filtering"""
        self._rate_limit_wait()
        
        pages = []
        has_more = True
        next_cursor = None
        
        query_params = {"database_id": database_id}
        if filter_dict:
            query_params["filter"] = filter_dict
        if sorts:
            query_params["sorts"] = sorts
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching pages from database...", total=None)
            
            while has_more:
                if next_cursor:
                    query_params["start_cursor"] = next_cursor
                if limit and len(pages) >= limit:
                    break
                
                try:
                    self._rate_limit_wait()
                    response = self.client.databases.query(**query_params)
                    
                    pages.extend(response.get("results", []))
                    has_more = response.get("has_more", False)
                    next_cursor = response.get("next_cursor")
                    
                    progress.update(task, description=f"Fetched {len(pages)} pages...")
                    
                except APIResponseError as e:
                    logger.error(f"Error querying database: {e}")
                    break
        
        logger.info(f"Retrieved {len(pages)} pages from database")
        return pages[:limit] if limit else pages
    
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """Get full page content including properties and blocks"""
        self._rate_limit_wait()
        
        # Check cache
        cache_key = f"page_{page_id}"
        if cache_key in self.cache:
            if time.time() < self.cache_expiry.get(cache_key, 0):
                logger.debug(f"Using cached content for page {page_id}")
                return self.cache[cache_key]
        
        try:
            # Get page metadata
            page = self.client.pages.retrieve(page_id)
            
            # Get page blocks (content)
            blocks = self.get_page_blocks(page_id)
            
            # Extract text content
            content = self.extract_text_from_blocks(blocks)
            
            result = {
                "id": page_id,
                "url": page.get("url", ""),
                "created_time": page.get("created_time"),
                "last_edited_time": page.get("last_edited_time"),
                "properties": page.get("properties", {}),
                "parent": page.get("parent", {}),
                "archived": page.get("archived", False),
                "content": content,
                "blocks": blocks,
                "title": self.extract_page_title(page)
            }
            
            # Cache result
            if self.settings.enable_caching:
                self.cache[cache_key] = result
                self.cache_expiry[cache_key] = time.time() + self.settings.cache_ttl.total_seconds()
            
            return result
            
        except APIResponseError as e:
            logger.error(f"Error retrieving page {page_id}: {e}")
            return {}
    
    def get_page_blocks(self, page_id: str) -> List[Dict[str, Any]]:
        """Get all blocks from a page"""
        blocks = []
        has_more = True
        next_cursor = None
        
        while has_more:
            try:
                self._rate_limit_wait()
                
                params = {"block_id": page_id}
                if next_cursor:
                    params["start_cursor"] = next_cursor
                
                response = self.client.blocks.children.list(**params)
                
                blocks.extend(response.get("results", []))
                has_more = response.get("has_more", False)
                next_cursor = response.get("next_cursor")
                
            except APIResponseError as e:
                logger.error(f"Error retrieving blocks for page {page_id}: {e}")
                break
        
        return blocks
    
    def extract_text_from_blocks(self, blocks: List[Dict[str, Any]]) -> str:
        """Extract plain text from Notion blocks"""
        text_parts = []
        
        for block in blocks:
            block_type = block.get("type")
            block_data = block.get(block_type, {})
            
            # Handle different block types
            if block_type in ["paragraph", "heading_1", "heading_2", "heading_3"]:
                rich_text = block_data.get("rich_text", [])
                text = self._extract_plain_text(rich_text)
                if text:
                    text_parts.append(text)
            
            elif block_type == "bulleted_list_item" or block_type == "numbered_list_item":
                rich_text = block_data.get("rich_text", [])
                text = self._extract_plain_text(rich_text)
                if text:
                    text_parts.append(f"• {text}")
            
            elif block_type == "to_do":
                rich_text = block_data.get("rich_text", [])
                text = self._extract_plain_text(rich_text)
                checked = block_data.get("checked", False)
                if text:
                    checkbox = "✓" if checked else "○"
                    text_parts.append(f"{checkbox} {text}")
            
            elif block_type == "toggle":
                rich_text = block_data.get("rich_text", [])
                text = self._extract_plain_text(rich_text)
                if text:
                    text_parts.append(f"▼ {text}")
            
            elif block_type == "code":
                rich_text = block_data.get("rich_text", [])
                text = self._extract_plain_text(rich_text)
                language = block_data.get("language", "")
                if text:
                    text_parts.append(f"```{language}\n{text}\n```")
            
            elif block_type == "quote":
                rich_text = block_data.get("rich_text", [])
                text = self._extract_plain_text(rich_text)
                if text:
                    text_parts.append(f"> {text}")
            
            elif block_type == "callout":
                rich_text = block_data.get("rich_text", [])
                text = self._extract_plain_text(rich_text)
                icon = block_data.get("icon", {}).get("emoji", "💡")
                if text:
                    text_parts.append(f"{icon} {text}")
        
        return "\n\n".join(text_parts)
    
    def _extract_plain_text(self, rich_text_array: List[Dict[str, Any]]) -> str:
        """Extract plain text from rich text array"""
        return "".join(
            rt.get("plain_text", "") for rt in rich_text_array
        )
    
    def extract_page_title(self, page: Dict[str, Any]) -> str:
        """Extract page title from properties"""
        properties = page.get("properties", {})
        
        # Try common title property names
        title_names = ["Name", "Title", "title", "name"]
        
        for prop_name in title_names:
            if prop_name in properties:
                prop = properties[prop_name]
                if prop.get("type") == "title":
                    title_array = prop.get("title", [])
                    if title_array:
                        return self._extract_plain_text(title_array)
        
        # Try to find any title property
        for prop_name, prop_value in properties.items():
            if prop_value.get("type") == "title":
                title_array = prop_value.get("title", [])
                if title_array:
                    return self._extract_plain_text(title_array)
        
        return "Untitled"
    
    async def scan_workspace(self) -> Dict[str, Any]:
        """Scan entire workspace for structure and relationships"""
        logger.info("Starting workspace scan...")
        
        workspace_data = {
            "databases": {},
            "total_pages": 0,
            "relationships": [],
            "property_types": {},
            "scan_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Search for all databases
        try:
            response = self.client.search(
                filter={"property": "object", "value": "database"}
            )
            
            databases = response.get("results", [])
            logger.info(f"Found {len(databases)} databases in workspace")
            
            for db in databases:
                db_id = db["id"]
                db_title = self._extract_database_title(db)
                
                workspace_data["databases"][db_id] = {
                    "title": db_title,
                    "properties": self._analyze_database_properties(db),
                    "created_time": db.get("created_time"),
                    "last_edited_time": db.get("last_edited_time"),
                    "page_count": 0  # Will be updated
                }
                
                # Count pages in database with proper pagination
                page_count = 0
                has_more = True
                next_cursor = None
                
                while has_more:
                    try:
                        query_params = {
                            "database_id": db_id,
                            "page_size": 100  # Max page size for counting
                        }
                        if next_cursor:
                            query_params["start_cursor"] = next_cursor
                        
                        response = self.client.databases.query(**query_params)
                        page_count += len(response.get("results", []))
                        has_more = response.get("has_more", False)
                        next_cursor = response.get("next_cursor")
                    except Exception as e:
                        logger.warning(f"Error counting pages in database {db_id}: {e}")
                        break
                
                workspace_data["databases"][db_id]["page_count"] = page_count
                workspace_data["total_pages"] += page_count
            
            # Detect relationships between databases
            workspace_data["relationships"] = self._detect_database_relationships(
                workspace_data["databases"]
            )
            
        except Exception as e:
            logger.error(f"Error scanning workspace: {e}")
        
        # Save workspace structure
        self.workspace_structure = workspace_data
        self._save_workspace_structure()
        
        return workspace_data
    
    def _extract_database_title(self, database: Dict[str, Any]) -> str:
        """Extract database title"""
        title_array = database.get("title", [])
        if title_array:
            return self._extract_plain_text(title_array)
        return "Untitled Database"
    
    def _analyze_database_properties(self, database: Dict) -> Dict[str, Any]:
        """Analyze database properties and their types"""
        properties = database.get("properties", {})
        analyzed = {}
        
        for prop_name, prop_config in properties.items():
            analyzed[prop_name] = {
                "type": prop_config.get("type"),
                "id": prop_config.get("id")
            }
            
            # Add type-specific details
            if prop_config.get("type") == "relation":
                analyzed[prop_name]["database_id"] = prop_config.get("relation", {}).get("database_id")
            elif prop_config.get("type") == "rollup":
                analyzed[prop_name]["relation_property"] = prop_config.get("rollup", {}).get("relation_property_name")
        
        return analyzed
    
    def _detect_database_relationships(self, databases: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect relationships between databases"""
        relationships = []
        
        for db_id, db_data in databases.items():
            for prop_name, prop_data in db_data["properties"].items():
                if prop_data["type"] == "relation":
                    related_db_id = prop_data.get("database_id")
                    if related_db_id and related_db_id in databases:
                        relationships.append({
                            "from_database": db_id,
                            "from_database_title": db_data["title"],
                            "to_database": related_db_id,
                            "to_database_title": databases[related_db_id]["title"],
                            "property_name": prop_name,
                            "type": "relation"
                        })
        
        return relationships
    
    def _save_workspace_structure(self) -> None:
        """Save workspace structure to file"""
        if self.workspace_structure:
            file_path = self.settings.data_dir / "workspace_structure.json"
            with open(file_path, "w") as f:
                json.dump(self.workspace_structure, f, indent=2)
            logger.info(f"Workspace structure saved to {file_path}")
    
    def load_workspace_structure(self) -> Optional[Dict]:
        """Load previously saved workspace structure"""
        file_path = self.settings.data_dir / "workspace_structure.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                self.workspace_structure = json.load(f)
                logger.info("Loaded existing workspace structure")
                return self.workspace_structure
        return None
    
    def create_page(
        self,
        parent_database_id: str,
        properties: Dict[str, Any],
        content: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Create a new page in a database"""
        self._rate_limit_wait()
        
        try:
            page_data = {
                "parent": {"database_id": parent_database_id},
                "properties": properties
            }
            
            if content:
                page_data["children"] = content
            
            page = self.client.pages.create(**page_data)
            logger.info(f"Created page: {page['id']}")
            return page
            
        except APIResponseError as e:
            logger.error(f"Error creating page: {e}")
            raise
    
    def update_page(
        self,
        page_id: str,
        properties: Optional[Dict[str, Any]] = None,
        archived: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Update page properties or archive status"""
        self._rate_limit_wait()
        
        try:
            update_data = {"page_id": page_id}
            
            if properties:
                update_data["properties"] = properties
            if archived is not None:
                update_data["archived"] = archived
            
            page = self.client.pages.update(**update_data)
            logger.info(f"Updated page: {page_id}")
            return page
            
        except APIResponseError as e:
            logger.error(f"Error updating page {page_id}: {e}")
            raise


if __name__ == "__main__":
    # Test the client
    client = NotionAdvancedClient()
    
    # Test getting database
    try:
        db = client.get_database(client.settings.notion_inbox_database_id)
        print(f"✅ Connected to database: {client._extract_database_title(db)}")
    except Exception as e:
        print(f"❌ Error: {e}")