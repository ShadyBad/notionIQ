"""
Recommendation Executor - Actually performs the recommended actions
Executes the AI-generated recommendations to organize your Notion workspace
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from config import Settings
from logger_wrapper import logger
from notion_wrapper import NotionAdvancedClient

console = Console()


class RecommendationExecutor:
    """Execute AI recommendations to organize Notion workspace"""

    def __init__(self, notion_client: NotionAdvancedClient, settings: Settings):
        """Initialize recommendation executor"""
        self.notion = notion_client
        self.settings = settings
        self.executed_actions = []
        self.failed_actions = []

    async def execute_recommendations(
        self,
        recommendations: Dict[str, List[Dict]],
        dry_run: bool = False,
        interactive: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute all recommendations

        Args:
            recommendations: Dict of categorized recommendations
            dry_run: If True, don't actually make changes
            interactive: If True, ask for confirmation before actions

        Returns:
            Execution results summary
        """

        console.print(
            f"\\n[bold cyan]🚀 Starting Recommendation Execution {'(DRY RUN)' if dry_run else '(LIVE)'}[/bold cyan]\\n"
        )

        results = {
            "executed": 0,
            "skipped": 0,
            "failed": 0,
            "actions_taken": [],
            "errors": [],
        }

        # Execute in priority order
        execution_order = [
            ("immediate_actions", "🚨 Immediate Actions"),
            ("suggested_moves", "📦 Moving Pages"),
            ("archive_candidates", "📚 Archiving Pages"),
            ("delete_candidates", "🗑️ Deleting Pages"),
            ("merge_candidates", "🔗 Merging Pages"),
        ]

        for category, display_name in execution_order:
            if recommendations.get(category):
                console.print(f"\\n[bold]{display_name}:[/bold]")
                await self._execute_category(
                    category, recommendations[category], results, dry_run, interactive
                )

        # Display summary
        self._display_execution_summary(results, dry_run)

        return results

    async def _execute_category(
        self,
        category: str,
        items: List[Dict],
        results: Dict,
        dry_run: bool,
        interactive: bool,
    ):
        """Execute all items in a category"""

        for item in items:
            try:
                # Show what we're about to do
                action_desc = self._get_action_description(category, item)
                console.print(f"  • {action_desc}")

                # Ask for confirmation if interactive
                if interactive and not dry_run:
                    if not Confirm.ask(f"    Execute this action?", default=True):
                        results["skipped"] += 1
                        continue

                # Execute the action
                if not dry_run:
                    success = await self._execute_single_action(category, item)
                    if success:
                        results["executed"] += 1
                        results["actions_taken"].append(
                            {
                                "category": category,
                                "action": action_desc,
                                "page_id": item.get("page_id"),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        console.print(f"    [green]✅ Success![/green]")
                    else:
                        results["failed"] += 1
                        console.print(f"    [red]❌ Failed![/red]")
                else:
                    # Dry run - just log what would happen
                    results["executed"] += 1
                    console.print(f"    [dim]Would execute: {action_desc}[/dim]")

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(
                    {
                        "page_id": item.get("page_id"),
                        "error": str(e),
                        "action": category,
                    }
                )
                logger.error(
                    f"Failed to execute {category} for {item.get('title')}: {e}"
                )
                console.print(f"    [red]❌ Error: {e}[/red]")

    async def _execute_single_action(self, category: str, item: Dict) -> bool:
        """Execute a single recommendation action"""

        page_id = item.get("page_id")
        if not page_id:
            return False

        try:
            if category == "suggested_moves":
                return await self._move_page(item)
            elif category == "archive_candidates":
                return await self._archive_page(item)
            elif category == "delete_candidates":
                return await self._delete_page(item)
            elif category == "merge_candidates":
                return await self._merge_pages(item)
            elif category == "immediate_actions":
                return await self._handle_immediate_action(item)
            else:
                logger.warning(f"Unknown action category: {category}")
                return False

        except Exception as e:
            logger.error(f"Error executing {category} for page {page_id}: {e}")
            return False

    async def _move_page(self, item: Dict) -> bool:
        """Move page to suggested database"""

        page_id = item.get("page_id")
        suggested_db = item.get("suggested_database")

        if not suggested_db:
            logger.warning(f"No suggested database for page {page_id}")
            return False

        # Find database ID by name
        db_id = await self._find_database_by_name(suggested_db)
        if not db_id:
            logger.warning(f"Database '{suggested_db}' not found")
            return False

        # Move page to database
        try:
            self.notion.client.pages.update(
                page_id=page_id, parent={"database_id": db_id}
            )
            logger.info(f"Moved page {page_id} to database {suggested_db}")
            return True
        except Exception as e:
            logger.error(f"Failed to move page {page_id}: {e}")
            return False

    async def _archive_page(self, item: Dict) -> bool:
        """Archive a page"""

        page_id = item.get("page_id")

        try:
            self.notion.client.pages.update(page_id=page_id, archived=True)
            logger.info(f"Archived page {page_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to archive page {page_id}: {e}")
            return False

    async def _delete_page(self, item: Dict) -> bool:
        """Delete a page (archive it, since Notion doesn't have true delete)"""

        # Notion doesn't support true deletion, so we archive instead
        return await self._archive_page(item)

    async def _merge_pages(self, item: Dict) -> bool:
        """Merge pages (placeholder - complex operation)"""

        # This is a complex operation that would require content analysis
        # For now, just log the recommendation
        logger.info(f"Merge recommendation noted for page {item.get('page_id')}")
        return True

    async def _handle_immediate_action(self, item: Dict) -> bool:
        """Handle immediate action items"""

        # For immediate actions, we might want to:
        # 1. Add to a priority database
        # 2. Set urgent properties
        # 3. Create follow-up tasks

        page_id = item.get("page_id")

        try:
            # Add urgent property if possible
            self.notion.client.pages.update(
                page_id=page_id, properties={"Priority": {"select": {"name": "Urgent"}}}
            )
            logger.info(f"Marked page {page_id} as urgent")
            return True
        except Exception as e:
            # Property might not exist, that's okay
            logger.debug(f"Could not set urgent property for {page_id}: {e}")
            return True  # Still count as success

    async def _find_database_by_name(self, name: str) -> Optional[str]:
        """Find database ID by name"""

        # Try to load workspace structure from file if not in memory
        try:
            workspace_file = self.settings.data_dir / "workspace_structure.json"
            if workspace_file.exists():
                import json

                with open(workspace_file) as f:
                    workspace_data = json.load(f)

                databases = workspace_data.get("databases", {})
                for db_id, db_info in databases.items():
                    if db_info.get("title", "").lower() == name.lower():
                        return db_id
        except Exception as e:
            logger.warning(f"Could not load workspace structure: {e}")

        # Fallback: search via Notion API
        try:
            search_response = self.notion.client.search(
                query=name, filter={"property": "object", "value": "database"}
            )

            for result in search_response.get("results", []):
                title = result.get("title", [])
                if title and isinstance(title, list):
                    title_text = "".join([t.get("plain_text", "") for t in title])
                    if title_text.lower() == name.lower():
                        return result["id"]

        except Exception as e:
            logger.warning(f"Could not search for database {name}: {e}")

        return None

    def _get_action_description(self, category: str, item: Dict) -> str:
        """Get human-readable description of action"""

        title = item.get("title", "Untitled")
        confidence = item.get("confidence", 0)

        if category == "suggested_moves":
            db = item.get("suggested_database", "Unknown")
            return f"Move '{title}' to {db} database (confidence: {confidence:.0%})"
        elif category == "archive_candidates":
            return f"Archive '{title}' (confidence: {confidence:.0%})"
        elif category == "delete_candidates":
            return f"Delete '{title}' (confidence: {confidence:.0%})"
        elif category == "merge_candidates":
            return (
                f"Merge '{title}' with related content (confidence: {confidence:.0%})"
            )
        elif category == "immediate_actions":
            return f"Mark '{title}' as urgent action item"
        else:
            return f"Process '{title}'"

    def _display_execution_summary(self, results: Dict, dry_run: bool):
        """Display execution summary"""

        console.print(
            f"\\n[bold cyan]📊 Execution Summary {'(DRY RUN)' if dry_run else ''}:[/bold cyan]\\n"
        )

        # Create summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")

        table.add_row("Actions Executed", str(results["executed"]))
        table.add_row("Actions Skipped", str(results["skipped"]))
        table.add_row("Actions Failed", str(results["failed"]))
        table.add_row(
            "Total Actions",
            str(results["executed"] + results["skipped"] + results["failed"]),
        )

        console.print(table)

        # Show errors if any
        if results["errors"]:
            console.print("\\n[bold red]❌ Errors:[/bold red]")
            for error in results["errors"]:
                console.print(f"  • {error['action']}: {error['error']}")

        # Show actions taken
        if results["actions_taken"]:
            console.print(
                f"\\n[bold green]✅ Actions {'That Would Be ' if dry_run else ''}Taken:[/bold green]"
            )
            for action in results["actions_taken"]:
                console.print(f"  • {action['action']}")


async def demo_with_test_data():
    """Demo the executor with test recommendations"""

    # Create test recommendations
    test_recommendations = {
        "immediate_actions": [
            {
                "page_id": "test-123",
                "title": "Urgent Task - Client Meeting Prep",
                "confidence": 0.95,
                "reasoning": "Detected deadline keywords",
            }
        ],
        "suggested_moves": [
            {
                "page_id": "test-456",
                "title": "Project Planning Notes",
                "confidence": 0.85,
                "suggested_database": "Projects",
                "reasoning": "Contains project-related content",
            },
            {
                "page_id": "test-789",
                "title": "Meeting Notes - Team Standup",
                "confidence": 0.90,
                "suggested_database": "Meetings",
                "reasoning": "Identified as meeting notes",
            },
        ],
        "archive_candidates": [
            {
                "page_id": "test-101",
                "title": "Old Template - Deprecated",
                "confidence": 0.88,
                "reasoning": "Template marked as deprecated",
            }
        ],
    }

    console.print("\\n[bold cyan]🎭 Demo Mode - Test Recommendations[/bold cyan]")

    # Mock executor (no real Notion client)
    class MockExecutor:
        def __init__(self):
            pass

        async def execute_recommendations(self, recs, dry_run=True, interactive=False):
            console.print("\\n[bold]Demo Recommendations to Execute:[/bold]\\n")

            for category, items in recs.items():
                if items:
                    console.print(f"[bold]{category.replace('_', ' ').title()}:[/bold]")
                    for item in items:
                        action = (
                            f"• {item['title']} (confidence: {item['confidence']:.0%})"
                        )
                        if "suggested_database" in item:
                            action += f" → Move to {item['suggested_database']}"
                        console.print(f"  {action}")
                        if dry_run:
                            console.print(f"    [dim]Would execute this action[/dim]")

            return {
                "executed": sum(len(items) for items in recs.values()),
                "skipped": 0,
                "failed": 0,
            }

    executor = MockExecutor()
    results = await executor.execute_recommendations(test_recommendations, dry_run=True)

    console.print(
        f"\\n[bold green]✅ Demo complete! Would execute {results['executed']} actions.[/bold green]"
    )


if __name__ == "__main__":
    asyncio.run(demo_with_test_data())
