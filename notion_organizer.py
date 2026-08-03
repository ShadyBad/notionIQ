#!/usr/bin/env python3
"""
NotionIQ - Intelligent Notion Workspace Organizer
Main orchestrator script that coordinates all components
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from pydantic import ValidationError
from rich import box
from rich import print as rprint
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from advanced_optimizer import OptimizationConfig
from ai_analyzer import UniversalAIAnalyzer
from ai_providers import AIProviderManager
from api_auto_config import get_auto_config
from api_optimizer import OptimizationLevel
from config import Settings, get_settings
from cost_monitor import CostMonitor
from logger_wrapper import logger
from notion_wrapper import NotionAdvancedClient
from recommendation_executor import RecommendationExecutor
from workspace_analyzer import WorkspaceAnalyzer

console = Console()


class NotionOrganizer:
    """Main orchestrator for NotionIQ"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.MINIMAL,
        preferred_provider: Optional[str] = None,
        enable_cost_monitoring: bool = True,
        verbose: bool = False,
        output_format: str = "json",
    ):
        """Initialize the organizer"""
        self.settings = settings or get_settings()
        self.optimization_level = optimization_level
        self.verbose = verbose
        self.output_format = output_format

        # Initialize cost monitoring
        self.cost_monitor = None
        if enable_cost_monitoring:
            self.cost_monitor = CostMonitor(
                self.settings.data_dir,
                budget_limits={
                    "hourly": 2.0,
                    "daily": 5.0,
                    "weekly": 25.0,
                    "monthly": 100.0,
                },
            )

        # Initialize components
        self.notion = NotionAdvancedClient(self.settings)
        # Use Universal AI Analyzer with automatic provider selection
        self.ai_analyzer = UniversalAIAnalyzer(
            self.settings,
            optimization_level=optimization_level,
            preferred_provider=preferred_provider,
        )
        self.workspace_analyzer = WorkspaceAnalyzer(self.notion, self.settings)

        # Initialize recommendation executor
        self.executor = RecommendationExecutor(self.notion, self.settings)

        # Results storage
        self.analysis_results = []
        self.workspace_analysis = {}
        self.report_data = {}

        # Configure logging
        logger.remove()  # Remove default handler
        log_file = self.settings.data_dir / "notioniq.log"
        logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            level=self.settings.log_level.value,
        )

        if self.settings.is_development:
            logger.add(sys.stderr, level="DEBUG")

        logger.info("NotionOrganizer initialized")

    async def run_analysis(
        self,
        analyze_workspace: bool = True,
        process_mode: str = "workspace",  # "workspace", "inbox", "page", "databases"
        target_databases: Optional[List[str]] = None,
        skip_databases: Optional[List[str]] = None,
        target_page_id: Optional[str] = None,
        create_recommendations: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Run the complete analysis workflow

        Args:
            analyze_workspace: Whether to analyze workspace structure
            process_mode: Processing mode - "workspace" (all), "inbox" (inbox only),
                         "page" (specific page hierarchy), "databases" (specific databases)
            target_databases: List of database names to process (for "databases" mode)
            skip_databases: List of database names to skip
            target_page_id: Page ID to process hierarchy from (for "page" mode)
            create_recommendations: Whether to generate recommendations
            dry_run: Run without making changes
        """

        start_time = time.monotonic()

        # Workspace structure analysis
        if analyze_workspace:
            self.workspace_analysis = await self.workspace_analyzer.analyze_workspace(
                deep_scan=self.settings.enable_workspace_scan
            )

        # Process pages based on mode
        if process_mode == "workspace":
            # Process all databases in workspace
            await self._process_all_databases(
                skip_databases=skip_databases, max_pages_per_db=self.settings.batch_size
            )
        elif process_mode == "inbox":
            # Process only inbox
            await self._process_inbox_pages()
        elif process_mode == "page" and target_page_id:
            # Process specific page hierarchy
            await self._process_page_hierarchy(target_page_id)
        elif process_mode == "databases" and target_databases:
            # Process specific databases
            await self._process_all_databases(
                target_databases=target_databases,
                skip_databases=skip_databases,
                max_pages_per_db=self.settings.batch_size,
            )
        else:
            console.print(
                f"[yellow]Invalid mode or missing parameters for mode: {process_mode}[/yellow]"
            )
            return {}

        # Generate recommendations
        if create_recommendations:
            self._generate_recommendations()

        # Execute recommendations (if enabled and not dry run)
        execution_results = {}
        if (
            not dry_run
            and self.report_data.get("recommendations")
            and (self.settings.enable_auto_organization or self.settings.auto_execute)
        ):
            execution_results = await self.executor.execute_recommendations(
                self.report_data["recommendations"],
                dry_run=dry_run,
                interactive=not self.settings.auto_execute,
            )

        # Create report
        self.report_data = self._create_report()
        if execution_results:
            self.report_data["execution_results"] = execution_results

        # Save results
        self._save_results()

        # Update Notion (if not dry run)
        if not dry_run and self.settings.enable_recommendations_page:
            await self._update_notion_recommendations()

        # Display the single run-summary panel
        self._display_summary(elapsed=time.monotonic() - start_time)

        return self.report_data

    async def _process_all_databases(
        self,
        target_databases: Optional[List[str]] = None,
        skip_databases: Optional[List[str]] = None,
        max_pages_per_db: Optional[int] = None,
    ):
        """Process pages from all databases in the workspace"""

        if self.verbose:
            console.print("\n[dim]Scanning all databases in workspace...[/dim]")

        # Get all databases from workspace
        # Use the already loaded workspace structure or scan it
        if not self.notion.workspace_structure:
            self.notion.workspace_structure = await self.notion.scan_workspace()

        databases = self.notion.workspace_structure.get("databases", {})

        skip_databases = skip_databases or []
        total_pages_processed = 0

        # Filter databases if specified
        if target_databases:
            databases = {
                k: v for k, v in databases.items() if v.get("title") in target_databases
            }

        if self.verbose:
            console.print(f"[green]Found {len(databases)} databases to process[/green]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            for db_id, db_info in databases.items():
                db_name = db_info.get("title", "Untitled")

                # Skip if in skip list
                if db_name in skip_databases:
                    continue

                task = progress.add_task(f"Processing {db_name}...", total=None)

                try:
                    # Reuse pages already fetched during the workspace scan to
                    # avoid re-querying the database (the scan/analyze
                    # double-fetch). Fall back to a fresh query only when the
                    # scan didn't retain payloads (e.g. structure loaded from
                    # the cached workspace_structure.json).
                    limit = max_pages_per_db or self.settings.batch_size
                    pages = self.notion.get_scanned_pages(db_id, limit=limit)
                    if pages is None:
                        pages = self.notion.get_database_pages(db_id, limit=limit)

                    if pages:
                        # Process pages like we do for inbox
                        pages_with_content = []
                        for page in pages:
                            page_id = page["id"]
                            content = self.notion.get_page_content(page_id)
                            pages_with_content.append(
                                {"page": page, "content": content}
                            )

                        # Analyze pages
                        for page_data in pages_with_content:
                            content = page_data["content"]

                            # get_page_content already returns the full analyzable
                            # dict; analyze_page optimizes the content internally.
                            if content.get("content"):
                                result = self.ai_analyzer.analyze_page(content)
                                self.analysis_results.append(result)
                                total_pages_processed += 1

                        if self.verbose:
                            console.print(
                                f"  [dim]{db_name}: {len(pages)} pages processed[/dim]"
                            )
                    elif self.verbose:
                        console.print(f"  [dim]{db_name}: No pages found[/dim]")

                except Exception as e:
                    logger.error(f"Error processing database {db_name}: {e}")
                    console.print(f"  [red]{db_name}: Error - {str(e)[:50]}[/red]")

                progress.remove_task(task)

        logger.info(f"Analyzed {total_pages_processed} pages from all databases")

    async def _process_page_hierarchy(self, page_id: str, max_depth: int = 10):
        """Process a specific page and all its children recursively"""

        if self.verbose:
            console.print(f"\n[dim]Processing page hierarchy from {page_id}...[/dim]")

        processed_pages = []

        async def process_page_and_children(page_id: str, depth: int = 0):
            if depth > max_depth:
                return

            try:
                # Get page content (full analyzable dict incl. title + properties)
                content = self.notion.get_page_content(page_id)

                # Analyze the page (analyze_page optimizes the content internally)
                if content.get("content"):
                    result = self.ai_analyzer.analyze_page(content)
                    self.analysis_results.append(result)
                    processed_pages.append(page_id)

                # Get child pages
                children = self.notion.get_page_blocks(page_id)
                for child in children:
                    if child.get("type") == "child_page":
                        child_id = child.get("id")
                        await process_page_and_children(child_id, depth + 1)

            except Exception as e:
                logger.error(f"Error processing page {page_id}: {e}")

        await process_page_and_children(page_id)

        logger.info(f"Analyzed {len(processed_pages)} pages in hierarchy")

    async def _process_inbox_pages(self):
        """Process pages from the inbox database"""

        # Get pages from inbox
        if self.verbose:
            console.print("\n[dim]Fetching pages from Inbox database...[/dim]")

        try:
            inbox_pages = self.notion.get_database_pages(
                self.settings.notion_inbox_database_id,
                limit=(
                    None if not self.settings.batch_size else self.settings.batch_size
                ),
            )

            if not inbox_pages:
                console.print("[yellow]No pages found in Inbox database[/yellow]")
                return

            if self.verbose:
                console.print(
                    f"[green]Found {len(inbox_pages)} pages to process[/green]"
                )

            # Get full content for each page
            pages_with_content = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Fetching page content...", total=len(inbox_pages)
                )

                for page in inbox_pages:
                    page_content = self.notion.get_page_content(page["id"])
                    pages_with_content.append(page_content)
                    progress.update(task, advance=1)

            # Analyze pages with AI
            self.analysis_results = self.ai_analyzer.batch_analyze(
                pages_with_content, self.workspace_analysis
            )

            logger.info(f"Analyzed {len(self.analysis_results)} pages")

        except Exception as e:
            logger.error(f"Error processing inbox pages: {e}")
            console.print(f"[red]Error: {e}[/red]")

    def _generate_recommendations(self):
        """Generate recommendations based on analysis"""

        if not self.analysis_results:
            console.print(
                "[yellow]No analysis results to generate recommendations from[/yellow]"
            )
            return

        recommendations = {
            "immediate_actions": [],
            "suggested_moves": [],
            "archive_candidates": [],
            "delete_candidates": [],
            "review_needed": [],
            "merge_candidates": [],
        }

        for analysis in self.analysis_results:
            if "error" in analysis:
                continue

            rec = analysis.get("recommendations", {})
            action = rec.get("primary_action")
            confidence = rec.get("confidence", 0)

            page_info = {
                "page_id": analysis.get("page_id"),
                "title": analysis.get("page_title"),
                "url": analysis.get("page_url"),
                "action": action,
                "confidence": confidence,
                "reasoning": rec.get("reasoning", ""),
                "suggested_database": rec.get("suggested_database"),
                "classification": analysis.get("classification", {}).get(
                    "primary_type"
                ),
                "urgency": analysis.get("urgency_assessment", {}).get("level"),
            }

            # Categorize by action and confidence
            if action == "move_to_database" and confidence > 0.7:
                recommendations["suggested_moves"].append(page_info)
            elif action == "archive" and confidence > 0.7:
                recommendations["archive_candidates"].append(page_info)
            elif action == "delete" and confidence > 0.8:
                recommendations["delete_candidates"].append(page_info)
            elif action == "review" or confidence < 0.6:
                recommendations["review_needed"].append(page_info)
            elif action == "merge_with":
                recommendations["merge_candidates"].append(page_info)

            # Check for immediate actions based on urgency
            if analysis.get("urgency_assessment", {}).get("level") == "immediate":
                recommendations["immediate_actions"].append(page_info)

        self.report_data["recommendations"] = recommendations

        # Display recommendation counts (verbose only; the summary panel covers the rest)
        if self.verbose:
            console.print("\n[bold cyan]Recommendation Summary:[/bold cyan]")
            for category, items in recommendations.items():
                if items:
                    console.print(
                        f"  - {category.replace('_', ' ').title()}: {len(items)} pages"
                    )

    def _create_report(self) -> Dict[str, Any]:
        """Create comprehensive analysis report"""

        report = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "settings": {
                    "batch_size": self.settings.batch_size,
                    "enable_workspace_scan": self.settings.enable_workspace_scan,
                    "enable_pattern_learning": self.settings.enable_pattern_learning,
                },
            },
            "workspace_summary": {
                "total_databases": len(
                    self.workspace_analysis.get("workspace_structure", {}).get(
                        "databases", {}
                    )
                ),
                "total_pages_analyzed": len(self.analysis_results),
                "health_score": self.workspace_analysis.get("health_score", 0),
                "metrics": self.workspace_analysis.get("metrics", {}),
            },
            "classification_summary": self._summarize_classifications(),
            "recommendations": self.report_data.get("recommendations", {}),
            "insights": self._generate_insights(),
            "detailed_analysis": self.analysis_results,
        }

        return report

    def _summarize_classifications(self) -> Dict[str, Any]:
        """Summarize classification results"""

        summary = {
            "document_types": {},
            "urgency_levels": {},
            "contexts": {},
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
        }

        for analysis in self.analysis_results:
            if "error" in analysis:
                continue

            # Document type
            doc_type = analysis.get("classification", {}).get("primary_type", "unknown")
            summary["document_types"][doc_type] = (
                summary["document_types"].get(doc_type, 0) + 1
            )

            # Urgency
            urgency = analysis.get("urgency_assessment", {}).get("level", "unknown")
            summary["urgency_levels"][urgency] = (
                summary["urgency_levels"].get(urgency, 0) + 1
            )

            # Context
            context = analysis.get("context_detection", {}).get(
                "primary_context", "unknown"
            )
            summary["contexts"][context] = summary["contexts"].get(context, 0) + 1

            # Confidence
            confidence = analysis.get("classification", {}).get("confidence", 0)
            if confidence > 0.8:
                summary["confidence_distribution"]["high"] += 1
            elif confidence > 0.6:
                summary["confidence_distribution"]["medium"] += 1
            else:
                summary["confidence_distribution"]["low"] += 1

        return summary

    def _generate_insights(self) -> List[Dict[str, str]]:
        """Generate actionable insights from analysis"""

        insights = []

        # Analyze patterns
        classification_summary = self._summarize_classifications()

        # Most common document type
        if classification_summary["document_types"]:
            most_common = max(
                classification_summary["document_types"].items(), key=lambda x: x[1]
            )
            insights.append(
                {
                    "type": "pattern",
                    "title": "Most Common Content Type",
                    "description": f"Your inbox primarily contains {most_common[0]} items ({most_common[1]} pages)",
                    "action": f"Consider creating dedicated workflows for {most_common[0]} processing",
                }
            )

        # Urgency distribution
        urgent_count = classification_summary["urgency_levels"].get("immediate", 0)
        if urgent_count > 0:
            insights.append(
                {
                    "type": "alert",
                    "title": "Urgent Items Detected",
                    "description": f"Found {urgent_count} items requiring immediate attention",
                    "action": "Review and act on urgent items first",
                }
            )

        # Low confidence items
        low_confidence = classification_summary["confidence_distribution"]["low"]
        if low_confidence > len(self.analysis_results) * 0.3:
            insights.append(
                {
                    "type": "improvement",
                    "title": "Content Structure Needs Improvement",
                    "description": f"{low_confidence} pages have unclear structure or purpose",
                    "action": "Add clearer titles and descriptions to improve classification accuracy",
                }
            )

        # Workspace health
        health_score = self.workspace_analysis.get("health_score", 0)
        if health_score < 60:
            insights.append(
                {
                    "type": "health",
                    "title": "Workspace Organization Needs Attention",
                    "description": f"Current health score is {health_score:.0f}/100",
                    "action": "Follow recommendations to improve workspace structure",
                }
            )

        return insights

    def _save_results(self):
        """Save analysis results to files"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Select report format(s): json (default), markdown, or both.
        write_json = self.output_format in ("json", "both")
        write_markdown = self.output_format in ("markdown", "both")

        # Save JSON report
        if write_json:
            json_file = self.settings.output_dir / f"analysis_report_{timestamp}.json"
            with open(json_file, "w") as f:
                json.dump(self.report_data, f, indent=2, default=str)

            self._last_json_report = json_file
            if self.verbose:
                console.print(f"\n[green]Report saved to: {json_file}[/green]")

        # Save human-readable Markdown report
        if write_markdown:
            md_file = _write_markdown_report(
                self.report_data, self.settings.output_dir, timestamp=timestamp
            )
            self._last_markdown_report = md_file
            if self.verbose:
                console.print(f"[green]Markdown report saved to: {md_file}[/green]")

        # Save summary for quick review
        summary_file = self.settings.output_dir / f"summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write("NotionIQ Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Pages Analyzed: {len(self.analysis_results)}\n")
            f.write(
                f"Health Score: {self.workspace_analysis.get('health_score', 0):.1f}/100\n\n"
            )

            f.write("Recommendations:\n")
            for category, items in self.report_data.get("recommendations", {}).items():
                if items:
                    f.write(
                        f"  • {category.replace('_', ' ').title()}: {len(items)} pages\n"
                    )

            f.write("\nTop Insights:\n")
            for insight in self.report_data.get("insights", [])[:5]:
                f.write(
                    f"  • {insight.get('title', '')}: {insight.get('description', '')}\n"
                )

        logger.info(f"Results saved to {self.settings.output_dir}")

    async def _update_notion_recommendations(self):
        """Update recommendations page in Notion"""

        try:
            # Create recommendations content
            recommendations_content = self._format_recommendations_for_notion()

            # Search for existing recommendations page
            search_response = self.notion.client.search(
                query="NotionIQ Recommendations",
                filter={"property": "object", "value": "page"},
            )

            existing_pages = search_response.get("results", [])
            recommendations_page = None

            for page in existing_pages:
                title = self.notion.extract_page_title(page)
                if "NotionIQ Recommendations" in title:
                    recommendations_page = page
                    break

            if recommendations_page:
                # Update existing page
                page_id = recommendations_page["id"]

                # Clear existing blocks
                existing_blocks = self.notion.get_page_blocks(page_id)
                for block in existing_blocks:
                    try:
                        self.notion.client.blocks.delete(block_id=block["id"])
                    except:
                        pass  # Some blocks might not be deletable

                # Add new content
                self.notion.client.blocks.children.append(
                    block_id=page_id, children=recommendations_content
                )

                console.print(
                    f"[green]✅ Updated recommendations page: {page_id}[/green]"
                )
            else:
                # Determine parent for new page
                parent_id = (
                    self.settings.notion_recommendations_parent_id
                    or self.settings.notion_inbox_database_id
                )

                # Create new page in the specified parent (database or page)
                # Try as database first, if that fails, try as page
                try:
                    parent = {"database_id": parent_id}
                    new_page = self.notion.client.pages.create(
                        parent=parent,
                        properties={
                            "title": {
                                "title": [
                                    {
                                        "text": {
                                            "content": f"NotionIQ Recommendations - {datetime.now().strftime('%Y-%m-%d')}"
                                        }
                                    }
                                ]
                            }
                        },
                        children=recommendations_content,
                    )
                except Exception as db_error:
                    # If database parent fails, try as page parent
                    try:
                        parent = {"page_id": parent_id}
                        new_page = self.notion.client.pages.create(
                            parent=parent,
                            properties={
                                "title": {
                                    "title": [
                                        {
                                            "text": {
                                                "content": f"NotionIQ Recommendations - {datetime.now().strftime('%Y-%m-%d')}"
                                            }
                                        }
                                    ]
                                }
                            },
                            children=recommendations_content,
                        )
                    except Exception as page_error:
                        raise Exception(
                            f"Could not create page as child of database or page {parent_id}: Database error: {db_error}, Page error: {page_error}"
                        )

                console.print(
                    f"[green]✅ Created recommendations page: {new_page['id']}[/green]"
                )
                console.print(f"[dim]View at: {new_page.get('url', 'N/A')}[/dim]")

            logger.info("Successfully updated Notion recommendations page")

        except Exception as e:
            logger.error(f"Failed to update Notion recommendations page: {e}")
            console.print(
                f"[yellow]⚠️ Could not update Notion recommendations page: {e}[/yellow]"
            )

    def _format_recommendations_for_notion(self) -> List[Dict[str, Any]]:
        """Format recommendations as Notion blocks"""
        blocks = []

        # Add header
        blocks.append(
            {
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": "🚀 NotionIQ Analysis Report"},
                        }
                    ]
                },
            }
        )

        # Add timestamp
        blocks.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            },
                            "annotations": {"italic": True},
                        }
                    ]
                },
            }
        )

        # Add divider
        blocks.append({"object": "block", "type": "divider", "divider": {}})

        # Add health score
        health_score = self.workspace_analysis.get("health_score", 0)
        blocks.append(
            {
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": f"Workspace Health Score: {health_score:.1f}/100"
                            },
                            "annotations": {"bold": True},
                        }
                    ],
                    "icon": {"emoji": "💯"},
                },
            }
        )

        # Add summary statistics
        blocks.append(
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "📊 Summary"}}]
                },
            }
        )

        stats_text = f"""• Pages Analyzed: {len(self.analysis_results)}
• Databases Found: {len(self.workspace_analysis.get('workspace_structure', {}).get('databases', {}))}
• Total Pages in Workspace: {self.workspace_analysis.get('workspace_structure', {}).get('total_pages', 'Unknown')}"""

        blocks.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": stats_text}}]
                },
            }
        )

        # Add recommendations sections
        recommendations = self.report_data.get("recommendations", {})

        if recommendations.get("immediate_actions"):
            blocks.extend(
                self._format_recommendation_section(
                    "⚡ Immediate Actions Required",
                    recommendations["immediate_actions"],
                    "red",
                )
            )

        if recommendations.get("suggested_moves"):
            blocks.extend(
                self._format_recommendation_section(
                    "📦 Suggested Page Moves",
                    recommendations["suggested_moves"][:10],  # Limit to top 10
                    "yellow",
                )
            )

        if recommendations.get("archive_candidates"):
            blocks.extend(
                self._format_recommendation_section(
                    "🗄️ Archive Candidates",
                    recommendations["archive_candidates"][:10],
                    "gray",
                )
            )

        # Add insights
        insights = self.report_data.get("insights", [])
        if insights:
            blocks.append(
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [
                            {"type": "text", "text": {"content": "💡 Key Insights"}}
                        ]
                    },
                }
            )

            for insight in insights[:5]:
                blocks.append(
                    {
                        "object": "block",
                        "type": "toggle",
                        "toggle": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": insight.get("title", "")},
                                    "annotations": {"bold": True},
                                }
                            ],
                            "children": [
                                {
                                    "object": "block",
                                    "type": "paragraph",
                                    "paragraph": {
                                        "rich_text": [
                                            {
                                                "type": "text",
                                                "text": {
                                                    "content": insight.get(
                                                        "description", ""
                                                    )
                                                },
                                            }
                                        ]
                                    },
                                }
                            ],
                        },
                    }
                )

        return blocks

    def _format_recommendation_section(
        self, title: str, items: List[Dict[str, Any]], color: str = "default"
    ) -> List[Dict[str, Any]]:
        """Format a recommendation section for Notion"""
        blocks = []

        # Add section header
        blocks.append(
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": title}}]
                },
            }
        )

        # Add items as a bulleted list
        for item in items:
            text = f"{item.get('title', 'Untitled')}"
            if item.get("action"):
                text += f" → {item.get('action')}"
            if item.get("suggested_database"):
                text += f" (to {item.get('suggested_database')})"

            blocks.append(
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": text}}]
                    },
                }
            )

            # Add reasoning as nested item if available
            if item.get("reasoning"):
                blocks.append(
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": f"→ {item['reasoning']}"},
                                    "annotations": {"italic": True, "color": "gray"},
                                }
                            ]
                        },
                    }
                )

        return blocks

    def _display_summary(self, elapsed: float = 0.0):
        """Render the single run-summary panel."""

        metrics = getattr(
            getattr(self.ai_analyzer, "api_optimizer", None), "metrics", None
        )
        panel = self._build_summary_panel(metrics, elapsed)
        console.print()
        console.print(panel)

    def _build_summary_panel(self, metrics: Any, elapsed: float) -> Panel:
        """Compose one tasteful summary Panel from api_optimizer.metrics.

        Reads metrics fields: total_cost, total_tokens, cache_hits, cache_misses.
        Pages analyzed comes from the in-run results; top classifications from
        the classification summary.
        """
        pages = len(self.analysis_results)

        # Cache-hit rate from the optimizer's hit/miss counters.
        cache_hits = getattr(metrics, "cache_hits", 0) or 0
        cache_misses = getattr(metrics, "cache_misses", 0) or 0
        lookups = cache_hits + cache_misses
        cache_rate = (cache_hits / lookups * 100) if lookups else 0.0

        total_cost = getattr(metrics, "total_cost", 0.0) or 0.0
        total_tokens = getattr(metrics, "total_tokens", 0) or 0

        # Stats lines (label : value), right-aligned values.
        stats = Table.grid(padding=(0, 2))
        stats.add_column(justify="left", style="dim")
        stats.add_column(justify="right", style="bold")
        stats.add_row("Pages analyzed", str(pages))
        stats.add_row("Cache hit rate", f"{cache_rate:.0f}%")
        stats.add_row("Total cost", f"${total_cost:.4f}")
        stats.add_row("Tokens", f"{total_tokens:,}")
        stats.add_row("Elapsed", f"{elapsed:.1f}s")

        renderables: List[Any] = [stats]

        # Compact top-classifications table (top 5 by count).
        doc_types = self._summarize_classifications().get("document_types", {})
        top = sorted(doc_types.items(), key=lambda kv: kv[1], reverse=True)[:5]
        if top:
            table = Table(
                box=box.SIMPLE_HEAD,
                show_edge=False,
                pad_edge=False,
                padding=(0, 2, 0, 0),
            )
            table.add_column("Classification", style="cyan")
            table.add_column("Pages", justify="right", style="bold")
            for name, count in top:
                table.add_row(str(name).replace("_", " ").title(), str(count))
            renderables.append(table)

        return Panel(
            Group(*renderables),
            title="[bold]NotionIQ[/bold]",
            border_style="cyan",
            expand=False,
        )


class DefaultGroup(click.Group):
    """A click.Group that runs a default subcommand when none is given.

    Lets bare ``notioniq`` (plus all its analysis options) behave exactly as
    the old single-command CLI while still exposing ``notioniq init`` etc.
    """

    def __init__(self, *args, **kwargs):
        self.default_cmd_name = kwargs.pop("default_cmd_name", None)
        super().__init__(*args, **kwargs)

    def resolve_command(self, ctx, args):
        # If the first arg isn't a known subcommand, fall back to the default
        # command so its options parse against bare `notioniq`.
        if args and self.default_cmd_name:
            first = args[0]
            if first not in self.commands and not first.startswith("-"):
                # Unknown positional: let the default command try to handle it.
                args = [self.default_cmd_name] + args
        return super().resolve_command(ctx, args)

    def parse_args(self, ctx, args):
        # No args at all, or a leading option (no subcommand) -> default
        # command. This also routes `notioniq --help` to the analysis
        # command so its options are listed for bare `notioniq`.
        if self.default_cmd_name:
            if not args:
                args = [self.default_cmd_name]
            elif args[0].startswith("-"):
                args = [self.default_cmd_name] + args
        return super().parse_args(ctx, args)


@click.group(
    cls=DefaultGroup,
    default_cmd_name="run",
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
def main():
    """NotionIQ - Intelligent Notion Workspace Organizer.

    Run with no arguments to analyze your workspace, or use a subcommand
    (e.g. `notioniq init` to set up your API keys).
    """


@main.command(name="run")
@click.option(
    "--mode",
    type=click.Choice(["workspace", "inbox", "page", "databases"]),
    default="workspace",
    help="Processing mode: workspace (all databases), inbox (inbox only), page (specific page hierarchy), databases (specific databases)",
)
@click.option(
    "--target-databases",
    multiple=True,
    help="Specific database names to process (for 'databases' mode)",
)
@click.option(
    "--skip-databases",
    multiple=True,
    help="Database names to skip when processing",
)
@click.option(
    "--target-page",
    help="Page ID to process hierarchy from (for 'page' mode)",
)
@click.option(
    "--analyze-workspace/--skip-workspace",
    default=True,
    help="Perform deep workspace analysis",
)
@click.option(
    "--create-recommendations/--skip-recommendations",
    default=True,
    help="Generate recommendations",
)
@click.option(
    "--dry-run", is_flag=True, help="Run analysis without making changes to Notion"
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Number of pages to process (overrides config)",
)
@click.option(
    "--optimization",
    type=click.Choice(["auto", "minimal", "balanced", "full"]),
    default="minimal",
    help="API optimization level (minimal=aggressive cost savings, balanced=moderate, full=complete analysis)",
)
@click.option(
    "--provider",
    type=click.Choice(["auto", "claude", "chatgpt", "gemini"]),
    default="auto",
    help="AI provider to use (auto=automatic selection based on available APIs)",
)
@click.option(
    "--cost-monitor/--no-cost-monitor",
    default=True,
    help="Enable real-time cost monitoring and budget enforcement",
)
@click.option(
    "--daily-budget",
    type=float,
    default=5.0,
    help="Maximum daily API spend in USD (default: $5)",
)
@click.option(
    "--auto-execute",
    is_flag=True,
    help="Automatically execute recommendations without confirmation",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show per-page classification and processing detail (quiet by default)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown", "both"]),
    default="json",
    help="Report output format: json (default), markdown, or both",
)
def run(
    mode: str,
    target_databases: tuple,
    skip_databases: tuple,
    target_page: Optional[str],
    analyze_workspace: bool,
    create_recommendations: bool,
    dry_run: bool,
    batch_size: Optional[int],
    optimization: str,
    provider: str,
    cost_monitor: bool,
    daily_budget: float,
    auto_execute: bool,
    verbose: bool,
    output_format: str,
):
    """Analyze a Notion workspace (default command)."""

    try:
        # Load settings — friendly, actionable message if not configured yet
        try:
            settings = get_settings()
        except ValidationError as cfg_err:
            missing = [
                str(err["loc"][0])
                for err in cfg_err.errors()
                if err.get("type") == "missing"
            ]
            console.print("[bold red]NotionIQ isn't configured yet.[/bold red]")
            if missing:
                console.print(f"  Missing: [yellow]{', '.join(missing)}[/yellow]")
            console.print(
                "  Run [bold cyan]notioniq init[/bold cyan] to set up your keys "
                "(or copy .env.example to .env)."
            )
            sys.exit(1)

        # Detect and display available providers if auto
        if provider == "auto":
            provider_manager = AIProviderManager()
            provider_manager.display_available_providers()

            # Auto-select provider
            preferred_provider = None
        else:
            preferred_provider = provider
            console.print(f"[cyan]Using specified provider: {provider}[/cyan]")

        # Handle automatic configuration
        if optimization == "auto":
            console.print("[bold cyan]🔧 Auto-configuring API settings...[/bold cyan]")
            auto_config = get_auto_config()

            # Apply auto-configured settings
            optimization_level = OptimizationLevel[
                auto_config["optimization_level"].upper()
            ]
            settings.batch_size = (
                auto_config["batch_size"] if not batch_size else batch_size
            )
            settings.rate_limit_requests_per_second = auto_config["rate_limit_rps"]
            settings.enable_caching = auto_config["enable_caching"]
            settings.cache_ttl_hours = auto_config["cache_ttl_hours"]

            console.print(
                f"[green]✅ Using {auto_config['model']} with {auto_config['optimization_level']} optimization[/green]"
            )
            console.print(
                f"[dim]Estimated cost: ${auto_config['estimated_cost_per_100']}/100 pages[/dim]\n"
            )
        else:
            # Manual optimization level
            optimization_level = OptimizationLevel[optimization.upper()]

            # Override batch size if provided
            if batch_size:
                settings.batch_size = batch_size

        # Update budget if specified
        if cost_monitor and daily_budget:
            console.print(f"[cyan]💰 Daily budget set to: ${daily_budget:.2f}[/cyan]")

        # Override auto_execute setting if flag provided
        if auto_execute:
            settings.auto_execute = True
            console.print(
                f"[yellow]⚡ Auto-execute enabled - recommendations will be applied automatically[/yellow]"
            )

        # Initialize organizer with optimization level and provider
        organizer = NotionOrganizer(
            settings,
            optimization_level=optimization_level,
            preferred_provider=preferred_provider,
            enable_cost_monitoring=cost_monitor,
            verbose=verbose,
            output_format=output_format,
        )

        # Update budget limits if cost monitoring is enabled
        if cost_monitor and organizer.cost_monitor:
            organizer.cost_monitor.budget_limits["daily"] = daily_budget
            organizer.cost_monitor.budget_limits["weekly"] = daily_budget * 7
            organizer.cost_monitor.budget_limits["monthly"] = daily_budget * 30

        # Run analysis
        asyncio.run(
            organizer.run_analysis(
                analyze_workspace=analyze_workspace,
                process_mode=mode,
                target_databases=list(target_databases) if target_databases else None,
                skip_databases=list(skip_databases) if skip_databases else None,
                target_page_id=target_page,
                create_recommendations=create_recommendations,
                dry_run=dry_run,
            )
        )

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        logger.exception("Fatal error in main")
        sys.exit(1)


def _write_markdown_report(
    report: Dict[str, Any],
    output_dir: Path,
    timestamp: Optional[str] = None,
) -> Path:
    """Write a human-readable Markdown summary of an analysis report.

    Pure and testable: takes the in-memory ``report`` dict (the same shape
    produced by :meth:`NotionOrganizer._create_report`) plus an output
    directory, writes ``analysis_report_{timestamp}.md``, and returns the
    written path. Performs no network or Notion calls.

    Sections:
      - title + generated-at
      - Summary counts (databases, pages analyzed, health score)
      - Classifications table (document type -> count)
      - Per-category recommendations table (category -> page count)
      - Top actions (highest-confidence recommended page moves/actions)
      - Key insights
    """
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"analysis_report_{timestamp}.md"

    metadata = report.get("metadata", {})
    workspace = report.get("workspace_summary", {})
    classifications = report.get("classification_summary", {})
    recommendations = report.get("recommendations", {})
    insights = report.get("insights", [])

    lines: List[str] = []
    lines.append("# NotionIQ Analysis Report")
    lines.append("")
    generated_at = metadata.get("generated_at", timestamp)
    lines.append(f"_Generated: {generated_at}_")
    lines.append("")

    # Summary counts
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Databases | {workspace.get('total_databases', 0)} |")
    lines.append(f"| Pages analyzed | {workspace.get('total_pages_analyzed', 0)} |")
    health = workspace.get("health_score", 0) or 0
    lines.append(f"| Health score | {health:.1f}/100 |")
    lines.append("")

    # Classifications table
    doc_types = classifications.get("document_types", {})
    if doc_types:
        lines.append("## Classifications")
        lines.append("")
        lines.append("| Document type | Pages |")
        lines.append("| --- | --- |")
        for name, count in sorted(
            doc_types.items(), key=lambda kv: kv[1], reverse=True
        ):
            label = str(name).replace("_", " ").title()
            lines.append(f"| {label} | {count} |")
        lines.append("")

    # Per-category recommendations table
    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        lines.append("| Category | Pages |")
        lines.append("| --- | --- |")
        for category, items in recommendations.items():
            label = str(category).replace("_", " ").title()
            count = len(items) if isinstance(items, list) else 0
            lines.append(f"| {label} | {count} |")
        lines.append("")

    # Top actions: highest-confidence recommended page actions across categories.
    actions: List[Dict[str, Any]] = []
    for items in recommendations.values():
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    actions.append(item)
    actions.sort(key=lambda i: i.get("confidence", 0) or 0, reverse=True)
    if actions:
        lines.append("## Top Actions")
        lines.append("")
        for item in actions[:10]:
            title = item.get("title") or "Untitled"
            action = item.get("action") or "review"
            confidence = item.get("confidence", 0) or 0
            target = item.get("suggested_database")
            suffix = f" (to {target})" if target else ""
            lines.append(
                f"- **{title}** -> {action}{suffix} " f"(confidence {confidence:.0%})"
            )
        lines.append("")

    # Key insights
    if insights:
        lines.append("## Key Insights")
        lines.append("")
        for insight in insights[:5]:
            title = insight.get("title", "")
            description = insight.get("description", "")
            lines.append(f"- **{title}**: {description}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _update_env_file(env_path: Path, updates: Dict[str, str]) -> None:
    """Write/update keys in a .env file without clobbering unrelated keys.

    Existing lines for keys in ``updates`` are replaced in place; all other
    lines (including comments, blanks, and unrelated keys) are preserved.
    New keys are appended at the end.
    """

    # Strip CR/LF so a value can never inject an extra line/key into .env.
    def _clean(v: str) -> str:
        return str(v).replace("\n", "").replace("\r", "")

    remaining = {k: _clean(v) for k, v in updates.items()}
    lines: List[str] = []

    if env_path.exists():
        existing = env_path.read_text(encoding="utf-8").splitlines()
        for line in existing:
            stripped = line.strip()
            # Preserve comments and blank lines untouched.
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                lines.append(line)
                continue
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                lines.append(f"{key}={remaining.pop(key)}")
            else:
                lines.append(line)

    for key, value in remaining.items():
        lines.append(f"{key}={value}")

    content = "\n".join(lines)
    if content and not content.endswith("\n"):
        content += "\n"
    env_path.write_text(content, encoding="utf-8")
    # Credentials file — restrict to owner read/write only.
    try:
        env_path.chmod(0o600)
    except OSError:
        pass


@main.command(name="init")
def init():
    """Interactive setup wizard: collect API keys and write a .env file."""
    try:
        from security import SecurityValidator
    except Exception as exc:  # pragma: no cover - security module is expected present
        console.print(
            f"[yellow]Warning: security module unavailable ({exc}); "
            f"key format checks will be skipped.[/yellow]"
        )
        SecurityValidator = None  # type: ignore[assignment]

    console.print(
        Panel(
            "Let's set up NotionIQ. You'll need your Notion integration token, "
            "your Inbox database ID, and your Anthropic (Claude) API key.",
            title="[bold]NotionIQ init[/bold]",
            border_style="cyan",
            expand=False,
        )
    )

    env_path = Path.cwd() / ".env"
    if env_path.exists():
        console.print(
            f"[yellow]An existing .env was found at {env_path}. "
            "Unrelated keys will be preserved.[/yellow]"
        )

    def _prompt_required(label: str, *, password: bool, validator=None) -> str:
        while True:
            value = Prompt.ask(label, password=password).strip()
            if not value:
                console.print("[red]This value cannot be empty.[/red]")
                continue
            if validator is not None and not validator(value):
                console.print(
                    "[yellow]Value doesn't match the expected format, "
                    "but it will be saved anyway.[/yellow]"
                )
            return value

    notion_validator = (
        SecurityValidator.validate_notion_api_key if SecurityValidator else None
    )
    anthropic_validator = (
        SecurityValidator.validate_anthropic_api_key if SecurityValidator else None
    )

    notion_key = _prompt_required(
        "Notion API key (NOTION_API_KEY)",
        password=True,
        validator=notion_validator,
    )
    inbox_db = _prompt_required(
        "Notion Inbox database ID (NOTION_INBOX_DATABASE_ID)",
        password=False,
    )
    anthropic_key = _prompt_required(
        "Anthropic API key (ANTHROPIC_API_KEY)",
        password=True,
        validator=anthropic_validator,
    )

    _update_env_file(
        env_path,
        {
            "NOTION_API_KEY": notion_key,
            "NOTION_INBOX_DATABASE_ID": inbox_db,
            "ANTHROPIC_API_KEY": anthropic_key,
        },
    )

    console.print(
        Panel(
            f"Configuration saved to [bold]{env_path}[/bold].\n\n"
            "Next, run:\n\n    [bold cyan]notioniq[/bold cyan]\n\n"
            "to analyze your workspace (add [bold]--dry-run[/bold] to preview "
            "without changing Notion).",
            title="[bold green]Setup complete[/bold green]",
            border_style="green",
            expand=False,
        )
    )


if __name__ == "__main__":
    main()
