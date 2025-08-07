#!/usr/bin/env python3
"""
NotionIQ - Intelligent Notion Workspace Organizer
Main orchestrator script that coordinates all components
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint
from logger_wrapper import logger

from config import get_settings, Settings
from notion_wrapper import NotionAdvancedClient
from ai_analyzer import UniversalAIAnalyzer
from workspace_analyzer import WorkspaceAnalyzer
from api_optimizer import OptimizationLevel
from api_auto_config import get_auto_config
from ai_providers import AIProviderManager
from cost_monitor import CostMonitor
from advanced_optimizer import OptimizationConfig
from recommendation_executor import RecommendationExecutor


console = Console()


class NotionOrganizer:
    """Main orchestrator for NotionIQ"""
    
    def __init__(
        self, 
        settings: Optional[Settings] = None, 
        optimization_level: OptimizationLevel = OptimizationLevel.MINIMAL,
        preferred_provider: Optional[str] = None,
        enable_cost_monitoring: bool = True
    ):
        """Initialize the organizer"""
        self.settings = settings or get_settings()
        self.optimization_level = optimization_level
        
        # Initialize cost monitoring
        self.cost_monitor = None
        if enable_cost_monitoring:
            self.cost_monitor = CostMonitor(
                self.settings.data_dir,
                budget_limits={
                    'hourly': 2.0,
                    'daily': 5.0,
                    'weekly': 25.0,
                    'monthly': 100.0
                }
            )
        
        # Initialize components
        self.notion = NotionAdvancedClient(self.settings)
        # Use Universal AI Analyzer with automatic provider selection
        self.ai_analyzer = UniversalAIAnalyzer(
            self.settings, 
            optimization_level=optimization_level,
            preferred_provider=preferred_provider
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
            level=self.settings.log_level.value
        )
        
        if self.settings.is_development:
            logger.add(sys.stderr, level="DEBUG")
        
        logger.info("NotionOrganizer initialized")
    
    async def run_analysis(
        self,
        analyze_workspace: bool = True,
        process_inbox: bool = True,
        create_recommendations: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Run the complete analysis workflow"""
        
        console.print(Panel.fit(
            "[bold cyan]🚀 NotionIQ - Intelligent Workspace Organizer[/bold cyan]\n"
            "[dim]Analyzing your Notion workspace for optimization opportunities[/dim]",
            border_style="cyan"
        ))
        
        # Step 1: Workspace Analysis
        if analyze_workspace:
            console.print("\n[bold]Step 1: Analyzing Workspace Structure[/bold]")
            self.workspace_analysis = await self.workspace_analyzer.analyze_workspace(
                deep_scan=self.settings.enable_workspace_scan
            )
        
        # Step 2: Process Inbox Pages
        if process_inbox:
            console.print("\n[bold]Step 2: Processing Inbox Pages[/bold]")
            await self._process_inbox_pages()
        
        # Step 3: Generate Recommendations
        if create_recommendations:
            console.print("\n[bold]Step 3: Generating Recommendations[/bold]")
            self._generate_recommendations()
        
        # Step 4: Execute Recommendations (if enabled and not dry run)
        execution_results = {}
        if (not dry_run and 
            self.report_data.get("recommendations") and 
            (self.settings.enable_auto_organization or self.settings.auto_execute)):
            
            console.print("\n[bold]Step 4: Executing Recommendations[/bold]")
            execution_results = await self.executor.execute_recommendations(
                self.report_data["recommendations"],
                dry_run=dry_run,
                interactive=not self.settings.auto_execute
            )
        
        # Step 5: Create Report
        console.print("\n[bold]Step 5: Creating Report[/bold]")
        self.report_data = self._create_report()
        if execution_results:
            self.report_data["execution_results"] = execution_results
        
        # Step 6: Save Results
        self._save_results()
        
        # Step 7: Update Notion (if not dry run)
        if not dry_run and self.settings.enable_recommendations_page:
            console.print("\n[bold]Step 6: Updating Notion Recommendations Page[/bold]")
            await self._update_notion_recommendations()
        
        # Display summary
        self._display_summary()
        
        return self.report_data
    
    async def _process_inbox_pages(self):
        """Process pages from the inbox database"""
        
        # Get pages from inbox
        console.print(f"\n[dim]Fetching pages from Inbox database...[/dim]")
        
        try:
            inbox_pages = self.notion.get_database_pages(
                self.settings.notion_inbox_database_id,
                limit=None if not self.settings.batch_size else self.settings.batch_size
            )
            
            if not inbox_pages:
                console.print("[yellow]No pages found in Inbox database[/yellow]")
                return
            
            console.print(f"[green]Found {len(inbox_pages)} pages to process[/green]")
            
            # Get full content for each page
            pages_with_content = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Fetching page content...",
                    total=len(inbox_pages)
                )
                
                for page in inbox_pages:
                    page_content = self.notion.get_page_content(page["id"])
                    pages_with_content.append(page_content)
                    progress.update(task, advance=1)
            
            # Analyze pages with AI
            self.analysis_results = self.ai_analyzer.batch_analyze(
                pages_with_content,
                self.workspace_analysis
            )
            
            logger.info(f"Analyzed {len(self.analysis_results)} pages")
            
        except Exception as e:
            logger.error(f"Error processing inbox pages: {e}")
            console.print(f"[red]Error: {e}[/red]")
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis"""
        
        if not self.analysis_results:
            console.print("[yellow]No analysis results to generate recommendations from[/yellow]")
            return
        
        recommendations = {
            "immediate_actions": [],
            "suggested_moves": [],
            "archive_candidates": [],
            "delete_candidates": [],
            "review_needed": [],
            "merge_candidates": []
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
                "classification": analysis.get("classification", {}).get("primary_type"),
                "urgency": analysis.get("urgency_assessment", {}).get("level")
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
        
        # Display recommendation counts
        console.print("\n[bold cyan]📊 Recommendation Summary:[/bold cyan]")
        for category, items in recommendations.items():
            if items:
                console.print(f"  • {category.replace('_', ' ').title()}: {len(items)} pages")
    
    def _create_report(self) -> Dict[str, Any]:
        """Create comprehensive analysis report"""
        
        report = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "settings": {
                    "batch_size": self.settings.batch_size,
                    "enable_workspace_scan": self.settings.enable_workspace_scan,
                    "enable_pattern_learning": self.settings.enable_pattern_learning
                }
            },
            "workspace_summary": {
                "total_databases": len(self.workspace_analysis.get("workspace_structure", {}).get("databases", {})),
                "total_pages_analyzed": len(self.analysis_results),
                "health_score": self.workspace_analysis.get("health_score", 0),
                "metrics": self.workspace_analysis.get("metrics", {})
            },
            "classification_summary": self._summarize_classifications(),
            "recommendations": self.report_data.get("recommendations", {}),
            "insights": self._generate_insights(),
            "detailed_analysis": self.analysis_results
        }
        
        return report
    
    def _summarize_classifications(self) -> Dict[str, Any]:
        """Summarize classification results"""
        
        summary = {
            "document_types": {},
            "urgency_levels": {},
            "contexts": {},
            "confidence_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
        
        for analysis in self.analysis_results:
            if "error" in analysis:
                continue
            
            # Document type
            doc_type = analysis.get("classification", {}).get("primary_type", "unknown")
            summary["document_types"][doc_type] = summary["document_types"].get(doc_type, 0) + 1
            
            # Urgency
            urgency = analysis.get("urgency_assessment", {}).get("level", "unknown")
            summary["urgency_levels"][urgency] = summary["urgency_levels"].get(urgency, 0) + 1
            
            # Context
            context = analysis.get("context_detection", {}).get("primary_context", "unknown")
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
            most_common = max(classification_summary["document_types"].items(), key=lambda x: x[1])
            insights.append({
                "type": "pattern",
                "title": "Most Common Content Type",
                "description": f"Your inbox primarily contains {most_common[0]} items ({most_common[1]} pages)",
                "action": f"Consider creating dedicated workflows for {most_common[0]} processing"
            })
        
        # Urgency distribution
        urgent_count = classification_summary["urgency_levels"].get("immediate", 0)
        if urgent_count > 0:
            insights.append({
                "type": "alert",
                "title": "Urgent Items Detected",
                "description": f"Found {urgent_count} items requiring immediate attention",
                "action": "Review and act on urgent items first"
            })
        
        # Low confidence items
        low_confidence = classification_summary["confidence_distribution"]["low"]
        if low_confidence > len(self.analysis_results) * 0.3:
            insights.append({
                "type": "improvement",
                "title": "Content Structure Needs Improvement",
                "description": f"{low_confidence} pages have unclear structure or purpose",
                "action": "Add clearer titles and descriptions to improve classification accuracy"
            })
        
        # Workspace health
        health_score = self.workspace_analysis.get("health_score", 0)
        if health_score < 60:
            insights.append({
                "type": "health",
                "title": "Workspace Organization Needs Attention",
                "description": f"Current health score is {health_score:.0f}/100",
                "action": "Follow recommendations to improve workspace structure"
            })
        
        return insights
    
    def _save_results(self):
        """Save analysis results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = self.settings.output_dir / f"analysis_report_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        console.print(f"\n[green]✅ Report saved to: {json_file}[/green]")
        
        # Save summary for quick review
        summary_file = self.settings.output_dir / f"summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write("NotionIQ Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Pages Analyzed: {len(self.analysis_results)}\n")
            f.write(f"Health Score: {self.workspace_analysis.get('health_score', 0):.1f}/100\n\n")
            
            f.write("Recommendations:\n")
            for category, items in self.report_data.get("recommendations", {}).items():
                if items:
                    f.write(f"  • {category.replace('_', ' ').title()}: {len(items)} pages\n")
            
            f.write("\nTop Insights:\n")
            for insight in self.report_data.get("insights", [])[:5]:
                f.write(f"  • {insight.get('title', '')}: {insight.get('description', '')}\n")
        
        logger.info(f"Results saved to {self.settings.output_dir}")
    
    async def _update_notion_recommendations(self):
        """Update recommendations page in Notion"""
        
        try:
            # Create recommendations content
            recommendations_content = self._format_recommendations_for_notion()
            
            # Search for existing recommendations page
            search_response = self.notion.client.search(
                query="NotionIQ Recommendations",
                filter={"property": "object", "value": "page"}
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
                    block_id=page_id,
                    children=recommendations_content
                )
                
                console.print(f"[green]✅ Updated recommendations page: {page_id}[/green]")
            else:
                # Create new page in the workspace root
                new_page = self.notion.client.pages.create(
                    parent={"workspace": True},
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
                    children=recommendations_content
                )
                
                console.print(f"[green]✅ Created recommendations page: {new_page['id']}[/green]")
                console.print(f"[dim]View at: {new_page.get('url', 'N/A')}[/dim]")
            
            logger.info("Successfully updated Notion recommendations page")
            
        except Exception as e:
            logger.error(f"Failed to update Notion recommendations page: {e}")
            console.print(f"[yellow]⚠️ Could not update Notion recommendations page: {e}[/yellow]")
    
    def _format_recommendations_for_notion(self) -> List[Dict[str, Any]]:
        """Format recommendations as Notion blocks"""
        blocks = []
        
        # Add header
        blocks.append({
            "object": "block",
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": "🚀 NotionIQ Analysis Report"}
                }]
            }
        })
        
        # Add timestamp
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{
                    "type": "text",
                    "text": {
                        "content": f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    },
                    "annotations": {"italic": True}
                }]
            }
        })
        
        # Add divider
        blocks.append({"object": "block", "type": "divider", "divider": {}})
        
        # Add health score
        health_score = self.workspace_analysis.get("health_score", 0)
        blocks.append({
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{
                    "type": "text",
                    "text": {
                        "content": f"Workspace Health Score: {health_score:.1f}/100"
                    },
                    "annotations": {"bold": True}
                }],
                "icon": {"emoji": "💯"}
            }
        })
        
        # Add summary statistics
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": "📊 Summary"}
                }]
            }
        })
        
        stats_text = f"""• Pages Analyzed: {len(self.analysis_results)}
• Databases Found: {len(self.workspace_analysis.get('workspace_structure', {}).get('databases', {}))}
• Total Pages in Workspace: {self.workspace_analysis.get('workspace_structure', {}).get('total_pages', 'Unknown')}"""
        
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": stats_text}
                }]
            }
        })
        
        # Add recommendations sections
        recommendations = self.report_data.get("recommendations", {})
        
        if recommendations.get("immediate_actions"):
            blocks.extend(self._format_recommendation_section(
                "⚡ Immediate Actions Required",
                recommendations["immediate_actions"],
                "red"
            ))
        
        if recommendations.get("suggested_moves"):
            blocks.extend(self._format_recommendation_section(
                "📦 Suggested Page Moves",
                recommendations["suggested_moves"][:10],  # Limit to top 10
                "yellow"
            ))
        
        if recommendations.get("archive_candidates"):
            blocks.extend(self._format_recommendation_section(
                "🗄️ Archive Candidates",
                recommendations["archive_candidates"][:10],
                "gray"
            ))
        
        # Add insights
        insights = self.report_data.get("insights", [])
        if insights:
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": "💡 Key Insights"}
                    }]
                }
            })
            
            for insight in insights[:5]:
                blocks.append({
                    "object": "block",
                    "type": "toggle",
                    "toggle": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": insight.get("title", "")},
                            "annotations": {"bold": True}
                        }],
                        "children": [{
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{
                                    "type": "text",
                                    "text": {"content": insight.get("description", "")}
                                }]
                            }
                        }]
                    }
                })
        
        return blocks
    
    def _format_recommendation_section(
        self,
        title: str,
        items: List[Dict[str, Any]],
        color: str = "default"
    ) -> List[Dict[str, Any]]:
        """Format a recommendation section for Notion"""
        blocks = []
        
        # Add section header
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": title}
                }]
            }
        })
        
        # Add items as a bulleted list
        for item in items:
            text = f"{item.get('title', 'Untitled')}"
            if item.get('action'):
                text += f" → {item.get('action')}"
            if item.get('suggested_database'):
                text += f" (to {item.get('suggested_database')})"
            
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{
                        "type": "text",
                        "text": {"content": text}
                    }]
                }
            })
            
            # Add reasoning as nested item if available
            if item.get('reasoning'):
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{
                            "type": "text",
                            "text": {"content": f"→ {item['reasoning']}"},
                            "annotations": {"italic": True, "color": "gray"}
                        }]
                    }
                })
        
        return blocks
    
    def _display_summary(self):
        """Display final summary"""
        
        console.print("\n" + "=" * 60)
        console.print(Panel.fit(
            "[bold green]✨ Analysis Complete![/bold green]\n\n"
            f"📊 Pages Analyzed: {len(self.analysis_results)}\n"
            f"💯 Health Score: {self.workspace_analysis.get('health_score', 0):.1f}/100\n"
            f"💡 Insights Generated: {len(self.report_data.get('insights', []))}\n"
            f"📁 Report saved to: {self.settings.output_dir}",
            border_style="green"
        ))
        
        # Display cost monitoring dashboard if enabled
        if self.cost_monitor:
            console.print("\n[bold cyan]💰 Cost Analytics:[/bold cyan]")
            metrics = self.cost_monitor.get_current_metrics()
            budget_status = self.cost_monitor.get_budget_status()
            
            # Display key metrics
            console.print(f"  Total API Cost: [bold green]${metrics.total_cost:.2f}[/bold green]")
            console.print(f"  Cache Savings: [bold green]${metrics.cost_saved:.2f}[/bold green]")
            console.print(f"  Daily Budget Used: [bold yellow]{budget_status['daily']['percentage']:.1f}%[/bold yellow]")
            console.print(f"  Remaining Today: [bold cyan]${budget_status['daily']['remaining']:.2f}[/bold cyan]")
        
        # Show top recommendations
        recommendations = self.report_data.get("recommendations", {})
        if recommendations.get("immediate_actions"):
            console.print("\n[bold red]⚡ Immediate Actions Required:[/bold red]")
            for item in recommendations["immediate_actions"][:3]:
                console.print(f"  • {item['title']}: {item['action']}")
        
        if recommendations.get("suggested_moves"):
            console.print("\n[bold yellow]📦 Suggested Moves:[/bold yellow]")
            for item in recommendations["suggested_moves"][:5]:
                console.print(
                    f"  • Move '{item['title']}' to {item.get('suggested_database', 'appropriate database')}"
                )


@click.command()
@click.option(
    "--analyze-workspace/--skip-workspace",
    default=True,
    help="Perform deep workspace analysis"
)
@click.option(
    "--process-inbox/--skip-inbox",
    default=True,
    help="Process inbox pages"
)
@click.option(
    "--create-recommendations/--skip-recommendations",
    default=True,
    help="Generate recommendations"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run analysis without making changes to Notion"
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Number of pages to process (overrides config)"
)
@click.option(
    "--optimization",
    type=click.Choice(["auto", "minimal", "balanced", "full"]),
    default="minimal",
    help="API optimization level (minimal=aggressive cost savings, balanced=moderate, full=complete analysis)"
)
@click.option(
    "--provider",
    type=click.Choice(["auto", "claude", "chatgpt", "gemini"]),
    default="auto",
    help="AI provider to use (auto=automatic selection based on available APIs)"
)
@click.option(
    "--cost-monitor/--no-cost-monitor",
    default=True,
    help="Enable real-time cost monitoring and budget enforcement"
)
@click.option(
    "--daily-budget",
    type=float,
    default=5.0,
    help="Maximum daily API spend in USD (default: $5)"
)
@click.option(
    "--auto-execute",
    is_flag=True,
    help="Automatically execute recommendations without confirmation"
)
def main(
    analyze_workspace: bool,
    process_inbox: bool,
    create_recommendations: bool,
    dry_run: bool,
    batch_size: Optional[int],
    optimization: str,
    provider: str,
    cost_monitor: bool,
    daily_budget: float,
    auto_execute: bool
):
    """NotionIQ - Intelligent Notion Workspace Organizer"""
    
    try:
        # Load settings
        settings = get_settings()
        
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
            optimization_level = OptimizationLevel[auto_config["optimization_level"].upper()]
            settings.batch_size = auto_config["batch_size"] if not batch_size else batch_size
            settings.rate_limit_requests_per_second = auto_config["rate_limit_rps"]
            settings.enable_caching = auto_config["enable_caching"]
            settings.cache_ttl_hours = auto_config["cache_ttl_hours"]
            
            console.print(f"[green]✅ Using {auto_config['model']} with {auto_config['optimization_level']} optimization[/green]")
            console.print(f"[dim]Estimated cost: ${auto_config['estimated_cost_per_100']}/100 pages[/dim]\n")
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
            console.print(f"[yellow]⚡ Auto-execute enabled - recommendations will be applied automatically[/yellow]")
        
        # Initialize organizer with optimization level and provider
        organizer = NotionOrganizer(
            settings, 
            optimization_level=optimization_level,
            preferred_provider=preferred_provider,
            enable_cost_monitoring=cost_monitor
        )
        
        # Update budget limits if cost monitoring is enabled
        if cost_monitor and organizer.cost_monitor:
            organizer.cost_monitor.budget_limits['daily'] = daily_budget
            organizer.cost_monitor.budget_limits['weekly'] = daily_budget * 7
            organizer.cost_monitor.budget_limits['monthly'] = daily_budget * 30
        
        # Run analysis
        asyncio.run(organizer.run_analysis(
            analyze_workspace=analyze_workspace,
            process_inbox=process_inbox,
            create_recommendations=create_recommendations,
            dry_run=dry_run
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        logger.exception("Fatal error in main")
        sys.exit(1)


if __name__ == "__main__":
    main()