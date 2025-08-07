#!/usr/bin/env python3
"""
Test script for optimized NotionIQ with cost tracking
Run a small test batch to verify everything works with minimal API costs
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings
from notion_organizer import NotionOrganizer
from api_optimizer import OptimizationLevel
from cost_monitor import CostMonitor

console = Console()


async def test_with_cost_tracking():
    """Run a test with aggressive cost optimization"""
    
    console.print(Panel.fit(
        "[bold cyan]🧪 NotionIQ Test Run with Cost Optimization[/bold cyan]\n"
        "[dim]Testing with minimal API usage to verify setup[/dim]",
        border_style="cyan"
    ))
    
    try:
        # Load settings
        settings = get_settings()
        
        # Override for test - process only 3 pages
        settings.batch_size = 3
        
        console.print("\n[bold]Test Configuration:[/bold]")
        console.print(f"  • Batch Size: [cyan]3 pages[/cyan] (minimal test)")
        console.print(f"  • Optimization: [cyan]MINIMAL[/cyan] (aggressive cost savings)")
        console.print(f"  • Daily Budget: [cyan]$5.00[/cyan]")
        console.print(f"  • Caching: [cyan]Enabled[/cyan]")
        console.print()
        
        # Initialize cost monitor
        cost_monitor = CostMonitor(
            settings.data_dir,
            budget_limits={
                'hourly': 1.0,
                'daily': 5.0,
                'weekly': 25.0,
                'monthly': 100.0
            }
        )
        
        # Check current budget status
        budget_status = cost_monitor.get_budget_status()
        console.print("[bold]Current Budget Status:[/bold]")
        console.print(f"  • Daily Spent: [yellow]${budget_status['daily']['spent']:.2f}[/yellow]")
        console.print(f"  • Daily Remaining: [green]${budget_status['daily']['remaining']:.2f}[/green]")
        
        if budget_status['daily']['remaining'] < 0.10:
            console.print("\n[bold red]⚠️ Daily budget nearly exhausted! Aborting test.[/bold red]")
            return
        
        # Initialize organizer with minimal optimization
        console.print("\n[bold]Initializing NotionIQ with optimization...[/bold]")
        organizer = NotionOrganizer(
            settings,
            optimization_level=OptimizationLevel.MINIMAL,
            enable_cost_monitoring=True
        )
        
        # Test Notion connection
        console.print("\n[bold]Testing Notion connection...[/bold]")
        try:
            test_pages = organizer.notion.get_database_pages(
                settings.notion_inbox_database_id,
                limit=1
            )
            if test_pages:
                console.print("[green]✅ Notion connection successful![/green]")
                console.print(f"[dim]Found inbox database with pages[/dim]")
            else:
                console.print("[yellow]⚠️ Inbox database is empty[/yellow]")
                return
        except Exception as e:
            console.print(f"[red]❌ Notion connection failed: {e}[/red]")
            return
        
        # Run minimal analysis
        console.print("\n[bold]Running optimized analysis on 3 pages...[/bold]")
        console.print("[dim]This will use minimal tokens and aggressive caching[/dim]\n")
        
        report = await organizer.run_analysis(
            analyze_workspace=False,  # Skip workspace analysis for test
            process_inbox=True,
            create_recommendations=True,
            dry_run=True  # Don't update Notion during test
        )
        
        # Display cost summary
        console.print("\n[bold cyan]💰 Test Cost Summary:[/bold cyan]")
        if organizer.cost_monitor:
            final_metrics = organizer.cost_monitor.get_current_metrics()
            final_budget = organizer.cost_monitor.get_budget_status()
            
            console.print(f"  • API Calls Made: [cyan]{final_metrics.total_calls}[/cyan]")
            console.print(f"  • Total Tokens Used: [cyan]{final_metrics.total_input_tokens + final_metrics.total_output_tokens:,}[/cyan]")
            console.print(f"  • Test Cost: [bold green]${final_metrics.total_cost:.4f}[/bold green]")
            console.print(f"  • Cache Hits: [cyan]{final_metrics.cache_hits}[/cyan]")
            console.print(f"  • Cost Saved by Caching: [green]${final_metrics.cost_saved:.4f}[/green]")
            console.print(f"  • Daily Budget Used: [yellow]{final_budget['daily']['percentage']:.1f}%[/yellow]")
        
        # Display optimization metrics
        if hasattr(organizer.ai_analyzer, 'api_optimizer'):
            usage_report = organizer.ai_analyzer.api_optimizer.get_usage_report()
            console.print("\n[bold]Optimization Metrics:[/bold]")
            console.print(f"  • Duplicates Skipped: [cyan]{usage_report['duplicates_skipped']}[/cyan]")
            console.print(f"  • Similar Pages Skipped: [cyan]{usage_report['similar_pages_skipped']}[/cyan]")
            console.print(f"  • Tokens Saved: [cyan]{usage_report['tokens_saved']:,}[/cyan]")
        
        # Estimate full run cost
        if report and organizer.cost_monitor:
            pages_analyzed = len(report.get("detailed_analysis", []))
            if pages_analyzed > 0:
                cost_per_page = final_metrics.total_cost / pages_analyzed
                estimated_100_pages = cost_per_page * 100
                console.print(f"\n[bold]Estimated Costs:[/bold]")
                console.print(f"  • Per Page: [cyan]${cost_per_page:.4f}[/cyan]")
                console.print(f"  • 100 Pages: [cyan]${estimated_100_pages:.2f}[/cyan]")
                console.print(f"  • 1000 Pages: [cyan]${estimated_100_pages * 10:.2f}[/cyan]")
        
        console.print("\n[bold green]✅ Test completed successfully![/bold green]")
        console.print("[dim]Ready to run full analysis with optimizations enabled[/dim]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Test failed: {e}[/bold red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    console.print("\n[bold]Starting NotionIQ Optimized Test Run[/bold]")
    console.print("[dim]This test will process only 3 pages with aggressive optimization[/dim]\n")
    
    asyncio.run(test_with_cost_tracking())
    
    console.print("\n[bold]Test Complete![/bold]")
    console.print("\nTo run full analysis with optimization:")
    console.print("[cyan]python notion_organizer.py --optimization minimal --batch-size 10[/cyan]")
    console.print("\nOr for more complete analysis:")
    console.print("[cyan]python notion_organizer.py --optimization balanced --daily-budget 10[/cyan]")