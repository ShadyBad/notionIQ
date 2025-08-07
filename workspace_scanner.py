#!/usr/bin/env python3
"""
Full Workspace Scanner for NotionIQ
Analyzes all databases and pages in your entire Notion workspace
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

from config import get_settings
from notion_wrapper import NotionAdvancedClient
from ai_analyzer import AIAnalyzer
from workspace_analyzer import WorkspaceAnalyzer
from api_optimizer import APIOptimizer
from cost_monitor import CostMonitor

console = Console()


class WorkspaceScanner:
    """Comprehensive scanner for entire Notion workspace"""
    
    def __init__(self):
        """Initialize the workspace scanner"""
        self.settings = get_settings()
        self.notion = NotionAdvancedClient(self.settings)
        self.ai_analyzer = AIAnalyzer(self.settings)
        self.workspace_analyzer = WorkspaceAnalyzer(self.notion)
        self.api_optimizer = APIOptimizer(self.settings)
        self.cost_monitor = CostMonitor(self.settings)
        
        self.all_pages = []
        self.analysis_results = {}
        self.database_summaries = {}
        
    async def scan_entire_workspace(self, 
                                   max_pages_per_db: Optional[int] = None,
                                   skip_databases: Optional[List[str]] = None,
                                   target_databases: Optional[List[str]] = None):
        """
        Scan and analyze the entire workspace
        
        Args:
            max_pages_per_db: Maximum pages to analyze per database
            skip_databases: List of database names to skip
            target_databases: If specified, only analyze these databases
        """
        skip_databases = skip_databases or []
        
        console.print(Panel.fit(
            "🌍 [bold cyan]Full Workspace Analysis[/bold cyan]\n"
            "Scanning all databases and pages in your Notion workspace",
            border_style="cyan"
        ))
        
        # Step 1: Get all databases
        console.print("\n[bold]Step 1: Discovering all databases...[/bold]")
        databases = await self._get_all_databases()
        
        # Filter databases
        if target_databases:
            databases = {k: v for k, v in databases.items() 
                        if v.get('title') in target_databases}
            console.print(f"[yellow]Targeting {len(databases)} specific databases[/yellow]")
        else:
            databases = {k: v for k, v in databases.items() 
                        if v.get('title') not in skip_databases}
        
        console.print(f"[green]Found {len(databases)} databases to analyze[/green]")
        
        # Step 2: Analyze each database
        console.print("\n[bold]Step 2: Analyzing databases and their pages...[/bold]")
        
        total_pages_analyzed = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task(
                "Analyzing databases...", 
                total=len(databases)
            )
            
            for db_id, db_info in databases.items():
                db_name = db_info.get('title', 'Untitled')
                
                # Skip if in skip list
                if db_name in skip_databases:
                    progress.advance(task)
                    continue
                
                progress.update(task, description=f"Analyzing {db_name}...")
                
                # Get pages from this database
                try:
                    pages = self.notion.get_database_pages(
                        db_id, 
                        limit=max_pages_per_db
                    )
                    
                    if pages:
                        # Analyze pages in this database
                        db_analysis = await self._analyze_database_pages(
                            db_id, db_name, pages
                        )
                        
                        self.database_summaries[db_name] = db_analysis
                        total_pages_analyzed += len(pages)
                        
                        console.print(
                            f"  [dim]✓ {db_name}: {len(pages)} pages analyzed[/dim]"
                        )
                    else:
                        self.database_summaries[db_name] = {
                            "page_count": 0,
                            "status": "empty",
                            "classifications": []
                        }
                        console.print(f"  [dim]- {db_name}: Empty database[/dim]")
                        
                except Exception as e:
                    logger.error(f"Error analyzing database {db_name}: {e}")
                    console.print(f"  [red]✗ {db_name}: Error - {str(e)[:50]}[/red]")
                
                progress.advance(task)
        
        # Step 3: Generate workspace-wide insights
        console.print("\n[bold]Step 3: Generating workspace insights...[/bold]")
        insights = await self._generate_workspace_insights()
        
        # Step 4: Create comprehensive report
        console.print("\n[bold]Step 4: Creating comprehensive report...[/bold]")
        report = self._create_workspace_report(total_pages_analyzed, insights)
        
        # Save report
        self._save_report(report)
        
        # Display summary
        self._display_summary(total_pages_analyzed, insights)
        
        return report
    
    async def _get_all_databases(self) -> Dict[str, Any]:
        """Get all databases in the workspace"""
        try:
            # Use the workspace scan to get all databases
            workspace_data = self.notion.scan_workspace()
            return workspace_data.get('databases', {})
        except Exception as e:
            logger.error(f"Error getting databases: {e}")
            return {}
    
    async def _analyze_database_pages(self, 
                                     db_id: str, 
                                     db_name: str, 
                                     pages: List[Dict]) -> Dict:
        """Analyze all pages in a database"""
        
        classifications = []
        page_summaries = []
        
        for page in pages:
            try:
                # Get page content
                page_id = page['id']
                content = self.notion.get_page_content(page_id)
                
                # Optimize content for AI analysis
                optimized_content = self.api_optimizer.optimize_page_content(
                    content, page
                )
                
                # Analyze with AI
                if optimized_content.get('content'):
                    analysis = await self.ai_analyzer.analyze_page(
                        page, optimized_content['content']
                    )
                    
                    classifications.append({
                        'page_id': page_id,
                        'title': self.notion.extract_page_title(page),
                        'type': analysis.get('type'),
                        'confidence': analysis.get('confidence'),
                        'recommendations': analysis.get('recommendations', [])
                    })
                    
                    # Track for cost monitoring
                    self.all_pages.append(analysis)
                    
            except Exception as e:
                logger.error(f"Error analyzing page {page_id}: {e}")
        
        return {
            'database_id': db_id,
            'database_name': db_name,
            'page_count': len(pages),
            'classifications': classifications,
            'summary': self._summarize_database_content(classifications)
        }
    
    def _summarize_database_content(self, classifications: List[Dict]) -> Dict:
        """Create a summary of database content"""
        
        if not classifications:
            return {'status': 'empty'}
        
        # Count document types
        type_counts = {}
        for item in classifications:
            doc_type = item.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        # Calculate average confidence
        confidences = [c.get('confidence', 0) for c in classifications]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'document_types': type_counts,
            'average_confidence': avg_confidence,
            'total_pages': len(classifications),
            'needs_review': sum(1 for c in classifications 
                               if c.get('confidence', 0) < 0.7)
        }
    
    async def _generate_workspace_insights(self) -> Dict:
        """Generate insights across the entire workspace"""
        
        insights = {
            'workspace_patterns': {},
            'optimization_opportunities': [],
            'content_distribution': {},
            'database_health': {}
        }
        
        # Analyze content distribution
        all_types = {}
        for db_name, db_data in self.database_summaries.items():
            if 'summary' in db_data and 'document_types' in db_data['summary']:
                for doc_type, count in db_data['summary']['document_types'].items():
                    all_types[doc_type] = all_types.get(doc_type, 0) + count
        
        insights['content_distribution'] = all_types
        
        # Identify optimization opportunities
        for db_name, db_data in self.database_summaries.items():
            if db_data.get('page_count', 0) == 0:
                insights['optimization_opportunities'].append({
                    'database': db_name,
                    'issue': 'Empty database',
                    'recommendation': 'Consider removing or populating this database'
                })
            elif db_data.get('summary', {}).get('needs_review', 0) > 5:
                insights['optimization_opportunities'].append({
                    'database': db_name,
                    'issue': 'Many uncertain classifications',
                    'recommendation': 'Review and better organize content in this database'
                })
        
        # Add workspace patterns from analyzer
        if hasattr(self, 'workspace_analyzer'):
            workspace_analysis = await self.workspace_analyzer.analyze_workspace()
            insights['workspace_patterns'] = workspace_analysis.get('patterns', {})
            insights['database_health'] = workspace_analysis.get('metrics', {})
        
        return insights
    
    def _create_workspace_report(self, 
                                total_pages: int, 
                                insights: Dict) -> Dict:
        """Create comprehensive workspace report"""
        
        report = {
            'scan_timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_databases_analyzed': len(self.database_summaries),
                'total_pages_analyzed': total_pages,
                'empty_databases': sum(1 for d in self.database_summaries.values() 
                                     if d.get('page_count', 0) == 0),
                'api_cost': self.cost_monitor.get_session_cost(),
                'cache_savings': self.api_optimizer.get_cache_savings()
            },
            'database_summaries': self.database_summaries,
            'insights': insights,
            'recommendations': self._generate_recommendations(insights),
            'api_usage': {
                'total_calls': self.ai_analyzer.api_call_count,
                'total_tokens': self.ai_analyzer.total_tokens,
                'cost_breakdown': self.cost_monitor.get_cost_breakdown()
            }
        }
        
        return report
    
    def _generate_recommendations(self, insights: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Based on optimization opportunities
        for opp in insights.get('optimization_opportunities', []):
            recommendations.append({
                'priority': 'medium',
                'category': 'organization',
                'title': f"Optimize {opp['database']}",
                'description': opp['recommendation'],
                'impact': 'Improved workspace organization'
            })
        
        # Based on content distribution
        content_dist = insights.get('content_distribution', {})
        if len(content_dist) > 10:
            recommendations.append({
                'priority': 'high',
                'category': 'structure',
                'title': 'Consolidate document types',
                'description': f'You have {len(content_dist)} different document types. Consider consolidating similar types.',
                'impact': 'Simplified navigation and better organization'
            })
        
        # Based on database health
        health = insights.get('database_health', {})
        if health.get('organization_score', 100) < 50:
            recommendations.append({
                'priority': 'high',
                'category': 'health',
                'title': 'Improve workspace organization',
                'description': 'Your workspace organization score is low. Focus on consistent naming and proper categorization.',
                'impact': 'Better findability and reduced cognitive load'
            })
        
        return recommendations
    
    def _save_report(self, report: Dict):
        """Save the report to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workspace_report_{timestamp}.json"
        filepath = self.settings.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        console.print(f"\n[green]✅ Report saved to: {filepath}[/green]")
    
    def _display_summary(self, total_pages: int, insights: Dict):
        """Display analysis summary"""
        
        # Create summary table
        table = Table(title="Workspace Analysis Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Databases Analyzed", str(len(self.database_summaries)))
        table.add_row("Total Pages Analyzed", str(total_pages))
        table.add_row("Empty Databases", 
                     str(sum(1 for d in self.database_summaries.values() 
                           if d.get('page_count', 0) == 0)))
        table.add_row("Document Types Found", 
                     str(len(insights.get('content_distribution', {}))))
        table.add_row("Optimization Opportunities", 
                     str(len(insights.get('optimization_opportunities', []))))
        table.add_row("API Cost", f"${self.cost_monitor.get_session_cost():.2f}")
        
        console.print("\n")
        console.print(table)
        
        # Show top recommendations
        console.print("\n[bold]📋 Top Recommendations:[/bold]")
        recommendations = self._generate_recommendations(insights)
        for i, rec in enumerate(recommendations[:5], 1):
            console.print(f"\n{i}. [bold]{rec['title']}[/bold]")
            console.print(f"   {rec['description']}")
            console.print(f"   [dim]Impact: {rec['impact']}[/dim]")


async def main():
    """Main function to run workspace scanner"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scan and analyze your entire Notion workspace"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to analyze per database"
    )
    parser.add_argument(
        "--skip-databases",
        nargs="+",
        default=[],
        help="Database names to skip"
    )
    parser.add_argument(
        "--target-databases",
        nargs="+",
        default=None,
        help="Only analyze these specific databases"
    )
    parser.add_argument(
        "--daily-budget",
        type=float,
        default=10.0,
        help="Daily budget limit for API calls"
    )
    
    args = parser.parse_args()
    
    # Set daily budget
    cost_monitor = CostMonitor(get_settings())
    cost_monitor.set_daily_budget(args.daily_budget)
    
    # Run scanner
    scanner = WorkspaceScanner()
    
    try:
        report = await scanner.scan_entire_workspace(
            max_pages_per_db=args.max_pages,
            skip_databases=args.skip_databases,
            target_databases=args.target_databases
        )
        
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "✨ [bold green]Workspace Analysis Complete![/bold green]\n\n"
            f"Check the output folder for your detailed report.",
            border_style="green"
        ))
        
    except Exception as e:
        logger.error(f"Error during workspace scan: {e}")
        console.print(f"\n[red]Error: {e}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())