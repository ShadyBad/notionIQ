"""
Workspace Analyzer for deep workspace intelligence
Analyzes workspace structure, patterns, and provides insights
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
import statistics
import json
from pathlib import Path

from logger_wrapper import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from notion_wrapper import NotionAdvancedClient
from config import Settings, get_settings, HEALTH_METRICS_CONFIG


console = Console()


class WorkspaceAnalyzer:
    """Deep workspace analysis and intelligence engine"""
    
    def __init__(
        self,
        notion_client: NotionAdvancedClient,
        settings: Optional[Settings] = None
    ):
        """Initialize workspace analyzer"""
        self.notion = notion_client
        self.settings = settings or get_settings()
        
        # Analysis results storage
        self.workspace_data = {}
        self.patterns = {}
        self.metrics = {}
        self.recommendations = []
        
        logger.info("WorkspaceAnalyzer initialized")
    
    async def analyze_workspace(self, deep_scan: bool = True) -> Dict[str, Any]:
        """Perform comprehensive workspace analysis"""
        
        console.print("\n[bold cyan]🔍 Starting Workspace Analysis...[/bold cyan]\n")
        
        # Load or scan workspace structure
        if deep_scan or not self.notion.workspace_structure:
            await self._scan_workspace_structure()
        else:
            self.notion.load_workspace_structure()
        
        # Analyze different aspects
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            
            tasks = [
                ("Analyzing database structures", self._analyze_database_structures),
                ("Detecting content patterns", self._analyze_content_patterns),
                ("Analyzing relationships", self._analyze_relationships),
                ("Calculating health metrics", self._calculate_health_metrics),
                ("Generating insights", self._generate_insights),
            ]
            
            for description, task_func in tasks:
                task = progress.add_task(description, total=None)
                await task_func()
                progress.update(task, completed=True)
        
        # Compile results
        self.workspace_data = {
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "workspace_structure": self.notion.workspace_structure,
            "patterns": self.patterns,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "health_score": self._calculate_overall_health_score()
        }
        
        # Save analysis results
        self._save_analysis_results()
        
        # Display summary
        self._display_analysis_summary()
        
        return self.workspace_data
    
    async def _scan_workspace_structure(self):
        """Scan workspace for databases and structure"""
        logger.info("Performing deep workspace scan...")
        await self.notion.scan_workspace()
    
    async def _analyze_database_structures(self):
        """Analyze database structures and properties"""
        
        if not self.notion.workspace_structure:
            return
        
        databases = self.notion.workspace_structure.get("databases", {})
        
        analysis = {
            "total_databases": len(databases),
            "database_types": {},
            "property_usage": Counter(),
            "database_sizes": {},
            "empty_databases": [],
            "large_databases": []
        }
        
        for db_id, db_data in databases.items():
            db_title = db_data.get("title", "Untitled")
            page_count = db_data.get("page_count", 0)
            
            # Track database sizes
            analysis["database_sizes"][db_title] = page_count
            
            if page_count == 0:
                analysis["empty_databases"].append(db_title)
            elif page_count > 100:
                analysis["large_databases"].append({
                    "title": db_title,
                    "count": page_count
                })
            
            # Analyze property usage
            for prop_name, prop_data in db_data.get("properties", {}).items():
                prop_type = prop_data.get("type")
                analysis["property_usage"][prop_type] += 1
            
            # Detect database type based on title and properties
            db_type = self._infer_database_type(db_title, db_data.get("properties", {}))
            analysis["database_types"][db_type] = analysis["database_types"].get(db_type, 0) + 1
        
        self.patterns["database_analysis"] = analysis
        logger.info(f"Analyzed {len(databases)} databases")
    
    def _infer_database_type(self, title: str, properties: Dict) -> str:
        """Infer database type from title and properties"""
        
        title_lower = title.lower()
        
        # Check common patterns
        if any(word in title_lower for word in ["task", "todo", "action"]):
            return "tasks"
        elif any(word in title_lower for word in ["project", "initiative"]):
            return "projects"
        elif any(word in title_lower for word in ["meeting", "note", "minutes"]):
            return "meetings"
        elif any(word in title_lower for word in ["idea", "brainstorm"]):
            return "ideas"
        elif any(word in title_lower for word in ["goal", "objective", "okr"]):
            return "goals"
        elif any(word in title_lower for word in ["contact", "people", "person"]):
            return "contacts"
        elif any(word in title_lower for word in ["resource", "reference", "wiki"]):
            return "resources"
        elif any(word in title_lower for word in ["inbox", "capture", "quick"]):
            return "inbox"
        else:
            return "other"
    
    async def _analyze_content_patterns(self):
        """Analyze content patterns across the workspace"""
        
        # Sample pages from inbox for pattern detection
        inbox_pages = self.notion.get_database_pages(
            self.settings.notion_inbox_database_id,
            limit=50  # Sample size
        )
        
        patterns = {
            "content_lengths": [],
            "title_patterns": Counter(),
            "creation_times": [],
            "update_frequency": [],
            "common_words": Counter(),
            "property_fill_rates": defaultdict(list)
        }
        
        for page in inbox_pages:
            # Get page details
            page_content = self.notion.get_page_content(page["id"])
            
            # Analyze content length
            content = page_content.get("content", "")
            patterns["content_lengths"].append(len(content))
            
            # Analyze title patterns
            title = page_content.get("title", "")
            if title:
                # Check for common prefixes
                if ":" in title:
                    prefix = title.split(":")[0]
                    patterns["title_patterns"][prefix] += 1
                elif "-" in title[:20]:
                    prefix = title.split("-")[0].strip()
                    patterns["title_patterns"][prefix] += 1
            
            # Analyze creation patterns
            created = page_content.get("created_time")
            if created:
                patterns["creation_times"].append(created)
            
            # Analyze common words (simple word frequency)
            words = content.lower().split()
            for word in words[:100]:  # Limit to first 100 words
                if len(word) > 4:  # Skip short words
                    patterns["common_words"][word] += 1
            
            # Analyze property fill rates
            properties = page_content.get("properties", {})
            for prop_name, prop_value in properties.items():
                is_filled = self._is_property_filled(prop_value)
                patterns["property_fill_rates"][prop_name].append(is_filled)
        
        # Calculate statistics
        if patterns["content_lengths"]:
            patterns["avg_content_length"] = statistics.mean(patterns["content_lengths"])
            patterns["median_content_length"] = statistics.median(patterns["content_lengths"])
        
        # Get most common patterns
        patterns["top_title_patterns"] = patterns["title_patterns"].most_common(5)
        patterns["top_words"] = patterns["common_words"].most_common(20)
        
        # Calculate property fill rates
        property_stats = {}
        for prop_name, fill_list in patterns["property_fill_rates"].items():
            if fill_list:
                fill_rate = sum(fill_list) / len(fill_list)
                property_stats[prop_name] = {
                    "fill_rate": fill_rate,
                    "always_filled": fill_rate == 1.0,
                    "never_filled": fill_rate == 0.0
                }
        patterns["property_statistics"] = property_stats
        
        self.patterns["content_patterns"] = patterns
        logger.info("Content pattern analysis complete")
    
    def _is_property_filled(self, prop_value: Dict) -> bool:
        """Check if a property has a value"""
        
        prop_type = prop_value.get("type")
        
        if prop_type == "title":
            return bool(prop_value.get("title"))
        elif prop_type == "rich_text":
            return bool(prop_value.get("rich_text"))
        elif prop_type == "number":
            return prop_value.get("number") is not None
        elif prop_type == "select":
            return prop_value.get("select") is not None
        elif prop_type == "multi_select":
            return bool(prop_value.get("multi_select"))
        elif prop_type == "date":
            return prop_value.get("date") is not None
        elif prop_type == "checkbox":
            return True  # Checkbox always has a value (True/False)
        elif prop_type == "url":
            return bool(prop_value.get("url"))
        elif prop_type == "email":
            return bool(prop_value.get("email"))
        elif prop_type == "phone_number":
            return bool(prop_value.get("phone_number"))
        elif prop_type == "relation":
            return bool(prop_value.get("relation"))
        
        return False
    
    async def _analyze_relationships(self):
        """Analyze relationships between databases and pages"""
        
        if not self.notion.workspace_structure:
            return
        
        relationships = self.notion.workspace_structure.get("relationships", [])
        
        analysis = {
            "total_relationships": len(relationships),
            "relationship_graph": defaultdict(list),
            "most_connected_databases": Counter(),
            "orphan_databases": []
        }
        
        # Build relationship graph
        connected_databases = set()
        for rel in relationships:
            from_db = rel["from_database_title"]
            to_db = rel["to_database_title"]
            
            analysis["relationship_graph"][from_db].append(to_db)
            analysis["most_connected_databases"][from_db] += 1
            analysis["most_connected_databases"][to_db] += 1
            
            connected_databases.add(rel["from_database"])
            connected_databases.add(rel["to_database"])
        
        # Find orphan databases
        all_databases = set(self.notion.workspace_structure.get("databases", {}).keys())
        orphans = all_databases - connected_databases
        
        for db_id in orphans:
            db_data = self.notion.workspace_structure["databases"].get(db_id, {})
            analysis["orphan_databases"].append(db_data.get("title", "Untitled"))
        
        self.patterns["relationship_analysis"] = analysis
        logger.info(f"Analyzed {len(relationships)} relationships")
    
    async def _calculate_health_metrics(self):
        """Calculate workspace health metrics"""
        
        metrics = {
            "organization_score": 0,
            "efficiency_score": 0,
            "completeness_score": 0,
            "relationship_score": 0,
            "overall_health": 0
        }
        
        # Organization score
        org_factors = {
            "has_multiple_databases": len(self.notion.workspace_structure.get("databases", {})) > 1,
            "uses_properties": self._check_property_usage(),
            "has_relationships": self.patterns.get("relationship_analysis", {}).get("total_relationships", 0) > 0,
            "consistent_naming": self._check_naming_consistency(),
            "no_empty_databases": len(self.patterns.get("database_analysis", {}).get("empty_databases", [])) == 0
        }
        
        org_weights = HEALTH_METRICS_CONFIG["organization_score"]["weights"]
        metrics["organization_score"] = self._calculate_weighted_score(org_factors, org_weights)
        
        # Efficiency score (based on content patterns)
        content_patterns = self.patterns.get("content_patterns", {})
        avg_length = content_patterns.get("avg_content_length", 0)
        
        efficiency_factors = {
            "appropriate_content_length": 100 < avg_length < 2000,
            "uses_title_patterns": len(content_patterns.get("top_title_patterns", [])) > 0,
            "properties_utilized": self._check_property_utilization()
        }
        
        metrics["efficiency_score"] = sum(efficiency_factors.values()) / len(efficiency_factors) * 100
        
        # Completeness score
        completeness_factors = {
            "has_inbox": self._has_inbox_database(),
            "has_projects": self._has_projects_database(),
            "has_tasks": self._has_tasks_database(),
            "has_archive": self._has_archive_database()
        }
        
        metrics["completeness_score"] = sum(completeness_factors.values()) / len(completeness_factors) * 100
        
        # Relationship score
        rel_analysis = self.patterns.get("relationship_analysis", {})
        total_dbs = len(self.notion.workspace_structure.get("databases", {}))
        orphan_count = len(rel_analysis.get("orphan_databases", []))
        
        if total_dbs > 0:
            metrics["relationship_score"] = ((total_dbs - orphan_count) / total_dbs) * 100
        
        # Overall health
        metrics["overall_health"] = statistics.mean([
            metrics["organization_score"],
            metrics["efficiency_score"],
            metrics["completeness_score"],
            metrics["relationship_score"]
        ])
        
        self.metrics = metrics
        logger.info(f"Calculated health score: {metrics['overall_health']:.1f}/100")
    
    def _calculate_weighted_score(self, factors: Dict[str, bool], weights: Dict[str, float]) -> float:
        """Calculate weighted score from factors"""
        
        total_score = 0
        total_weight = 0
        
        for factor, value in factors.items():
            weight = weights.get(factor, 0.2)  # Default weight
            if value:
                total_score += weight
            total_weight += weight
        
        if total_weight > 0:
            return (total_score / total_weight) * 100
        return 0
    
    def _check_property_usage(self) -> bool:
        """Check if properties are being used"""
        property_usage = self.patterns.get("database_analysis", {}).get("property_usage", {})
        return len(property_usage) > 3  # Using more than 3 property types
    
    def _check_naming_consistency(self) -> bool:
        """Check if there's naming consistency"""
        title_patterns = self.patterns.get("content_patterns", {}).get("top_title_patterns", [])
        return len(title_patterns) > 0  # Has some naming patterns
    
    def _check_property_utilization(self) -> bool:
        """Check if properties are being filled"""
        property_stats = self.patterns.get("content_patterns", {}).get("property_statistics", {})
        if property_stats:
            fill_rates = [stats["fill_rate"] for stats in property_stats.values()]
            return statistics.mean(fill_rates) > 0.5 if fill_rates else False
        return False
    
    def _has_inbox_database(self) -> bool:
        """Check if workspace has an inbox database"""
        db_types = self.patterns.get("database_analysis", {}).get("database_types", {})
        return db_types.get("inbox", 0) > 0
    
    def _has_projects_database(self) -> bool:
        """Check if workspace has a projects database"""
        db_types = self.patterns.get("database_analysis", {}).get("database_types", {})
        return db_types.get("projects", 0) > 0
    
    def _has_tasks_database(self) -> bool:
        """Check if workspace has a tasks database"""
        db_types = self.patterns.get("database_analysis", {}).get("database_types", {})
        return db_types.get("tasks", 0) > 0
    
    def _has_archive_database(self) -> bool:
        """Check if workspace has an archive database"""
        db_types = self.patterns.get("database_analysis", {}).get("database_types", {})
        # Check for archive in database titles
        databases = self.notion.workspace_structure.get("databases", {})
        for db_data in databases.values():
            if "archive" in db_data.get("title", "").lower():
                return True
        return False
    
    async def _generate_insights(self):
        """Generate actionable insights and recommendations"""
        
        recommendations = []
        
        # Check for missing essential databases
        if not self._has_tasks_database():
            recommendations.append({
                "type": "database_creation",
                "priority": "high",
                "title": "Create a Tasks database",
                "description": "A dedicated Tasks database helps track actionable items",
                "impact": "high"
            })
        
        if not self._has_archive_database():
            recommendations.append({
                "type": "database_creation",
                "priority": "medium",
                "title": "Create an Archive database",
                "description": "An Archive database keeps completed items accessible but out of the way",
                "impact": "medium"
            })
        
        # Check for empty databases
        empty_dbs = self.patterns.get("database_analysis", {}).get("empty_databases", [])
        for db_name in empty_dbs[:3]:  # Limit to 3
            recommendations.append({
                "type": "database_cleanup",
                "priority": "low",
                "title": f"Remove or populate '{db_name}' database",
                "description": "Empty databases add clutter without value",
                "impact": "low"
            })
        
        # Check for orphan databases
        orphan_dbs = self.patterns.get("relationship_analysis", {}).get("orphan_databases", [])
        if orphan_dbs:
            recommendations.append({
                "type": "relationship_creation",
                "priority": "medium",
                "title": "Connect isolated databases",
                "description": f"Databases like {', '.join(orphan_dbs[:3])} have no relationships",
                "impact": "medium"
            })
        
        # Check property utilization
        if not self._check_property_utilization():
            recommendations.append({
                "type": "property_improvement",
                "priority": "medium",
                "title": "Improve property utilization",
                "description": "Many properties are not being filled consistently",
                "impact": "medium"
            })
        
        # Check for large databases
        large_dbs = self.patterns.get("database_analysis", {}).get("large_databases", [])
        for db_info in large_dbs[:2]:  # Limit to 2
            recommendations.append({
                "type": "database_optimization",
                "priority": "medium",
                "title": f"Optimize '{db_info['title']}' database",
                "description": f"This database has {db_info['count']} items and may benefit from archiving or splitting",
                "impact": "high"
            })
        
        self.recommendations = recommendations
        logger.info(f"Generated {len(recommendations)} recommendations")
    
    def _calculate_overall_health_score(self) -> float:
        """Calculate overall workspace health score"""
        return self.metrics.get("overall_health", 0)
    
    def _save_analysis_results(self):
        """Save analysis results to file"""
        
        file_path = self.settings.data_dir / "workspace_analysis.json"
        
        try:
            with open(file_path, "w") as f:
                json.dump(self.workspace_data, f, indent=2, default=str)
            logger.info(f"Analysis results saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
    
    def _display_analysis_summary(self):
        """Display analysis summary in console"""
        
        # Create summary table
        table = Table(title="Workspace Analysis Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim", width=30)
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")
        
        # Add metrics
        metrics_display = [
            ("Total Databases", 
             len(self.notion.workspace_structure.get("databases", {})),
             "✅" if len(self.notion.workspace_structure.get("databases", {})) > 3 else "⚠️"),
            ("Total Pages (estimated)",
             self.notion.workspace_structure.get("total_pages", "Unknown"),
             "ℹ️"),
            ("Organization Score",
             f"{self.metrics.get('organization_score', 0):.1f}/100",
             "✅" if self.metrics.get('organization_score', 0) > 70 else "⚠️"),
            ("Efficiency Score",
             f"{self.metrics.get('efficiency_score', 0):.1f}/100",
             "✅" if self.metrics.get('efficiency_score', 0) > 70 else "⚠️"),
            ("Completeness Score",
             f"{self.metrics.get('completeness_score', 0):.1f}/100",
             "✅" if self.metrics.get('completeness_score', 0) > 70 else "⚠️"),
            ("Relationship Score",
             f"{self.metrics.get('relationship_score', 0):.1f}/100",
             "✅" if self.metrics.get('relationship_score', 0) > 70 else "⚠️"),
        ]
        
        for metric, value, status in metrics_display:
            table.add_row(metric, str(value), status)
        
        # Add overall health with color
        health_score = self.metrics.get("overall_health", 0)
        health_color = "green" if health_score > 80 else "yellow" if health_score > 60 else "red"
        table.add_row(
            "[bold]Overall Health Score[/bold]",
            f"[{health_color}]{health_score:.1f}/100[/{health_color}]",
            "🏆" if health_score > 80 else "📈" if health_score > 60 else "⚠️"
        )
        
        console.print("\n")
        console.print(table)
        
        # Display top recommendations
        if self.recommendations:
            console.print("\n[bold cyan]📋 Top Recommendations:[/bold cyan]\n")
            
            for i, rec in enumerate(self.recommendations[:5], 1):
                priority_color = {
                    "high": "red",
                    "medium": "yellow",
                    "low": "green"
                }.get(rec.get("priority", "medium"), "white")
                
                console.print(
                    f"{i}. [{priority_color}][{rec.get('priority', 'medium').upper()}][/{priority_color}] "
                    f"[bold]{rec.get('title', 'Untitled')}[/bold]"
                )
                console.print(f"   {rec.get('description', '')}\n")


if __name__ == "__main__":
    # Test the analyzer
    import asyncio
    
    async def test():
        notion_client = NotionAdvancedClient()
        analyzer = WorkspaceAnalyzer(notion_client)
        
        results = await analyzer.analyze_workspace(deep_scan=False)
        
        console.print("\n[bold green]Analysis Complete![/bold green]")
    
    asyncio.run(test())