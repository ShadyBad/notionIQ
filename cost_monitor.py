"""
Real-time Cost Monitoring and Analytics
Tracks API costs and provides detailed usage insights
"""

import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from logger_wrapper import logger

console = Console()


@dataclass
class TokenUsage:
    """Track token usage for a single API call"""

    timestamp: datetime
    input_tokens: int
    output_tokens: int
    cost: float
    model: str
    cached: bool = False
    page_id: Optional[str] = None
    page_title: Optional[str] = None


@dataclass
class CostMetrics:
    """Aggregated cost metrics"""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cost_saved: float = 0.0
    average_input_tokens: float = 0.0
    average_output_tokens: float = 0.0
    average_cost_per_call: float = 0.0
    peak_hour: Optional[int] = None
    peak_hour_cost: float = 0.0


class CostMonitor:
    """Monitor and track API costs in real-time"""

    # Pricing per 1M tokens (Claude 3 Opus)
    CLAUDE_INPUT_COST = 15.00  # $15 per 1M input tokens
    CLAUDE_OUTPUT_COST = 75.00  # $75 per 1M output tokens

    # OpenAI GPT-4 pricing
    GPT4_INPUT_COST = 30.00  # $30 per 1M input tokens
    GPT4_OUTPUT_COST = 60.00  # $60 per 1M output tokens

    def __init__(self, data_dir: Path, budget_limits: Optional[Dict] = None):
        """Initialize cost monitor"""
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Budget limits
        self.budget_limits = budget_limits or {
            "hourly": 2.0,  # $2 per hour
            "daily": 10.0,  # $10 per day
            "weekly": 50.0,  # $50 per week
            "monthly": 150.0,  # $150 per month
        }

        # Usage tracking
        self.usage_history: List[TokenUsage] = []
        self.hourly_costs = defaultdict(float)
        self.daily_costs = defaultdict(float)
        self.model_costs = defaultdict(float)

        # Load historical data
        self._load_history()

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model: str = "claude-3-opus"
    ) -> float:
        """Calculate cost for token usage"""

        if "claude" in model.lower():
            input_cost = (input_tokens / 1_000_000) * self.CLAUDE_INPUT_COST
            output_cost = (output_tokens / 1_000_000) * self.CLAUDE_OUTPUT_COST
        elif "gpt-4" in model.lower():
            input_cost = (input_tokens / 1_000_000) * self.GPT4_INPUT_COST
            output_cost = (output_tokens / 1_000_000) * self.GPT4_OUTPUT_COST
        else:
            # Default to Claude pricing
            input_cost = (input_tokens / 1_000_000) * self.CLAUDE_INPUT_COST
            output_cost = (output_tokens / 1_000_000) * self.CLAUDE_OUTPUT_COST

        return input_cost + output_cost

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "claude-3-opus",
        cached: bool = False,
        page_id: Optional[str] = None,
        page_title: Optional[str] = None,
    ) -> TokenUsage:
        """Record API token usage"""

        cost = (
            self.calculate_cost(input_tokens, output_tokens, model)
            if not cached
            else 0.0
        )

        usage = TokenUsage(
            timestamp=datetime.now(),
            input_tokens=input_tokens if not cached else 0,
            output_tokens=output_tokens if not cached else 0,
            cost=cost,
            model=model,
            cached=cached,
            page_id=page_id,
            page_title=page_title,
        )

        self.usage_history.append(usage)

        # Update aggregates
        hour_key = usage.timestamp.strftime("%Y-%m-%d %H")
        day_key = usage.timestamp.strftime("%Y-%m-%d")

        self.hourly_costs[hour_key] += cost
        self.daily_costs[day_key] += cost
        self.model_costs[model] += cost

        # Check budget alerts
        self._check_budget_alerts()

        # Save periodically
        if len(self.usage_history) % 10 == 0:
            self._save_history()

        return usage

    def get_current_metrics(self) -> CostMetrics:
        """Get current aggregated metrics"""

        metrics = CostMetrics()

        if not self.usage_history:
            return metrics

        # Calculate totals
        for usage in self.usage_history:
            if not usage.cached:
                metrics.total_calls += 1
                metrics.total_input_tokens += usage.input_tokens
                metrics.total_output_tokens += usage.output_tokens
                metrics.total_cost += usage.cost
            else:
                metrics.cache_hits += 1

        metrics.cache_misses = metrics.total_calls

        # Calculate averages
        if metrics.total_calls > 0:
            metrics.average_input_tokens = (
                metrics.total_input_tokens / metrics.total_calls
            )
            metrics.average_output_tokens = (
                metrics.total_output_tokens / metrics.total_calls
            )
            metrics.average_cost_per_call = metrics.total_cost / metrics.total_calls

        # Find peak hour
        if self.hourly_costs:
            peak = max(self.hourly_costs.items(), key=lambda x: x[1])
            metrics.peak_hour = int(peak[0].split()[-1].split(":")[0])
            metrics.peak_hour_cost = peak[1]

        # Calculate saved costs
        if metrics.cache_hits > 0:
            avg_cost = (
                metrics.average_cost_per_call
                if metrics.average_cost_per_call > 0
                else 0.02
            )
            metrics.cost_saved = metrics.cache_hits * avg_cost

        return metrics

    def get_budget_status(self) -> Dict[str, Dict]:
        """Get current budget status"""

        now = datetime.now()
        status = {}

        # Hourly budget
        current_hour = now.strftime("%Y-%m-%d %H")
        hourly_spent = self.hourly_costs.get(current_hour, 0.0)
        status["hourly"] = {
            "spent": hourly_spent,
            "limit": self.budget_limits["hourly"],
            "remaining": max(0, self.budget_limits["hourly"] - hourly_spent),
            "percentage": (
                (hourly_spent / self.budget_limits["hourly"] * 100)
                if self.budget_limits["hourly"] > 0
                else 0
            ),
        }

        # Daily budget
        today = now.strftime("%Y-%m-%d")
        daily_spent = self.daily_costs.get(today, 0.0)
        status["daily"] = {
            "spent": daily_spent,
            "limit": self.budget_limits["daily"],
            "remaining": max(0, self.budget_limits["daily"] - daily_spent),
            "percentage": (
                (daily_spent / self.budget_limits["daily"] * 100)
                if self.budget_limits["daily"] > 0
                else 0
            ),
        }

        # Weekly budget
        week_start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
        weekly_spent = sum(
            cost for date, cost in self.daily_costs.items() if date >= week_start
        )
        status["weekly"] = {
            "spent": weekly_spent,
            "limit": self.budget_limits["weekly"],
            "remaining": max(0, self.budget_limits["weekly"] - weekly_spent),
            "percentage": (
                (weekly_spent / self.budget_limits["weekly"] * 100)
                if self.budget_limits["weekly"] > 0
                else 0
            ),
        }

        # Monthly budget
        month_start = now.strftime("%Y-%m-01")
        monthly_spent = sum(
            cost for date, cost in self.daily_costs.items() if date >= month_start
        )
        status["monthly"] = {
            "spent": monthly_spent,
            "limit": self.budget_limits["monthly"],
            "remaining": max(0, self.budget_limits["monthly"] - monthly_spent),
            "percentage": (
                (monthly_spent / self.budget_limits["monthly"] * 100)
                if self.budget_limits["monthly"] > 0
                else 0
            ),
        }

        return status

    def display_dashboard(self):
        """Display cost monitoring dashboard"""

        console.clear()

        # Get current metrics
        metrics = self.get_current_metrics()
        budget_status = self.get_budget_status()

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        # Header
        layout["header"].update(
            Panel(
                f"[bold cyan]NotionIQ Cost Monitor[/bold cyan] | "
                f"Total Cost: [bold green]${metrics.total_cost:.2f}[/bold green] | "
                f"Calls: {metrics.total_calls} | "
                f"Cache Hits: {metrics.cache_hits}",
                style="bold",
            )
        )

        # Main content - split into two columns
        layout["main"].split_row(Layout(name="metrics"), Layout(name="budget"))

        # Metrics table
        metrics_table = Table(title="Usage Metrics", show_header=True)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        metrics_table.add_row("Total API Calls", str(metrics.total_calls))
        metrics_table.add_row("Total Input Tokens", f"{metrics.total_input_tokens:,}")
        metrics_table.add_row("Total Output Tokens", f"{metrics.total_output_tokens:,}")
        metrics_table.add_row("Avg Input Tokens", f"{metrics.average_input_tokens:.0f}")
        metrics_table.add_row(
            "Avg Output Tokens", f"{metrics.average_output_tokens:.0f}"
        )
        metrics_table.add_row(
            "Avg Cost per Call", f"${metrics.average_cost_per_call:.4f}"
        )
        metrics_table.add_row(
            "Cache Hit Rate",
            f"{(metrics.cache_hits / (metrics.total_calls + metrics.cache_hits) * 100) if metrics.total_calls > 0 else 0:.1f}%",
        )
        metrics_table.add_row("Cost Saved", f"${metrics.cost_saved:.2f}")

        layout["metrics"].update(metrics_table)

        # Budget table
        budget_table = Table(title="Budget Status", show_header=True)
        budget_table.add_column("Period", style="cyan")
        budget_table.add_column("Spent", style="yellow")
        budget_table.add_column("Limit", style="green")
        budget_table.add_column("Remaining", style="blue")
        budget_table.add_column("Usage", style="magenta")

        for period, status in budget_status.items():
            color = (
                "red"
                if status["percentage"] > 80
                else "yellow" if status["percentage"] > 50 else "green"
            )
            budget_table.add_row(
                period.capitalize(),
                f"${status['spent']:.2f}",
                f"${status['limit']:.2f}",
                f"${status['remaining']:.2f}",
                f"[{color}]{status['percentage']:.1f}%[/{color}]",
            )

        layout["budget"].update(budget_table)

        # Footer
        layout["footer"].update(
            Panel(
                f"Peak Hour: {metrics.peak_hour or 'N/A'} | "
                f"Peak Cost: ${metrics.peak_hour_cost:.2f} | "
                f"Models Used: {', '.join(self.model_costs.keys())}",
                style="dim",
            )
        )

        console.print(layout)

    def generate_report(self) -> Dict:
        """Generate detailed cost report"""

        metrics = self.get_current_metrics()
        budget_status = self.get_budget_status()

        # Analyze patterns
        hourly_pattern = self._analyze_hourly_pattern()
        daily_pattern = self._analyze_daily_pattern()

        report = {
            "generated_at": datetime.now().isoformat(),
            "metrics": asdict(metrics),
            "budget_status": budget_status,
            "patterns": {"hourly": hourly_pattern, "daily": daily_pattern},
            "model_breakdown": dict(self.model_costs),
            "recommendations": self._generate_recommendations(metrics, budget_status),
        }

        # Save report
        report_file = (
            self.data_dir
            / f"cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def _analyze_hourly_pattern(self) -> Dict:
        """Analyze hourly usage patterns"""

        hourly_totals = defaultdict(list)

        for hour_key, cost in self.hourly_costs.items():
            hour = int(hour_key.split()[-1].split(":")[0])
            hourly_totals[hour].append(cost)

        pattern = {}
        for hour in range(24):
            costs = hourly_totals.get(hour, [])
            if costs:
                pattern[hour] = {
                    "average": statistics.mean(costs),
                    "max": max(costs),
                    "min": min(costs),
                    "total": sum(costs),
                }

        return pattern

    def _analyze_daily_pattern(self) -> Dict:
        """Analyze daily usage patterns"""

        daily_pattern = {}

        for day, cost in sorted(self.daily_costs.items()):
            daily_pattern[day] = {
                "cost": cost,
                "day_of_week": datetime.strptime(day, "%Y-%m-%d").strftime("%A"),
            }

        return daily_pattern

    def _generate_recommendations(
        self, metrics: CostMetrics, budget_status: Dict
    ) -> List[str]:
        """Generate cost optimization recommendations"""

        recommendations = []

        # Check if approaching budget limits
        for period, status in budget_status.items():
            if status["percentage"] > 80:
                recommendations.append(
                    f"⚠️ {period.capitalize()} budget usage at {status['percentage']:.1f}% - consider reducing API calls"
                )

        # Check cache efficiency
        if metrics.total_calls > 0:
            cache_rate = metrics.cache_hits / (metrics.total_calls + metrics.cache_hits)
            if cache_rate < 0.3:
                recommendations.append(
                    f"💡 Low cache hit rate ({cache_rate:.1%}) - consider enabling more aggressive caching"
                )

        # Check token usage
        if metrics.average_input_tokens > 1000:
            recommendations.append(
                f"📝 High average input tokens ({metrics.average_input_tokens:.0f}) - consider content truncation"
            )

        if metrics.average_output_tokens > 500:
            recommendations.append(
                f"📤 High average output tokens ({metrics.average_output_tokens:.0f}) - request shorter responses"
            )

        # Cost per call
        if metrics.average_cost_per_call > 0.05:
            recommendations.append(
                f"💰 High cost per call (${metrics.average_cost_per_call:.3f}) - optimize prompts and responses"
            )

        return recommendations

    def _check_budget_alerts(self):
        """Check and alert on budget thresholds"""

        status = self.get_budget_status()

        for period, info in status.items():
            if info["percentage"] > 90:
                logger.warning(
                    f"Budget Alert: {period} spending at {info['percentage']:.1f}% of limit!"
                )
            elif info["percentage"] > 75:
                logger.info(
                    f"Budget Warning: {period} spending at {info['percentage']:.1f}% of limit"
                )

    def _load_history(self):
        """Load historical usage data"""

        history_file = self.data_dir / "cost_history.json"
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)

                    # Reconstruct usage history
                    for item in data.get("usage_history", []):
                        usage = TokenUsage(
                            timestamp=datetime.fromisoformat(item["timestamp"]),
                            input_tokens=item["input_tokens"],
                            output_tokens=item["output_tokens"],
                            cost=item["cost"],
                            model=item["model"],
                            cached=item.get("cached", False),
                            page_id=item.get("page_id"),
                            page_title=item.get("page_title"),
                        )
                        self.usage_history.append(usage)

                    # Reconstruct aggregates
                    self.hourly_costs = defaultdict(float, data.get("hourly_costs", {}))
                    self.daily_costs = defaultdict(float, data.get("daily_costs", {}))
                    self.model_costs = defaultdict(float, data.get("model_costs", {}))

                    logger.info(
                        f"Loaded {len(self.usage_history)} historical usage records"
                    )
            except Exception as e:
                logger.error(f"Failed to load cost history: {e}")

    def _save_history(self):
        """Save usage history to file"""

        history_file = self.data_dir / "cost_history.json"

        # Convert to serializable format
        data = {
            "usage_history": [
                {
                    "timestamp": usage.timestamp.isoformat(),
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cost": usage.cost,
                    "model": usage.model,
                    "cached": usage.cached,
                    "page_id": usage.page_id,
                    "page_title": usage.page_title,
                }
                for usage in self.usage_history[-1000:]  # Keep last 1000 records
            ],
            "hourly_costs": dict(self.hourly_costs),
            "daily_costs": dict(self.daily_costs),
            "model_costs": dict(self.model_costs),
        }

        try:
            with open(history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cost history: {e}")


if __name__ == "__main__":
    # Test the cost monitor
    monitor = CostMonitor(Path("data"))

    # Simulate some usage
    monitor.record_usage(500, 200, "claude-3-opus")
    monitor.record_usage(300, 150, "claude-3-opus", cached=True)
    monitor.record_usage(800, 400, "gpt-4")

    # Display dashboard
    monitor.display_dashboard()

    # Generate report
    report = monitor.generate_report()
    console.print(Panel(json.dumps(report, indent=2, default=str), title="Cost Report"))
