"""
Automatic API Configuration Selector
Intelligently selects the best API configuration based on availability and limits
"""

import json
import os
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from api_optimizer import OptimizationLevel
from logger_wrapper import logger


class APIProvider(Enum):
    """Available API providers"""

    CLAUDE_OPUS = "claude-3-opus"
    CLAUDE_SONNET = "claude-3-sonnet"
    CLAUDE_HAIKU = "claude-3-haiku"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_3_5 = "gpt-3.5-turbo"


class RateLimitTier(Enum):
    """API rate limit tiers"""

    FREE = "free"  # Very limited
    BASIC = "basic"  # Standard limits
    PREMIUM = "premium"  # Higher limits
    ENTERPRISE = "enterprise"  # Highest limits


class APIAutoConfigurator:
    """Automatically configures API settings for optimal performance and cost"""

    # Model cost per 1M tokens (input/output)
    MODEL_COSTS = {
        APIProvider.CLAUDE_OPUS: (15.00, 75.00),  # Most expensive, best quality
        APIProvider.CLAUDE_SONNET: (3.00, 15.00),  # Balanced
        APIProvider.CLAUDE_HAIKU: (0.25, 1.25),  # Cheapest Claude
        APIProvider.GPT_4_TURBO: (10.00, 30.00),  # Good alternative
        APIProvider.GPT_4: (30.00, 60.00),  # Expensive
        APIProvider.GPT_3_5: (0.50, 1.50),  # Cheapest overall
    }

    # Rate limits by tier (requests per minute)
    RATE_LIMITS = {
        RateLimitTier.FREE: 5,
        RateLimitTier.BASIC: 50,
        RateLimitTier.PREMIUM: 500,
        RateLimitTier.ENTERPRISE: 5000,
    }

    def __init__(self, settings_file: Optional[Path] = None):
        """Initialize auto configurator"""
        self.settings_file = settings_file or Path("data/api_config.json")
        self.usage_history = self._load_usage_history()

    def auto_configure(self) -> Dict[str, Any]:
        """
        Automatically determine the best API configuration

        Returns:
            Configuration dictionary with optimal settings
        """
        logger.info("🔧 Auto-configuring API settings...")

        # Step 1: Detect available APIs
        available_apis = self._detect_available_apis()

        if not available_apis:
            raise ValueError(
                "No API keys found! Please set ANTHROPIC_API_KEY or OPENAI_API_KEY"
            )

        # Step 2: Determine rate limit tier
        rate_tier = self._detect_rate_limit_tier(available_apis)

        # Step 3: Analyze usage patterns
        usage_pattern = self._analyze_usage_patterns()

        # Step 4: Select optimal model
        selected_model, provider = self._select_optimal_model(
            available_apis, rate_tier, usage_pattern
        )

        # Step 5: Configure optimization level
        optimization_level = self._determine_optimization_level(
            selected_model, rate_tier, usage_pattern
        )

        # Step 6: Set rate limiting
        rate_limit = self._calculate_rate_limit(rate_tier, optimization_level)

        # Step 7: Configure caching and batching
        cache_config = self._optimize_cache_settings(usage_pattern)

        # Build configuration
        config = {
            "provider": provider,
            "model": selected_model.value,
            "optimization_level": optimization_level.value,
            "rate_limit_rps": rate_limit,
            "batch_size": self._calculate_batch_size(rate_tier, optimization_level),
            "enable_caching": True,
            "cache_ttl_hours": cache_config["ttl"],
            "smart_cache_enabled": cache_config["smart_cache"],
            "deduplication_enabled": True,
            "skip_templates": optimization_level == OptimizationLevel.MINIMAL,
            "estimated_cost_per_100": self._estimate_cost(
                selected_model, optimization_level
            ),
            "confidence_threshold": self._get_confidence_threshold(optimization_level),
            "auto_configured": True,
            "configuration_reason": self._explain_configuration(
                selected_model, rate_tier, optimization_level
            ),
        }

        # Save configuration
        self._save_configuration(config)

        # Display configuration
        self._display_configuration(config)

        return config

    def _detect_available_apis(self) -> Dict[str, APIProvider]:
        """Detect which API keys are available"""
        available = {}

        # Check Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            # Try to detect which Claude models are available
            # Default to Opus if we have a key
            available["anthropic"] = APIProvider.CLAUDE_OPUS
            logger.info("✅ Anthropic API key detected")

            # Check for model-specific environment hints
            if os.getenv("CLAUDE_MODEL", "").lower() == "sonnet":
                available["anthropic"] = APIProvider.CLAUDE_SONNET
            elif os.getenv("CLAUDE_MODEL", "").lower() == "haiku":
                available["anthropic"] = APIProvider.CLAUDE_HAIKU

        # Check OpenAI
        if os.getenv("OPENAI_API_KEY"):
            # Default to GPT-4 Turbo
            available["openai"] = APIProvider.GPT_4_TURBO
            logger.info("✅ OpenAI API key detected")

            # Check for model hints
            if os.getenv("OPENAI_MODEL", "").lower() == "gpt-4":
                available["openai"] = APIProvider.GPT_4
            elif "3.5" in os.getenv("OPENAI_MODEL", "").lower():
                available["openai"] = APIProvider.GPT_3_5

        return available

    def _detect_rate_limit_tier(self, available_apis: Dict) -> RateLimitTier:
        """Detect rate limit tier based on API keys and usage"""

        # Check for tier hints in environment
        tier_hint = os.getenv("API_TIER", "").lower()
        if tier_hint:
            if "enterprise" in tier_hint:
                return RateLimitTier.ENTERPRISE
            elif "premium" in tier_hint:
                return RateLimitTier.PREMIUM
            elif "free" in tier_hint:
                return RateLimitTier.FREE

        # Check usage history for rate limit errors
        if self.usage_history.get("rate_limit_errors", 0) > 10:
            logger.info("⚠️ Detected rate limit issues - assuming FREE tier")
            return RateLimitTier.FREE

        # Default to BASIC for most users
        logger.info("📊 Assuming BASIC tier rate limits")
        return RateLimitTier.BASIC

    def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyze historical usage patterns"""
        patterns = {
            "avg_pages_per_run": self.usage_history.get("avg_pages_per_run", 50),
            "frequency": self.usage_history.get("run_frequency", "weekly"),
            "peak_hours": self.usage_history.get("peak_hours", []),
            "total_pages_processed": self.usage_history.get("total_pages", 0),
            "cache_hit_rate": self.usage_history.get("cache_hit_rate", 0.0),
        }

        # Determine usage intensity
        if patterns["avg_pages_per_run"] > 100:
            patterns["intensity"] = "high"
        elif patterns["avg_pages_per_run"] > 30:
            patterns["intensity"] = "medium"
        else:
            patterns["intensity"] = "low"

        return patterns

    def _select_optimal_model(
        self, available_apis: Dict, rate_tier: RateLimitTier, usage_pattern: Dict
    ) -> Tuple[APIProvider, str]:
        """Select the optimal model based on availability and constraints"""

        # Priority order based on cost-effectiveness
        if rate_tier == RateLimitTier.FREE:
            # Use cheapest available
            preference_order = [
                APIProvider.GPT_3_5,
                APIProvider.CLAUDE_HAIKU,
                APIProvider.CLAUDE_SONNET,
                APIProvider.GPT_4_TURBO,
                APIProvider.CLAUDE_OPUS,
                APIProvider.GPT_4,
            ]
        elif usage_pattern["intensity"] == "high":
            # Balance cost and quality for high volume
            preference_order = [
                APIProvider.CLAUDE_HAIKU,
                APIProvider.GPT_3_5,
                APIProvider.CLAUDE_SONNET,
                APIProvider.GPT_4_TURBO,
                APIProvider.CLAUDE_OPUS,
                APIProvider.GPT_4,
            ]
        else:
            # Quality first for low-medium volume
            preference_order = [
                APIProvider.CLAUDE_SONNET,  # Best balance
                APIProvider.CLAUDE_OPUS,  # Highest quality
                APIProvider.GPT_4_TURBO,  # Good alternative
                APIProvider.CLAUDE_HAIKU,  # Cheaper option
                APIProvider.GPT_3_5,  # Cheapest
                APIProvider.GPT_4,  # Most expensive
            ]

        # Select first available from preference order
        for model in preference_order:
            for provider, available_model in available_apis.items():
                if available_model == model:
                    logger.info(f"📌 Selected model: {model.value} ({provider})")
                    return model, provider

        # Fallback to any available
        provider = list(available_apis.keys())[0]
        model = available_apis[provider]
        logger.info(f"📌 Using available model: {model.value} ({provider})")
        return model, provider

    def _determine_optimization_level(
        self, model: APIProvider, rate_tier: RateLimitTier, usage_pattern: Dict
    ) -> OptimizationLevel:
        """Determine optimal optimization level"""

        # Get model cost
        input_cost, output_cost = self.MODEL_COSTS[model]
        total_cost_per_1m = input_cost + output_cost

        # Decision logic
        if rate_tier == RateLimitTier.FREE:
            # Always use minimal for free tier
            return OptimizationLevel.MINIMAL

        elif total_cost_per_1m > 50:  # Expensive models
            # Use minimal for expensive models
            return OptimizationLevel.MINIMAL

        elif total_cost_per_1m > 10:  # Medium cost models
            # Use balanced for medium cost
            if usage_pattern["intensity"] == "high":
                return OptimizationLevel.MINIMAL
            else:
                return OptimizationLevel.BALANCED

        else:  # Cheap models
            # Can afford less optimization with cheap models
            if usage_pattern["intensity"] == "high":
                return OptimizationLevel.BALANCED
            else:
                return OptimizationLevel.FULL

    def _calculate_rate_limit(
        self, rate_tier: RateLimitTier, optimization_level: OptimizationLevel
    ) -> float:
        """Calculate requests per second limit"""

        # Get base limit (requests per minute)
        base_rpm = self.RATE_LIMITS[rate_tier]

        # Convert to requests per second
        base_rps = base_rpm / 60.0

        # Apply safety margin
        if optimization_level == OptimizationLevel.MINIMAL:
            # Can be more aggressive with minimal tokens
            safety_margin = 0.8
        else:
            # Be more conservative with more tokens
            safety_margin = 0.5

        return round(base_rps * safety_margin, 1)

    def _calculate_batch_size(
        self, rate_tier: RateLimitTier, optimization_level: OptimizationLevel
    ) -> int:
        """Calculate optimal batch size"""

        if rate_tier == RateLimitTier.FREE:
            return 5  # Small batches for free tier
        elif rate_tier == RateLimitTier.BASIC:
            if optimization_level == OptimizationLevel.MINIMAL:
                return 20  # Can handle more with minimal tokens
            else:
                return 10
        else:  # Premium/Enterprise
            if optimization_level == OptimizationLevel.MINIMAL:
                return 50
            elif optimization_level == OptimizationLevel.BALANCED:
                return 25
            else:
                return 15

    def _optimize_cache_settings(self, usage_pattern: Dict) -> Dict[str, Any]:
        """Optimize cache configuration"""

        cache_config = {
            "smart_cache": True,  # Always use smart cache
            "ttl": 168,  # Default 1 week
        }

        # Adjust TTL based on usage frequency
        if usage_pattern["frequency"] == "daily":
            cache_config["ttl"] = 24  # 1 day
        elif usage_pattern["frequency"] == "weekly":
            cache_config["ttl"] = 168  # 1 week
        else:
            cache_config["ttl"] = 720  # 1 month

        # Enable aggressive caching for high volume
        if usage_pattern["intensity"] == "high":
            cache_config["similarity_threshold"] = 0.75  # More aggressive
        else:
            cache_config["similarity_threshold"] = 0.85  # Standard

        return cache_config

    def _estimate_cost(
        self, model: APIProvider, optimization_level: OptimizationLevel
    ) -> float:
        """Estimate cost per 100 pages"""

        input_cost, output_cost = self.MODEL_COSTS[model]

        # Estimate tokens based on optimization level
        if optimization_level == OptimizationLevel.MINIMAL:
            avg_input_tokens = 200
            avg_output_tokens = 100
        elif optimization_level == OptimizationLevel.BALANCED:
            avg_input_tokens = 800
            avg_output_tokens = 400
        else:
            avg_input_tokens = 2000
            avg_output_tokens = 1000

        # Calculate per page
        cost_per_page = (avg_input_tokens / 1_000_000) * input_cost + (
            avg_output_tokens / 1_000_000
        ) * output_cost

        # Per 100 pages
        return round(cost_per_page * 100, 2)

    def _get_confidence_threshold(self, optimization_level: OptimizationLevel) -> float:
        """Get confidence threshold for actions"""
        if optimization_level == OptimizationLevel.MINIMAL:
            # Higher threshold for minimal mode (be more conservative)
            return 0.8
        elif optimization_level == OptimizationLevel.BALANCED:
            return 0.7
        else:
            return 0.6

    def _explain_configuration(
        self,
        model: APIProvider,
        rate_tier: RateLimitTier,
        optimization_level: OptimizationLevel,
    ) -> str:
        """Explain why this configuration was chosen"""

        explanations = []

        # Model selection
        if model in [APIProvider.CLAUDE_HAIKU, APIProvider.GPT_3_5]:
            explanations.append(f"Using {model.value} for lowest cost")
        elif model in [APIProvider.CLAUDE_SONNET, APIProvider.GPT_4_TURBO]:
            explanations.append(f"Using {model.value} for best cost/quality balance")
        else:
            explanations.append(f"Using {model.value} for highest quality")

        # Rate tier
        if rate_tier == RateLimitTier.FREE:
            explanations.append("Configured for free tier limits")
        elif rate_tier == RateLimitTier.BASIC:
            explanations.append("Configured for standard API limits")
        else:
            explanations.append("Configured for premium API limits")

        # Optimization
        if optimization_level == OptimizationLevel.MINIMAL:
            explanations.append("Maximum token reduction for cost savings")
        elif optimization_level == OptimizationLevel.BALANCED:
            explanations.append("Balanced token usage and accuracy")
        else:
            explanations.append("Full analysis for maximum accuracy")

        return " | ".join(explanations)

    def _save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file"""
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)

        config["timestamp"] = datetime.now().isoformat()

        with open(self.settings_file, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"💾 Configuration saved to {self.settings_file}")

    def _display_configuration(self, config: Dict[str, Any]):
        """Display the selected configuration"""

        print("\n" + "=" * 60)
        print("🎯 AUTO-CONFIGURED API SETTINGS")
        print("=" * 60)
        print(f"Provider: {config['provider'].upper()}")
        print(f"Model: {config['model']}")
        print(f"Optimization: {config['optimization_level'].upper()}")
        print(f"Rate Limit: {config['rate_limit_rps']} req/sec")
        print(f"Batch Size: {config['batch_size']} pages")
        print(f"Est. Cost: ${config['estimated_cost_per_100']}/100 pages")
        print("-" * 60)
        print(f"Reason: {config['configuration_reason']}")
        print("=" * 60)

    def _load_usage_history(self) -> Dict[str, Any]:
        """Load historical usage data"""
        history_file = Path("data/usage_history.json")

        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    return json.load(f)
            except:
                pass

        return {}

    def update_usage_history(self, metrics: Dict[str, Any]):
        """Update usage history with new metrics"""

        # Update running averages
        if "total_pages" not in self.usage_history:
            self.usage_history["total_pages"] = 0

        self.usage_history["total_pages"] += metrics.get("pages_processed", 0)
        self.usage_history["last_run"] = datetime.now().isoformat()

        # Track rate limit errors
        if metrics.get("rate_limit_error"):
            self.usage_history["rate_limit_errors"] = (
                self.usage_history.get("rate_limit_errors", 0) + 1
            )

        # Save history
        history_file = Path("data/usage_history.json")
        history_file.parent.mkdir(parents=True, exist_ok=True)

        with open(history_file, "w") as f:
            json.dump(self.usage_history, f, indent=2)


def get_auto_config() -> Dict[str, Any]:
    """
    Get automatically configured API settings

    Returns:
        Optimal configuration dictionary
    """
    configurator = APIAutoConfigurator()
    return configurator.auto_configure()


if __name__ == "__main__":
    # Test auto configuration
    config = get_auto_config()
    print(f"\n✅ Auto-configuration complete!")
    print(f"   You can now run: python notion_organizer.py")
    print(f"   No additional configuration needed!")
