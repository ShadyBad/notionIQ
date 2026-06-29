"""
API Optimization Module for NotionIQ
Minimizes API token usage and costs through intelligent optimization
"""

import hashlib
import json
import re
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from logger_wrapper import logger

# Try to import optional dependencies
try:
    import tiktoken
except ImportError:
    tiktoken = None
    logger.warning("tiktoken not available, using approximate token counting")


class OptimizationLevel(Enum):
    """API optimization levels"""

    MINIMAL = "minimal"  # Maximum token reduction
    BALANCED = "balanced"  # Balance between accuracy and cost
    FULL = "full"  # Full analysis (original)


@dataclass
class APIUsageMetrics:
    """Track API usage metrics"""

    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    duplicates_skipped: int = 0
    similar_pages_skipped: int = 0
    requests_by_hour: Dict[int, int] = field(default_factory=dict)
    tokens_saved: int = 0
    cost_saved: float = 0.0


class TokenOptimizer:
    """Optimizes content to minimize token usage"""

    def __init__(
        self,
        optimization_level: OptimizationLevel = OptimizationLevel.MINIMAL,
        input_cost_per_1m: float = 1.00,
        output_cost_per_1m: float = 5.00,
    ):
        """Initialize token optimizer with the selected model's pricing"""
        self.optimization_level = optimization_level
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self.encoder = None
        if tiktoken:
            try:
                # Use cl100k_base encoding (similar to Claude's tokenizer)
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                logger.warning("Failed to initialize tiktoken encoder")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4

    def optimize_content(self, content: str, max_length: int = 500) -> str:
        """
        Optimize content to minimize tokens while preserving meaning

        Args:
            content: Original content
            max_length: Maximum content length (chars)

        Returns:
            Optimized content
        """
        if not content:
            return ""

        if self.optimization_level == OptimizationLevel.FULL:
            return content[:10000]  # Original behavior

        # Remove excessive whitespace
        content = re.sub(r"\s+", " ", content)

        # Remove markdown formatting for minimal mode
        if self.optimization_level == OptimizationLevel.MINIMAL:
            # Remove code blocks (keep first line as indicator)
            content = re.sub(r"```[\s\S]*?```", "[code block]", content)
            # Remove URLs
            content = re.sub(r"https?://[^\s]+", "[url]", content)
            # Remove images
            content = re.sub(r"!\[.*?\]\(.*?\)", "[image]", content)
            # Remove excessive punctuation
            content = re.sub(r"[.]{3,}", "...", content)
            content = re.sub(r"[-=]{3,}", "---", content)

        # Smart truncation - try to end at sentence boundary
        if len(content) > max_length:
            truncated = content[:max_length]

            # Try to find last sentence end
            last_period = truncated.rfind(".")
            last_newline = truncated.rfind("\n")

            cut_point = max(last_period, last_newline)
            if cut_point > max_length * 0.7:  # Only if we don't lose too much
                content = truncated[: cut_point + 1]
            else:
                content = truncated + "..."

        return content.strip()

    def optimize_prompt(self, prompt: str) -> str:
        """
        Optimize prompt to use fewer tokens

        Args:
            prompt: Original prompt

        Returns:
            Optimized prompt
        """
        if self.optimization_level == OptimizationLevel.FULL:
            return prompt

        # Shorter, more direct prompt for minimal mode
        if self.optimization_level == OptimizationLevel.MINIMAL:
            # Extract just the essential parts
            lines = prompt.split("\n")

            # Keep only essential instructions
            essential_lines = []
            skip_sections = ["## Workspace Context", "## Analysis Requirements"]
            current_section = None

            for line in lines:
                if line.startswith("##"):
                    current_section = line
                    if current_section not in skip_sections:
                        essential_lines.append(line)
                elif current_section not in skip_sections:
                    # Keep page info and content
                    if "Title:" in line or "Content:" in line or "Properties:" in line:
                        essential_lines.append(line)
                    elif "```json" in line:
                        # Start of JSON template
                        break

            # Add minimal JSON instruction
            essential_lines.append("\nRespond with JSON only:")
            essential_lines.append(
                '{"classification":{"primary_type":"<type>","confidence":0.0},"recommendations":{"primary_action":"<action>","suggested_database":"<db>"}}'
            )

            return "\n".join(essential_lines)

        return prompt

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0
    ) -> float:
        """Calculate API cost in USD using the selected model's rates"""
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_1m
        cache_cost = (cache_read_tokens / 1_000_000) * self.input_cost_per_1m * 0.1
        return input_cost + output_cost + cache_cost


class RequestBatcher:
    """Batch and deduplicate API requests"""

    def __init__(self, batch_size: int = 10, wait_time: float = 0.5):
        """Initialize request batcher"""
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.pending_requests = []
        self.seen_hashes = set()
        self.batch_cache = {}  # Cache for batch processing results

    def add_request(self, content: Dict[str, Any]) -> bool:
        """
        Add request to batch, return False if duplicate

        Args:
            content: Content to analyze

        Returns:
            True if added, False if duplicate
        """
        # Generate content hash (handle sets and other non-serializable objects)
        try:
            content_hash = hashlib.md5(
                json.dumps(content, sort_keys=True, default=str).encode()
            ).hexdigest()
        except Exception as e:
            # Fallback: use string representation
            logger.debug(f"Could not serialize content for hash: {e}")
            content_hash = hashlib.md5(str(content).encode()).hexdigest()

        if content_hash in self.seen_hashes:
            logger.debug(f"Skipping duplicate content: {content_hash}")
            return False

        self.seen_hashes.add(content_hash)
        self.pending_requests.append(content)
        return True

    def should_process_batch(self) -> bool:
        """Check if batch should be processed"""
        return len(self.pending_requests) >= self.batch_size

    def get_batch(self) -> List[Dict[str, Any]]:
        """Get current batch and reset"""
        batch = self.pending_requests[: self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size :]
        return batch

    def process_batch_with_cache(
        self, batch: List[Dict[str, Any]], processor_func
    ) -> List[Dict[str, Any]]:
        """Process batch with caching to avoid redundant API calls"""
        results = []
        uncached = []

        for item in batch:
            try:
                item_hash = hashlib.md5(
                    json.dumps(item, sort_keys=True, default=str).encode()
                ).hexdigest()
            except Exception as e:
                # Fallback: use string representation
                logger.debug(f"Could not serialize item for hash: {e}")
                item_hash = hashlib.md5(str(item).encode()).hexdigest()

            if item_hash in self.batch_cache:
                results.append(self.batch_cache[item_hash])
            else:
                uncached.append((item, item_hash))

        # Process uncached items
        if uncached:
            for item, item_hash in uncached:
                result = processor_func(item)
                self.batch_cache[item_hash] = result
                results.append(result)

        return results

    def get_remaining(self) -> List[Dict[str, Any]]:
        """Get remaining requests"""
        remaining = self.pending_requests
        self.pending_requests = []
        return remaining


class APIOptimizer:
    """Main API optimization coordinator"""

    def __init__(
        self,
        settings,
        optimization_level: OptimizationLevel = OptimizationLevel.MINIMAL,
        input_cost_per_1m: float = 1.00,
        output_cost_per_1m: float = 5.00,
    ):
        """Initialize API optimizer with the selected model's pricing"""
        self.settings = settings
        self.optimization_level = optimization_level

        # Initialize components
        self.token_optimizer = TokenOptimizer(
            optimization_level,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
        )
        self.request_batcher = RequestBatcher()
        self.metrics = APIUsageMetrics()

        # Load previous metrics
        self._load_metrics()

    def optimize_page_content(self, page_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize page content for minimal API usage

        Args:
            page_content: Original page content

        Returns:
            Optimized content
        """
        optimized = page_content.copy()

        # Optimize text content
        if "content" in optimized:
            original_length = len(optimized["content"])
            optimized["content"] = self.token_optimizer.optimize_content(
                optimized["content"],
                max_length=(
                    300
                    if self.optimization_level == OptimizationLevel.MINIMAL
                    else 1000
                ),
            )

            tokens_saved = self.token_optimizer.count_tokens(
                page_content["content"]
            ) - self.token_optimizer.count_tokens(optimized["content"])

            self.metrics.tokens_saved += tokens_saved

            logger.debug(
                f"Content optimized: {original_length} -> {len(optimized['content'])} chars, "
                f"~{tokens_saved} tokens saved"
            )

        # Minimize properties for minimal mode
        if self.optimization_level == OptimizationLevel.MINIMAL:
            if "properties" in optimized:
                # Keep only essential properties
                essential_props = ["Status", "Priority", "Due Date", "Tags"]
                optimized["properties"] = {
                    k: v
                    for k, v in optimized["properties"].items()
                    if k in essential_props
                }

        # Remove unnecessary fields
        for field in ["blocks", "url", "parent"]:
            optimized.pop(field, None)

        return optimized

    def should_skip_page(self, page_content: Dict[str, Any]) -> bool:
        """
        Determine if page should be skipped

        Args:
            page_content: Page to check

        Returns:
            True if page should be skipped
        """
        # Skip truly empty pages (no title and no content)
        title = page_content.get("title", "").strip()
        content = page_content.get("content", "").strip()
        if not title and not content:
            logger.debug("Skipping completely empty page")
            return True

        # Skip archived pages
        if page_content.get("archived"):
            logger.debug("Skipping archived page")
            return True

        # Skip if both title and content are very short (likely not meaningful)
        if len(title + content) < 10:
            logger.debug("Skipping page with minimal content")
            return True

        # For minimal mode, skip pages that look like ACTUAL empty templates
        if self.optimization_level == OptimizationLevel.MINIMAL:
            content_lower = page_content.get("content", "").lower()
            title_lower = page_content.get("title", "").lower()

            # Only skip if it's obviously an empty template (very restrictive)
            empty_template_indicators = [
                "[placeholder]",
                "{{",
                "lorem ipsum",
                "sample text",
                "untitled page",
                "new page",
            ]

            # Skip only if title AND content suggest it's empty template
            title_is_template = any(
                indicator in title_lower
                for indicator in ["untitled", "new page", "template page"]
            )
            content_is_template = any(
                indicator in content_lower for indicator in empty_template_indicators
            )

            if title_is_template and (content_is_template or len(content_lower) < 50):
                logger.debug("Skipping genuinely empty template page")
                self.metrics.similar_pages_skipped += 1
                return True

        return False

    def record_api_usage(
        self, input_tokens: int, output_tokens: int, from_cache: bool = False
    ):
        """Record API usage metrics"""
        if not from_cache:
            self.metrics.total_requests += 1
            self.metrics.total_tokens += input_tokens + output_tokens

            cost = self.token_optimizer.calculate_cost(input_tokens, output_tokens)
            self.metrics.total_cost += cost

            # Track by hour
            current_hour = datetime.now().hour
            self.metrics.requests_by_hour[current_hour] = (
                self.metrics.requests_by_hour.get(current_hour, 0) + 1
            )
        else:
            # Calculate saved cost
            estimated_cost = self.token_optimizer.calculate_cost(
                input_tokens, 200  # Estimated output tokens
            )
            self.metrics.cost_saved += estimated_cost

        self._save_metrics()

    def get_usage_report(self) -> Dict[str, Any]:
        """Get usage report"""
        cache_rate = (
            (
                self.metrics.cache_hits
                / (self.metrics.cache_hits + self.metrics.cache_misses)
            )
            if (self.metrics.cache_hits + self.metrics.cache_misses) > 0
            else 0
        )

        return {
            "total_requests": self.metrics.total_requests,
            "total_tokens": self.metrics.total_tokens,
            "total_cost": f"${self.metrics.total_cost:.2f}",
            "cost_saved": f"${self.metrics.cost_saved:.2f}",
            "tokens_saved": self.metrics.tokens_saved,
            "cache_hit_rate": f"{cache_rate:.1%}",
            "duplicates_skipped": self.metrics.duplicates_skipped,
            "similar_pages_skipped": self.metrics.similar_pages_skipped,
            "optimization_level": self.optimization_level.value,
            "average_tokens_per_request": (
                (self.metrics.total_tokens / self.metrics.total_requests)
                if self.metrics.total_requests > 0
                else 0
            ),
        }

    def _load_metrics(self):
        """Load saved metrics"""
        metrics_file = self.settings.data_dir / "api_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                    self.metrics.total_requests = data.get("total_requests", 0)
                    self.metrics.total_tokens = data.get("total_tokens", 0)
                    self.metrics.total_cost = data.get("total_cost", 0.0)
                    self.metrics.tokens_saved = data.get("tokens_saved", 0)
                    self.metrics.cost_saved = data.get("cost_saved", 0.0)
            except:
                pass

    def _save_metrics(self):
        """Save metrics to file"""
        metrics_file = self.settings.data_dir / "api_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(
                {
                    "total_requests": self.metrics.total_requests,
                    "total_tokens": self.metrics.total_tokens,
                    "total_cost": self.metrics.total_cost,
                    "tokens_saved": self.metrics.tokens_saved,
                    "cost_saved": self.metrics.cost_saved,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )


def compress_data(data: str) -> bytes:
    """Compress data for storage/transmission"""
    return zlib.compress(data.encode(), level=9)


def decompress_data(data: bytes) -> str:
    """Decompress data"""
    return zlib.decompress(data).decode()
