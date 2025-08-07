"""
API Optimization Module for NotionIQ
Minimizes API token usage and costs through intelligent optimization
"""

import hashlib
import json
import pickle
import re
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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

    # Claude 3 Opus pricing (per 1M tokens)
    INPUT_COST_PER_1M = 15.00  # $15 per 1M input tokens
    OUTPUT_COST_PER_1M = 75.00  # $75 per 1M output tokens

    def __init__(
        self, optimization_level: OptimizationLevel = OptimizationLevel.MINIMAL
    ):
        """Initialize token optimizer"""
        self.optimization_level = optimization_level
        self.encoder = None
        if tiktoken:
            try:
                # Use cl100k_base encoding (similar to Claude's tokenizer)
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except:
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

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost in USD"""
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        return input_cost + output_cost


class SmartCache:
    """Advanced caching with fingerprinting and similarity detection"""

    def __init__(self, cache_dir: Path, ttl_hours: int = 168):  # 1 week default
        """Initialize smart cache"""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.fingerprint_cache: Dict[str, Any] = {}
        self.similarity_threshold = 0.85

        # Load existing cache
        self._load_cache()

    def _generate_fingerprint(self, content: Dict[str, Any]) -> str:
        """Generate content fingerprint for similarity detection"""
        # Extract key features
        features = {
            "title_words": set(content.get("title", "").lower().split()),
            "content_length": len(content.get("content", ""))
            // 100,  # Bucket by 100 chars
            "properties": sorted(content.get("properties", {}).keys()),
            "has_dates": bool(re.search(r"\d{4}-\d{2}-\d{2}", str(content))),
            "has_todos": "todo" in str(content).lower()
            or "task" in str(content).lower(),
            "has_meeting": "meeting" in str(content).lower(),
            "word_count": len(str(content).split()) // 10,  # Bucket by 10 words
        }

        # Create fingerprint (handle sets and other non-serializable objects)
        try:
            fingerprint = hashlib.md5(
                json.dumps(features, sort_keys=True, default=str).encode()
            ).hexdigest()
        except Exception as e:
            # Fallback: use string representation
            logger.debug(f"Could not serialize features for fingerprint: {e}")
            fingerprint = hashlib.md5(str(features).encode()).hexdigest()

        return fingerprint

    def _calculate_similarity(self, content1: Dict, content2: Dict) -> float:
        """Calculate similarity between two pieces of content"""
        # Simple Jaccard similarity on words
        words1 = set(str(content1).lower().split())
        words2 = set(str(content2).lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def get_similar_cached(self, content: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find similar content in cache"""
        fingerprint = self._generate_fingerprint(content)

        # Check exact fingerprint match
        if fingerprint in self.fingerprint_cache:
            cached_file = self.fingerprint_cache[fingerprint]
            if cached_file.exists():
                try:
                    with open(cached_file, "rb") as f:
                        cached_data = pickle.load(f)

                    # Check TTL
                    if datetime.now() - cached_data["timestamp"] < self.ttl:
                        logger.debug(f"Found exact cache match: {fingerprint}")
                        return cached_data["result"]
                except:
                    pass

        # Check for similar content
        for fp, cached_file in self.fingerprint_cache.items():
            if cached_file.exists():
                try:
                    with open(cached_file, "rb") as f:
                        cached_data = pickle.load(f)

                    # Check TTL
                    if datetime.now() - cached_data["timestamp"] < self.ttl:
                        similarity = self._calculate_similarity(
                            content, cached_data.get("original_content", {})
                        )

                        if similarity > self.similarity_threshold:
                            logger.debug(f"Found similar cache match: {similarity:.2%}")
                            # Return cached result with adjusted confidence
                            result = cached_data["result"].copy()
                            if "classification" in result:
                                result["classification"]["confidence"] *= similarity
                            return result
                except:
                    pass

        return None

    def store(self, content: Dict[str, Any], result: Dict[str, Any]):
        """Store result in cache with fingerprint"""
        fingerprint = self._generate_fingerprint(content)
        cache_file = self.cache_dir / f"{fingerprint}.pkl"

        cache_data = {
            "timestamp": datetime.now(),
            "fingerprint": fingerprint,
            "original_content": content,
            "result": result,
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        self.fingerprint_cache[fingerprint] = cache_file
        self._save_cache()

    def _load_cache(self):
        """Load cache index"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    data = json.load(f)
                    self.fingerprint_cache = {k: Path(v) for k, v in data.items()}
            except:
                self.fingerprint_cache = {}

    def _save_cache(self):
        """Save cache index"""
        index_file = self.cache_dir / "cache_index.json"
        with open(index_file, "w") as f:
            json.dump({k: str(v) for k, v in self.fingerprint_cache.items()}, f)

    def clean_expired(self):
        """Clean expired cache entries"""
        expired = []
        for fingerprint, cache_file in self.fingerprint_cache.items():
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)

                    if datetime.now() - cached_data["timestamp"] > self.ttl:
                        cache_file.unlink()
                        expired.append(fingerprint)
                except:
                    expired.append(fingerprint)

        for fp in expired:
            del self.fingerprint_cache[fp]

        if expired:
            self._save_cache()
            logger.info(f"Cleaned {len(expired)} expired cache entries")


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
    ):
        """Initialize API optimizer"""
        self.settings = settings
        self.optimization_level = optimization_level

        # Initialize components
        self.token_optimizer = TokenOptimizer(optimization_level)
        self.smart_cache = SmartCache(
            settings.data_dir / "smart_cache",
            ttl_hours=settings.cache_ttl_hours * 7,  # Longer TTL for smart cache
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

    def check_cache_and_similarity(
        self, content: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check cache and find similar analyzed content"""
        # Check smart cache
        cached_result = self.smart_cache.get_similar_cached(content)
        if cached_result:
            self.metrics.cache_hits += 1
            logger.info("Using cached/similar result - no API call needed")
            return cached_result

        self.metrics.cache_misses += 1
        return None

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
