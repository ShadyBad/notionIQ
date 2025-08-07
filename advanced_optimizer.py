"""
Advanced API Optimization Module
Implements aggressive cost-saving strategies for API usage
"""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logger_wrapper import logger


class ContentPriority(Enum):
    """Priority levels for content analysis"""

    CRITICAL = "critical"  # Must analyze (active tasks, urgent items)
    HIGH = "high"  # Should analyze (recent content)
    MEDIUM = "medium"  # Can analyze if budget allows
    LOW = "low"  # Skip unless specifically requested
    SKIP = "skip"  # Never analyze


@dataclass
class OptimizationConfig:
    """Configuration for advanced optimization"""

    max_daily_cost: float = 10.0  # Maximum daily API cost in USD
    max_tokens_per_page: int = 200  # Max tokens per page analysis
    similarity_threshold: float = 0.90  # Threshold for content similarity
    batch_size: int = 20  # Larger batches for better dedup
    enable_progressive: bool = True  # Enable progressive analysis
    enable_sampling: bool = True  # Sample large workspaces
    sampling_rate: float = 0.8  # Sample 80% of low-priority content
    cache_ttl_days: int = 30  # Cache results for 30 days


class ContentPrioritizer:
    """Prioritize content for analysis based on various signals"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.priority_keywords = {
            ContentPriority.CRITICAL: [
                "urgent",
                "asap",
                "critical",
                "blocker",
                "emergency",
                "deadline",
                "overdue",
                "today",
                "now",
            ],
            ContentPriority.HIGH: [
                "important",
                "priority",
                "meeting",
                "todo",
                "task",
                "action",
                "review",
                "feedback",
                "this week",
            ],
            ContentPriority.MEDIUM: [
                "project",
                "plan",
                "idea",
                "note",
                "document",
                "draft",
                "proposal",
                "research",
                "wiki",
                "dashboard",
                "hub",
                "system",
                "analysis",
                "management",
            ],
            ContentPriority.LOW: [
                "archive",
                "old",
                "backup",
                "copy",
                "untitled",
                "empty",
                "test page",
                "sample page",
                "template page",
            ],
        }

    def calculate_priority(self, page_content: Dict[str, Any]) -> ContentPriority:
        """Calculate priority for a page"""

        # Check if archived or deleted
        if page_content.get("archived") or page_content.get("in_trash"):
            return ContentPriority.SKIP

        # Check last edited time
        last_edited = page_content.get("last_edited_time", "")
        if last_edited:
            try:
                edited_date = datetime.fromisoformat(last_edited.replace("Z", "+00:00"))
                days_old = (datetime.now(edited_date.tzinfo) - edited_date).days

                if days_old > 365:  # Over a year old
                    return ContentPriority.LOW
                elif days_old > 90:  # Over 3 months
                    return ContentPriority.MEDIUM
                elif days_old > 30:  # Over a month
                    return ContentPriority.HIGH
                else:  # Recent
                    return ContentPriority.CRITICAL
            except:
                pass

        # Check content for keywords
        content = (
            page_content.get("title", "") + " " + page_content.get("content", "")
        ).lower()

        # Check for priority keywords
        for priority, keywords in self.priority_keywords.items():
            if any(keyword in content for keyword in keywords):
                return priority

        # Check content length
        content_length = len(page_content.get("content", ""))
        if content_length < 50:
            return ContentPriority.LOW
        elif content_length < 500:
            return ContentPriority.MEDIUM
        else:
            return ContentPriority.HIGH

    def should_analyze(
        self, priority: ContentPriority, sampling_enabled: bool = True
    ) -> bool:
        """Determine if content should be analyzed based on priority"""

        if priority == ContentPriority.SKIP:
            return False
        elif priority == ContentPriority.CRITICAL:
            return True
        elif priority == ContentPriority.HIGH:
            return True
        elif priority == ContentPriority.MEDIUM:
            # Sample medium priority content
            if sampling_enabled and self.config.enable_sampling:
                import random

                return random.random() < self.config.sampling_rate * 1.5
            return True
        else:  # LOW
            if sampling_enabled and self.config.enable_sampling:
                import random

                return random.random() < self.config.sampling_rate
            return False


class SemanticDeduplicator:
    """Advanced deduplication using semantic similarity"""

    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
        self.processed_signatures = {}
        self.similar_groups = defaultdict(list)

    def _create_semantic_signature(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Create semantic signature for content"""

        text = (content.get("title", "") + " " + content.get("content", "")).lower()

        # Extract features
        signature = {
            "word_count": len(text.split()),
            "char_count": len(text),
            "has_numbers": bool(re.search(r"\d+", text)),
            "has_dates": bool(
                re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}", text)
            ),
            "has_urls": bool(re.search(r"https?://", text)),
            "has_emails": bool(
                re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
            ),
            "line_count": text.count("\n"),
            "question_count": text.count("?"),
            "exclamation_count": text.count("!"),
            "bullet_points": text.count("•") + text.count("-") + text.count("*"),
        }

        # Extract key terms (simple approach)
        words = text.split()
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] += 1

        # Top 10 most frequent words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        signature["top_words"] = [w[0] for w in top_words]

        # Create hash for exact matching
        signature["hash"] = hashlib.md5(text.encode()).hexdigest()

        return signature

    def _calculate_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate similarity between two signatures"""

        # Exact match
        if sig1["hash"] == sig2["hash"]:
            return 1.0

        # Calculate feature similarity
        features_score = 0
        feature_weights = {
            "word_count": 0.1,
            "char_count": 0.1,
            "has_numbers": 0.05,
            "has_dates": 0.05,
            "has_urls": 0.05,
            "has_emails": 0.05,
            "line_count": 0.05,
            "question_count": 0.05,
            "exclamation_count": 0.05,
            "bullet_points": 0.05,
        }

        for feature, weight in feature_weights.items():
            if sig1[feature] == sig2[feature]:
                features_score += weight
            elif isinstance(sig1[feature], (int, float)):
                # For numeric features, calculate relative difference
                max_val = max(sig1[feature], sig2[feature])
                if max_val > 0:
                    diff = abs(sig1[feature] - sig2[feature]) / max_val
                    features_score += weight * (1 - diff)

        # Calculate word overlap
        words1 = set(sig1.get("top_words", []))
        words2 = set(sig2.get("top_words", []))

        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            features_score += word_overlap * 0.4

        return min(features_score, 1.0)

    def is_duplicate(self, content: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if content is duplicate or highly similar to already processed"""

        signature = self._create_semantic_signature(content)
        content_hash = signature["hash"]

        # Check for exact duplicate
        if content_hash in self.processed_signatures:
            return True, content_hash

        # Check for semantic similarity
        for existing_hash, existing_sig in self.processed_signatures.items():
            similarity = self._calculate_similarity(signature, existing_sig)

            if similarity >= self.similarity_threshold:
                logger.debug(f"Found similar content: {similarity:.2%} similarity")
                self.similar_groups[existing_hash].append(content_hash)
                return True, existing_hash

        # Not a duplicate, store signature
        self.processed_signatures[content_hash] = signature
        return False, None

    def get_representative(self, group_hash: str) -> Dict[str, Any]:
        """Get representative content for a group of similar items"""
        return self.processed_signatures.get(group_hash, {})


class ProgressiveAnalyzer:
    """Implement progressive analysis with early stopping"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.analysis_history = []
        self.confidence_threshold = 0.85
        self.pattern_cache = {}

    def should_continue_analysis(self, pages_analyzed: int, total_pages: int) -> bool:
        """Determine if analysis should continue based on patterns"""

        if pages_analyzed < 10:  # Always analyze first 10
            return True

        if not self.config.enable_progressive:
            return True

        # Calculate confidence in patterns
        if self.analysis_history:
            recent_analyses = self.analysis_history[-20:]

            # Check for repetitive patterns
            classifications = [
                a.get("classification", {}).get("primary_type") for a in recent_analyses
            ]

            if classifications:
                from collections import Counter

                type_counts = Counter(classifications)
                most_common = type_counts.most_common(1)[0]

                # If 80% of recent pages are same type, we have a pattern
                if most_common[1] / len(classifications) > 0.8:
                    logger.info(
                        f"Pattern detected: {most_common[0]} dominates recent analyses"
                    )

                    # Sample remaining pages instead of full analysis
                    return pages_analyzed < (total_pages * 0.2)  # Analyze only 20%

        return True

    def record_analysis(self, analysis_result: Dict[str, Any]):
        """Record analysis for pattern detection"""
        self.analysis_history.append(analysis_result)

        # Update pattern cache
        classification = analysis_result.get("classification", {}).get("primary_type")
        if classification:
            if classification not in self.pattern_cache:
                self.pattern_cache[classification] = []
            self.pattern_cache[classification].append(analysis_result)

    def predict_classification(
        self, content: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Predict classification based on learned patterns"""

        if len(self.analysis_history) < 50:  # Need enough data
            return None

        # Simple pattern matching based on title/content features
        title = content.get("title", "").lower()
        content_text = content.get("content", "").lower()

        # Check against known patterns
        for classification, examples in self.pattern_cache.items():
            if len(examples) >= 5:  # Need enough examples
                # Check for common words in titles
                title_words = set(title.split())
                example_title_words = set()
                for ex in examples[-10:]:  # Last 10 examples
                    ex_title = ex.get("page_title", "").lower()
                    example_title_words.update(ex_title.split())

                overlap = len(title_words.intersection(example_title_words))
                if overlap >= 2:  # At least 2 common words
                    return {
                        "classification": {
                            "primary_type": classification,
                            "confidence": 0.7,  # Lower confidence for predicted
                            "reasoning": "Predicted based on pattern matching",
                        },
                        "predicted": True,
                    }

        return None


class CostTracker:
    """Track and enforce cost limits"""

    def __init__(self, max_daily_cost: float = 10.0):
        self.max_daily_cost = max_daily_cost
        self.daily_costs = defaultdict(float)
        self.hourly_costs = defaultdict(float)

    def can_make_request(self, estimated_cost: float) -> bool:
        """Check if request can be made within budget"""
        today = datetime.now().date()
        current_hour = datetime.now().strftime("%Y-%m-%d %H")

        # Check daily limit
        if self.daily_costs[today] + estimated_cost > self.max_daily_cost:
            logger.warning(f"Daily cost limit reached: ${self.daily_costs[today]:.2f}")
            return False

        # Check hourly rate limit (max 25% of daily in one hour)
        hourly_limit = self.max_daily_cost * 0.25
        if self.hourly_costs[current_hour] + estimated_cost > hourly_limit:
            logger.warning(
                f"Hourly rate limit reached: ${self.hourly_costs[current_hour]:.2f}"
            )
            return False

        return True

    def record_cost(self, actual_cost: float):
        """Record actual cost"""
        today = datetime.now().date()
        current_hour = datetime.now().strftime("%Y-%m-%d %H")

        self.daily_costs[today] += actual_cost
        self.hourly_costs[current_hour] += actual_cost

        logger.info(
            f"Cost recorded: ${actual_cost:.4f} (Daily total: ${self.daily_costs[today]:.2f})"
        )

    def get_remaining_budget(self) -> float:
        """Get remaining budget for today"""
        today = datetime.now().date()
        return max(0, self.max_daily_cost - self.daily_costs[today])


class AdvancedOptimizer:
    """Main advanced optimization coordinator"""

    def __init__(self, settings, config: Optional[OptimizationConfig] = None):
        self.settings = settings
        self.config = config or OptimizationConfig()

        # Initialize components
        self.prioritizer = ContentPrioritizer(self.config)
        self.deduplicator = SemanticDeduplicator(self.config.similarity_threshold)
        self.progressive = ProgressiveAnalyzer(self.config)
        self.cost_tracker = CostTracker(self.config.max_daily_cost)

        logger.info(
            f"Advanced Optimizer initialized with daily budget: ${self.config.max_daily_cost}"
        )

    def optimize_batch(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize a batch of pages for analysis

        Returns: Filtered and prioritized list of pages to analyze
        """

        optimized = []
        skipped_stats = defaultdict(int)

        # First pass: prioritize and deduplicate
        prioritized = []
        for page in pages:
            # Calculate priority
            priority = self.prioritizer.calculate_priority(page)

            # Check if should analyze
            if not self.prioritizer.should_analyze(priority):
                skipped_stats["low_priority"] += 1
                continue

            # Check for duplicates
            is_dup, dup_hash = self.deduplicator.is_duplicate(page)
            if is_dup:
                skipped_stats["duplicate"] += 1
                continue

            # Check if can predict classification
            if priority == ContentPriority.LOW:
                prediction = self.progressive.predict_classification(page)
                if prediction:
                    skipped_stats["predicted"] += 1
                    # Store prediction instead of making API call
                    page["predicted_analysis"] = prediction
                    continue

            prioritized.append((priority, page))

        # Sort by priority
        prioritized.sort(
            key=lambda x: [
                ContentPriority.CRITICAL,
                ContentPriority.HIGH,
                ContentPriority.MEDIUM,
                ContentPriority.LOW,
            ].index(x[0])
        )

        # Apply budget constraints
        remaining_budget = self.cost_tracker.get_remaining_budget()
        estimated_cost_per_page = 0.02  # Rough estimate

        max_pages = int(remaining_budget / estimated_cost_per_page)
        if max_pages < len(prioritized):
            logger.warning(
                f"Budget allows only {max_pages} pages, have {len(prioritized)}"
            )
            prioritized = prioritized[:max_pages]

        # Extract pages
        optimized = [page for _, page in prioritized]

        # Log optimization stats
        logger.info(
            f"Optimization complete: {len(optimized)}/{len(pages)} pages selected"
        )
        logger.info(f"Skipped: {dict(skipped_stats)}")

        return optimized

    def create_minimal_prompt(self, content: Dict[str, Any]) -> str:
        """Create ultra-minimal prompt for API"""

        # Extract only essential info
        title = content.get("title", "Untitled")[:50]
        text = content.get("content", "")[:200]

        # Ultra-compact prompt
        prompt = f"""Classify this Notion page:
Title: {title}
Content: {text}

Return JSON only:
{{"type":"<task|project|note|idea|meeting|reference>","action":"<move|archive|keep|review>","confidence":0.0}}"""

        return prompt

    def should_stop_analysis(self, pages_analyzed: int, total_pages: int) -> bool:
        """Check if should stop analysis early"""
        return not self.progressive.should_continue_analysis(
            pages_analyzed, total_pages
        )
