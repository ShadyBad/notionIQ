"""
Claude AI integration for intelligent content analysis
Handles all AI-powered classification and recommendation generation
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic
from rich.console import Console
from rich.panel import Panel
from tenacity import retry, stop_after_attempt, wait_exponential

from advanced_optimizer import AdvancedOptimizer, ContentPriority, OptimizationConfig
from api_optimizer import APIOptimizer, OptimizationLevel, TokenOptimizer
from config import CLASSIFICATION_CONFIG, Settings, get_settings
from logger_wrapper import logger

console = Console()


def _safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """Safely serialize object to JSON, converting sets and other non-serializable types"""

    def convert_sets(o):
        if isinstance(o, set):
            return list(o)
        elif isinstance(o, dict):
            return {k: convert_sets(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert_sets(item) for item in o]
        return o

    try:
        converted = convert_sets(obj)
        return json.dumps(converted, indent=indent, default=str)
    except Exception as e:
        return f"[Could not serialize properties: {e}]"


class ClaudeAnalyzer:
    """Claude-powered content analyzer and classifier"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.MINIMAL,
    ):
        """Initialize Claude analyzer with API optimization"""
        self.settings = settings or get_settings()
        self.optimization_level = optimization_level

        # Initialize API optimizer for minimal token usage
        self.api_optimizer = APIOptimizer(self.settings, optimization_level)

        # Initialize advanced optimizer for aggressive cost savings
        self.advanced_optimizer = AdvancedOptimizer(
            self.settings,
            OptimizationConfig(
                max_daily_cost=5.0,  # $5 daily limit
                max_tokens_per_page=150,
                similarity_threshold=0.85,
                batch_size=20,
                enable_progressive=True,
                enable_sampling=True,
                sampling_rate=0.2,  # Sample 20% of low priority
            ),
        )

        logger.info(f"API Optimization Level: {optimization_level.value}")

        # Check if we should auto-configure
        if hasattr(self.settings, "auto_configured") and self.settings.auto_configured:
            # Use auto-configured settings
            ai_config = self.settings.ai_config
        else:
            # Use manual configuration
            ai_config = self.settings.get_ai_config()

        if ai_config["provider"] == "anthropic":
            self.client = Anthropic(api_key=ai_config["api_key"])
            self.model = ai_config["model"]
            self.provider = "anthropic"
        else:
            # OpenAI implementation
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=ai_config["api_key"])
                self.model = ai_config["model"]
                self.provider = "openai"
                logger.info("Using OpenAI provider")
            except ImportError:
                logger.error(
                    "OpenAI library not installed. Please install with: pip install openai"
                )
                raise ImportError("OpenAI library required for OpenAI provider")

        # Cache for responses
        self.response_cache = {}
        self._load_cache()

        logger.info(f"ClaudeAnalyzer initialized with {self.provider}")

    def analyze_page(
        self,
        page_content: Dict[str, Any],
        workspace_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze a single page and return classification and recommendations"""

        # Check if page should be skipped (minimal mode optimization)
        if self.api_optimizer.should_skip_page(page_content):
            logger.info(
                f"Skipping page: {page_content.get('title', 'Untitled')} (optimization)"
            )
            return self._create_skipped_response(page_content)

        # Check smart cache for similar content
        cached_result = self.api_optimizer.check_cache_and_similarity(page_content)
        if cached_result:
            cached_result["page_id"] = page_content.get("id")
            cached_result["page_title"] = page_content.get("title", "Untitled")
            cached_result["from_cache"] = True
            return cached_result

        # Generate cache key for exact match
        cache_key = self._generate_cache_key(page_content)

        # Check traditional cache
        if self.settings.enable_caching and cache_key in self.response_cache:
            logger.debug(f"Using cached analysis for page {page_content.get('id')}")
            self.api_optimizer.metrics.cache_hits += 1
            return self.response_cache[cache_key]

        # Optimize page content for minimal tokens
        optimized_content = self.api_optimizer.optimize_page_content(page_content)

        # Prepare content for analysis
        prepared_content = self._prepare_content(optimized_content)

        # Generate optimized analysis prompt
        prompt = self._create_analysis_prompt(prepared_content, workspace_context)

        # Optimize prompt for minimal tokens
        if self.optimization_level == OptimizationLevel.MINIMAL:
            prompt = self.api_optimizer.token_optimizer.optimize_prompt(prompt)

        # Count input tokens for metrics
        input_tokens = self.api_optimizer.token_optimizer.count_tokens(prompt)

        # Get AI response
        response = self._get_ai_response(prompt)

        # Count output tokens and record usage
        output_tokens = self.api_optimizer.token_optimizer.count_tokens(response)
        self.api_optimizer.record_api_usage(
            input_tokens, output_tokens, from_cache=False
        )

        logger.info(
            f"API Call - Input: {input_tokens} tokens, Output: {output_tokens} tokens, Cost: ${self.api_optimizer.token_optimizer.calculate_cost(input_tokens, output_tokens):.4f}"
        )

        # Parse response
        analysis = self._parse_analysis_response(response)

        # Add metadata
        analysis["page_id"] = page_content.get("id")
        analysis["page_title"] = page_content.get("title", "Untitled")
        analysis["page_url"] = page_content.get("url", "")
        analysis["analysis_timestamp"] = datetime.now().isoformat()

        # Cache result in both caches
        if self.settings.enable_caching:
            self.response_cache[cache_key] = analysis
            self._save_cache()

            # Store in smart cache for similarity detection
            self.api_optimizer.smart_cache.store(page_content, analysis)

        return analysis

    def _prepare_content(self, page_content: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare page content for AI analysis"""

        # Extract key information
        title = page_content.get("title", "Untitled")
        content = page_content.get("content", "")
        properties = page_content.get("properties", {})
        created_time = page_content.get("created_time", "")
        last_edited_time = page_content.get("last_edited_time", "")

        # Truncate content based on optimization level
        if self.optimization_level == OptimizationLevel.MINIMAL:
            max_length = 500  # Much shorter for minimal mode
        elif self.optimization_level == OptimizationLevel.BALANCED:
            max_length = 2000
        else:
            max_length = self.settings.max_content_length

        if len(content) > max_length:
            content = content[:max_length] + "... [truncated]"

        # Extract property values (simplified for now)
        property_summary = {}
        for prop_name, prop_value in properties.items():
            prop_type = prop_value.get("type")

            if prop_type == "title":
                continue  # Already have title
            elif prop_type == "rich_text":
                text_array = prop_value.get("rich_text", [])
                if text_array:
                    property_summary[prop_name] = "".join(
                        t.get("plain_text", "") for t in text_array
                    )
            elif prop_type == "select":
                select_value = prop_value.get("select")
                if select_value:
                    property_summary[prop_name] = select_value.get("name", "")
            elif prop_type == "multi_select":
                multi_select = prop_value.get("multi_select", [])
                if multi_select:
                    property_summary[prop_name] = ", ".join(
                        s.get("name", "") for s in multi_select
                    )
            elif prop_type == "date":
                date_value = prop_value.get("date")
                if date_value:
                    property_summary[prop_name] = date_value.get("start", "")
            elif prop_type == "checkbox":
                property_summary[prop_name] = prop_value.get("checkbox", False)
            elif prop_type == "number":
                property_summary[prop_name] = prop_value.get("number")
            elif prop_type == "status":
                status_value = prop_value.get("status")
                if status_value:
                    property_summary[prop_name] = status_value.get("name", "")

        return {
            "title": title,
            "content": content,
            "properties": property_summary,
            "created_time": created_time,
            "last_edited_time": last_edited_time,
            "content_length": len(content),
            "has_properties": bool(property_summary),
        }

    def _create_analysis_prompt(
        self,
        prepared_content: Dict[str, Any],
        workspace_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a sophisticated prompt for Claude"""

        # Use ultra-minimal prompt for aggressive optimization
        if self.optimization_level == OptimizationLevel.MINIMAL:
            return self._create_minimal_prompt(prepared_content)

        # Build context about the workspace if available
        workspace_info = ""
        if workspace_context:
            databases = workspace_context.get("databases", {})
            if databases:
                db_list = [f"- {db['title']}" for db in databases.values()]
                workspace_info = f"""
## Workspace Context
This workspace contains the following databases:
{chr(10).join(db_list[:10])}

Total pages in workspace: {workspace_context.get('total_pages', 'Unknown')}
"""

        # Build the main prompt
        prompt = f"""You are an expert Notion workspace organizer and productivity consultant. Analyze the following Notion page and provide intelligent classification and recommendations.

{workspace_info}

## Page Information
**Title:** {prepared_content['title']}
**Created:** {prepared_content['created_time'][:10] if prepared_content['created_time'] else 'Unknown'}
**Last Edited:** {prepared_content['last_edited_time'][:10] if prepared_content['last_edited_time'] else 'Unknown'}
**Content Length:** {prepared_content['content_length']} characters

## Page Properties
{_safe_json_dumps(prepared_content['properties']) if prepared_content['properties'] else 'No properties set'}

## Page Content
{prepared_content['content'][:5000] if prepared_content['content'] else '[No content]'}

## Analysis Requirements

Please analyze this page and provide a comprehensive response in valid JSON format with the following structure:

```json
{{
  "classification": {{
    "primary_type": "<one of: task, project, meeting_note, idea, journal, reference, sop, goal, archive>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation of classification>",
    "secondary_types": ["<other applicable types>"]
  }},
  
  "content_analysis": {{
    "summary": "<50-100 word summary of the content>",
    "key_topics": ["<topic1>", "<topic2>", ...],
    "sentiment": "<positive, neutral, negative, mixed>",
    "completeness": "<complete, partial, stub, empty>",
    "information_density": "<high, medium, low>",
    "actionable_items": <count of action items found>
  }},
  
  "urgency_assessment": {{
    "level": "<immediate, this_week, this_month, someday, no_deadline>",
    "confidence": <0.0-1.0>,
    "detected_deadline": "<date if found, null otherwise>",
    "reasoning": "<brief explanation>"
  }},
  
  "context_detection": {{
    "primary_context": "<work, personal, learning, creative, administrative>",
    "confidence": <0.0-1.0>,
    "detected_project": "<project name if identified>",
    "detected_people": ["<person1>", "<person2>", ...],
    "detected_department": "<department if identified>"
  }},
  
  "recommendations": {{
    "primary_action": "<move_to_database, archive, delete, review, break_down, merge_with, no_action>",
    "confidence": <0.0-1.0>,
    "suggested_database": "<database name or null>",
    "reasoning": "<detailed explanation of recommendation>",
    "additional_actions": [
      {{
        "action": "<action type>",
        "description": "<what to do>",
        "priority": "<high, medium, low>"
      }}
    ]
  }},
  
  "organization_suggestions": {{
    "suggested_title": "<improved title if needed>",
    "suggested_tags": ["<tag1>", "<tag2>", ...],
    "suggested_properties": {{
      "<property_name>": "<suggested_value>"
    }},
    "suggested_relationships": [
      {{
        "type": "<parent, child, related>",
        "target": "<page or database name>",
        "reasoning": "<why this relationship>"
      }}
    ]
  }},
  
  "quality_assessment": {{
    "organization_score": <0-100>,
    "factors": {{
      "has_clear_title": <true/false>,
      "has_meaningful_content": <true/false>,
      "properly_tagged": <true/false>,
      "has_clear_purpose": <true/false>,
      "well_structured": <true/false>
    }},
    "improvement_potential": "<high, medium, low, none>"
  }}
}}
```

Provide ONLY valid JSON in your response, no additional text or explanation outside the JSON structure."""

        return prompt

    def _create_minimal_prompt(self, prepared_content: Dict[str, Any]) -> str:
        """Create ultra-minimal prompt for maximum cost savings"""

        title = prepared_content["title"][:50]
        content = prepared_content["content"][:150]

        # Ultra-compact JSON-only prompt
        return f"""Classify Notion page:
Title: {title}
Text: {content[:100]}

Respond with JSON only:
{{
  "classification": {{"primary_type": "<task|project|note|idea|meeting|reference>", "confidence": 0.0}},
  "recommendations": {{"primary_action": "<move|archive|keep|review>", "suggested_database": null}}
}}"""

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _get_ai_response(self, prompt: str) -> str:
        """Get response from Claude with retry logic"""

        if self.provider == "anthropic":
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=(
                        100
                        if self.optimization_level == OptimizationLevel.MINIMAL
                        else 500
                    ),
                    temperature=0.1,  # Very low temperature for consistent, concise responses
                    messages=[{"role": "user", "content": prompt}],
                )

                response = message.content[0].text
                logger.debug("Received response from Claude")
                return response

            except Exception as e:
                logger.error(f"Error getting Claude response: {e}")
                raise
        else:
            # OpenAI implementation
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Respond with minimal JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=(
                        100
                        if self.optimization_level == OptimizationLevel.MINIMAL
                        else 500
                    ),
                    response_format={"type": "json_object"},  # Ensure JSON response
                )

                response_text = response.choices[0].message.content
                logger.debug("Received response from OpenAI")
                return response_text

            except Exception as e:
                logger.error(f"Error getting OpenAI response: {e}")
                raise

    def _create_skipped_response(self, page_content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a response for skipped pages"""
        return {
            "page_id": page_content.get("id"),
            "page_title": page_content.get("title", "Untitled"),
            "page_url": page_content.get("url", ""),
            "analysis_timestamp": datetime.now().isoformat(),
            "skipped": True,
            "classification": {
                "primary_type": (
                    "archive" if page_content.get("archived") else "template"
                ),
                "confidence": 0.9,
                "reasoning": "Page skipped by optimization rules",
            },
            "content_analysis": {
                "summary": "Page skipped to optimize API usage",
                "key_topics": [],
            },
            "recommendations": {
                "primary_action": "no_action",
                "confidence": 0.9,
                "reasoning": "Page appears to be template/archive/minimal content",
            },
        }

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured data"""

        try:
            # Extract JSON from response
            # Sometimes the model might include markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end]
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end]

            # Parse JSON
            analysis = json.loads(response.strip())

            # Validate structure
            required_keys = [
                "classification",
                "content_analysis",
                "urgency_assessment",
                "recommendations",
            ]

            for key in required_keys:
                if key not in analysis:
                    logger.warning(f"Missing key in analysis: {key}")
                    analysis[key] = {}

            return analysis

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            logger.debug(f"Response: {response[:500]}")

            # Return default structure
            return {
                "classification": {
                    "primary_type": "unknown",
                    "confidence": 0.0,
                    "reasoning": "Failed to parse AI response",
                },
                "content_analysis": {"summary": "Analysis failed", "key_topics": []},
                "urgency_assessment": {"level": "unknown", "confidence": 0.0},
                "recommendations": {
                    "primary_action": "review",
                    "confidence": 0.0,
                    "reasoning": "Manual review required",
                },
            }

    def _generate_cache_key(self, page_content: Dict[str, Any]) -> str:
        """Generate cache key for page content"""
        content_str = json.dumps(
            {
                "title": page_content.get("title"),
                "content": page_content.get("content"),
                "properties": page_content.get("properties"),
            },
            sort_keys=True,
        )

        return hashlib.md5(content_str.encode()).hexdigest()

    def _load_cache(self):
        """Load response cache from file"""
        cache_file = self.settings.data_dir / "ai_response_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    self.response_cache = json.load(f)
                logger.info(f"Loaded {len(self.response_cache)} cached responses")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                self.response_cache = {}

    def _save_cache(self):
        """Save response cache to file"""
        cache_file = self.settings.data_dir / "ai_response_cache.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(self.response_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def batch_analyze(
        self,
        pages: List[Dict[str, Any]],
        workspace_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze multiple pages in batch"""

        results = []
        original_total = len(pages)

        # Apply advanced optimization
        pages = self.advanced_optimizer.optimize_batch(pages)
        total = len(pages)

        console.print(
            f"\n[bold cyan]Analyzing {total}/{original_total} pages with AI (Optimization: {self.optimization_level.value})...[/bold cyan]"
        )

        if total < original_total:
            console.print(
                f"[yellow]Reduced pages by {original_total - total} through advanced optimization[/yellow]"
            )

        # Use request batcher for deduplication
        unique_pages = []
        skipped_duplicates = 0
        for page in pages:
            if self.api_optimizer.request_batcher.add_request(page):
                unique_pages.append(page)
            else:
                skipped_duplicates += 1

        if skipped_duplicates > 0:
            console.print(
                f"[yellow]Skipped {skipped_duplicates} duplicate pages[/yellow]"
            )
            self.api_optimizer.metrics.duplicates_skipped += skipped_duplicates

        pages = unique_pages
        total = len(pages)

        for i, page in enumerate(pages, 1):
            # Check if should stop early (progressive analysis)
            if self.advanced_optimizer.should_stop_analysis(i, total):
                console.print(
                    f"[yellow]Stopping early due to pattern detection (analyzed {i}/{total})[/yellow]"
                )
                break

            # Check for predicted analysis
            if "predicted_analysis" in page:
                results.append(page["predicted_analysis"])
                console.print(
                    f"[dim]Using predicted classification for: {page.get('title', 'Untitled')}[/dim]"
                )
                continue

            console.print(
                f"[dim]Processing page {i}/{total}: {page.get('title', 'Untitled')}[/dim]"
            )

            try:
                analysis = self.analyze_page(page, workspace_context)
                results.append(analysis)

                # Record for pattern learning
                self.advanced_optimizer.progressive.record_analysis(analysis)

                # Show brief result
                classification = analysis.get("classification", {})
                confidence = classification.get("confidence", 0)
                doc_type = classification.get("primary_type", "unknown")

                confidence_color = (
                    "green"
                    if confidence > 0.8
                    else "yellow" if confidence > 0.6 else "red"
                )
                console.print(
                    f"  → Type: [bold]{doc_type}[/bold] "
                    f"(Confidence: [{confidence_color}]{confidence:.0%}[/{confidence_color}])"
                )

            except Exception as e:
                logger.error(f"Failed to analyze page {page.get('id')}: {e}")
                results.append(
                    {
                        "page_id": page.get("id"),
                        "page_title": page.get("title"),
                        "error": str(e),
                    }
                )

        console.print(f"\n[bold green]✓ Analysis complete![/bold green]")

        # Display optimization metrics
        usage_report = self.api_optimizer.get_usage_report()
        console.print("\n[bold]API Usage Summary:[/bold]")
        console.print(f"  Total API Calls: {usage_report['total_requests']}")
        console.print(f"  Total Tokens: {usage_report['total_tokens']:,}")
        console.print(f"  Total Cost: {usage_report['total_cost']}")
        console.print(f"  Cost Saved: {usage_report['cost_saved']}")
        console.print(f"  Cache Hit Rate: {usage_report['cache_hit_rate']}")
        console.print(f"  Duplicates Skipped: {usage_report['duplicates_skipped']}")
        console.print(
            f"  Similar Pages Skipped: {usage_report['similar_pages_skipped']}"
        )

        return results


if __name__ == "__main__":
    # Test the analyzer
    analyzer = ClaudeAnalyzer()

    # Test with sample content
    test_page = {
        "id": "test-123",
        "title": "Q4 Planning Meeting Notes",
        "content": """
        Meeting Date: October 15, 2024
        Attendees: John, Sarah, Mike
        
        Agenda:
        1. Review Q3 results
        2. Set Q4 objectives
        3. Resource allocation
        
        Key Decisions:
        - Increase marketing budget by 20%
        - Hire 2 new engineers
        - Launch new product feature by November
        
        Action Items:
        - John: Prepare budget proposal
        - Sarah: Start recruitment process
        - Mike: Finalize feature specifications
        """,
        "properties": {"Status": "In Progress", "Priority": "High"},
    }

    try:
        result = analyzer.analyze_page(test_page)
        console.print(Panel(json.dumps(result, indent=2), title="Analysis Result"))
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
