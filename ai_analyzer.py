"""
Universal AI Analyzer - Supports Claude, ChatGPT, and Gemini
Handles all AI-powered classification and recommendation generation
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from logger_wrapper import logger
from rich.console import Console
from rich.panel import Panel

from config import Settings, get_settings
from api_optimizer import APIOptimizer, OptimizationLevel
from ai_providers import AIProvider, AIProviderManager, get_ai_provider_config


console = Console()


class UniversalAIAnalyzer:
    """Universal AI analyzer supporting multiple providers"""
    
    def __init__(
        self, 
        settings: Optional[Settings] = None, 
        optimization_level: OptimizationLevel = OptimizationLevel.MINIMAL,
        preferred_provider: Optional[str] = None
    ):
        """Initialize universal AI analyzer"""
        self.settings = settings or get_settings()
        self.optimization_level = optimization_level
        
        # Initialize API optimizer for minimal token usage
        self.api_optimizer = APIOptimizer(self.settings, optimization_level)
        logger.info(f"API Optimization Level: {optimization_level.value}")
        
        # Get AI provider configuration
        self.ai_config = self._configure_ai_provider(preferred_provider)
        
        # Initialize the appropriate client
        self.client = self._initialize_client()
        
        # Cache for responses
        self.response_cache = {}
        self._load_cache()
        
        logger.info(f"AI Analyzer initialized with {self.ai_config['provider']} - {self.ai_config['model_info']['name']}")
    
    def _configure_ai_provider(self, preferred_provider: Optional[str]) -> Dict[str, Any]:
        """Configure AI provider with automatic selection"""
        
        # Check if user specified a provider
        if preferred_provider:
            logger.info(f"Using user-specified provider: {preferred_provider}")
        else:
            # Check environment for preference
            preferred_provider = os.getenv("PREFERRED_AI_PROVIDER")
            if preferred_provider:
                logger.info(f"Using environment-specified provider: {preferred_provider}")
        
        # Get configuration
        config = get_ai_provider_config(
            preferred_provider=preferred_provider,
            cost_priority=(self.optimization_level == OptimizationLevel.MINIMAL),
            quality_priority=(self.optimization_level == OptimizationLevel.FULL)
        )
        
        return config
    
    def _initialize_client(self):
        """Initialize the appropriate AI client"""
        
        provider = self.ai_config["provider"]
        
        if provider == "claude":
            return self._init_claude_client()
        elif provider == "chatgpt":
            return self._init_openai_client()
        elif provider == "gemini":
            return self._init_gemini_client()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_claude_client(self):
        """Initialize Claude/Anthropic client"""
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.ai_config["api_key"])
            self.provider_type = "claude"
            return client
        except ImportError:
            logger.error("Anthropic library not installed. Install with: pip install anthropic")
            raise
    
    def _init_openai_client(self):
        """Initialize OpenAI/ChatGPT client"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.ai_config["api_key"])
            self.provider_type = "openai"
            return client
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            raise
    
    def _init_gemini_client(self):
        """Initialize Google Gemini client"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.ai_config["api_key"])
            
            # Create model
            model = genai.GenerativeModel(
                self.ai_config["model"],
                generation_config={
                    "temperature": 0.3,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 500 if self.optimization_level == OptimizationLevel.MINIMAL else 2000,
                }
            )
            self.provider_type = "gemini"
            return model
        except ImportError:
            logger.error("Google GenerativeAI library not installed. Install with: pip install google-generativeai")
            raise
    
    def analyze_page(
        self,
        page_content: Dict[str, Any],
        workspace_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze a single page and return classification and recommendations"""
        
        # Check if page should be skipped (minimal mode optimization)
        if self.api_optimizer.should_skip_page(page_content):
            logger.info(f"Skipping page: {page_content.get('title', 'Untitled')} (optimization)")
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
        self.api_optimizer.record_api_usage(input_tokens, output_tokens, from_cache=False)
        
        cost = self.api_optimizer.token_optimizer.calculate_cost(input_tokens, output_tokens)
        logger.info(f"API Call - Provider: {self.provider_type}, Tokens: {input_tokens}→{output_tokens}, Cost: ${cost:.4f}")
        
        # Parse response
        analysis = self._parse_analysis_response(response)
        
        # Add metadata
        analysis["page_id"] = page_content.get("id")
        analysis["page_title"] = page_content.get("title", "Untitled")
        analysis["page_url"] = page_content.get("url", "")
        analysis["analysis_timestamp"] = datetime.now().isoformat()
        analysis["ai_provider"] = self.provider_type
        analysis["ai_model"] = self.ai_config["model_info"]["name"]
        
        # Cache result in both caches
        if self.settings.enable_caching:
            self.response_cache[cache_key] = analysis
            self._save_cache()
            
            # Store in smart cache for similarity detection
            self.api_optimizer.smart_cache.store(page_content, analysis)
        
        return analysis
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _get_ai_response(self, prompt: str) -> str:
        """Get response from AI provider with retry logic"""
        
        if self.provider_type == "claude":
            return self._get_claude_response(prompt)
        elif self.provider_type == "openai":
            return self._get_openai_response(prompt)
        elif self.provider_type == "gemini":
            return self._get_gemini_response(prompt)
        else:
            raise ValueError(f"Unknown provider type: {self.provider_type}")
    
    def _get_claude_response(self, prompt: str) -> str:
        """Get response from Claude"""
        try:
            message = self.client.messages.create(
                model=self.ai_config["model"],
                max_tokens=500 if self.optimization_level == OptimizationLevel.MINIMAL else 2000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            response = message.content[0].text
            logger.debug("Received response from Claude")
            return response
            
        except Exception as e:
            logger.error(f"Error getting Claude response: {e}")
            raise
    
    def _get_openai_response(self, prompt: str) -> str:
        """Get response from ChatGPT/OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.ai_config["model"],
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Notion workspace organizer. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500 if self.optimization_level == OptimizationLevel.MINIMAL else 2000,
                response_format={"type": "json_object"} if self.ai_config["model_info"]["supports_json"] else None
            )
            
            response_text = response.choices[0].message.content
            logger.debug("Received response from ChatGPT")
            return response_text
            
        except Exception as e:
            logger.error(f"Error getting ChatGPT response: {e}")
            raise
    
    def _get_gemini_response(self, prompt: str) -> str:
        """Get response from Gemini"""
        try:
            # Add JSON instruction to prompt for Gemini
            gemini_prompt = prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON, no other text."
            
            response = self.client.generate_content(gemini_prompt)
            response_text = response.text
            
            # Gemini sometimes adds markdown, strip it
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            logger.debug("Received response from Gemini")
            return response_text
            
        except Exception as e:
            logger.error(f"Error getting Gemini response: {e}")
            raise
    
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
        
        return {
            "title": title,
            "content": content,
            "properties": properties,
            "created_time": created_time,
            "last_edited_time": last_edited_time,
            "content_length": len(content),
            "has_properties": bool(properties)
        }
    
    def _create_analysis_prompt(
        self,
        prepared_content: Dict[str, Any],
        workspace_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create analysis prompt for AI"""
        
        # Build the main prompt (simplified for all providers)
        prompt = f"""Analyze this Notion page and provide classification and recommendations.

Page Title: {prepared_content['title']}
Content: {prepared_content['content'][:1000]}

Provide a JSON response with:
- classification: primary_type (task/project/meeting_note/idea/journal/reference/archive), confidence (0-1)
- recommendations: primary_action (move_to_database/archive/delete/review/no_action), suggested_database
- urgency: level (immediate/this_week/this_month/someday/no_deadline)

Respond with valid JSON only."""
        
        return prompt
    
    def _create_skipped_response(self, page_content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a response for skipped pages"""
        return {
            "page_id": page_content.get("id"),
            "page_title": page_content.get("title", "Untitled"),
            "page_url": page_content.get("url", ""),
            "analysis_timestamp": datetime.now().isoformat(),
            "skipped": True,
            "classification": {
                "primary_type": "archive" if page_content.get("archived") else "template",
                "confidence": 0.9,
                "reasoning": "Page skipped by optimization rules"
            },
            "recommendations": {
                "primary_action": "no_action",
                "confidence": 0.9,
                "reasoning": "Page appears to be template/archive/minimal content"
            }
        }
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured data"""
        
        try:
            # Parse JSON
            analysis = json.loads(response.strip())
            
            # Ensure required keys exist
            if "classification" not in analysis:
                analysis["classification"] = {
                    "primary_type": "unknown",
                    "confidence": 0.0
                }
            
            if "recommendations" not in analysis:
                analysis["recommendations"] = {
                    "primary_action": "review",
                    "confidence": 0.0
                }
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            logger.debug(f"Response: {response[:500]}")
            
            # Return default structure
            return {
                "classification": {
                    "primary_type": "unknown",
                    "confidence": 0.0,
                    "reasoning": "Failed to parse AI response"
                },
                "recommendations": {
                    "primary_action": "review",
                    "confidence": 0.0,
                    "reasoning": "Manual review required"
                }
            }
    
    def _generate_cache_key(self, page_content: Dict[str, Any]) -> str:
        """Generate cache key for page content"""
        try:
            # Convert any sets to lists for JSON serialization
            props = page_content.get("properties", {})
            if isinstance(props, dict):
                # Convert any sets in the properties to lists
                safe_props = {}
                for k, v in props.items():
                    if isinstance(v, set):
                        safe_props[k] = list(v)
                    else:
                        safe_props[k] = v
                props = safe_props
            
            content_str = json.dumps({
                "title": page_content.get("title"),
                "content": page_content.get("content"),
                "properties": props
            }, sort_keys=True, default=str)
        except Exception as e:
            logger.warning(f"Could not create content hash: {e}")
            content_str = f"{page_content.get('title', '')}_{page_content.get('id', '')}"
        
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
        workspace_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Analyze multiple pages in batch"""
        
        results = []
        total = len(pages)
        
        provider_name = self.ai_config["model_info"]["name"]
        console.print(f"\n[bold cyan]Analyzing {total} pages with {provider_name} (Optimization: {self.optimization_level.value})...[/bold cyan]")
        
        # Use request batcher for deduplication
        unique_pages = []
        skipped_duplicates = 0
        for page in pages:
            if self.api_optimizer.request_batcher.add_request(page):
                unique_pages.append(page)
            else:
                skipped_duplicates += 1
        
        if skipped_duplicates > 0:
            console.print(f"[yellow]Skipped {skipped_duplicates} duplicate pages[/yellow]")
            self.api_optimizer.metrics.duplicates_skipped += skipped_duplicates
        
        pages = unique_pages
        total = len(pages)
        
        for i, page in enumerate(pages, 1):
            console.print(f"[dim]Processing page {i}/{total}: {page.get('title', 'Untitled')}[/dim]")
            
            try:
                analysis = self.analyze_page(page, workspace_context)
                results.append(analysis)
                
                # Show brief result
                classification = analysis.get("classification", {})
                confidence = classification.get("confidence", 0)
                doc_type = classification.get("primary_type", "unknown")
                
                confidence_color = "green" if confidence > 0.8 else "yellow" if confidence > 0.6 else "red"
                console.print(
                    f"  → Type: [bold]{doc_type}[/bold] "
                    f"(Confidence: [{confidence_color}]{confidence:.0%}[/{confidence_color}])"
                )
                
            except Exception as e:
                logger.error(f"Failed to analyze page {page.get('id')}: {e}")
                results.append({
                    "page_id": page.get("id"),
                    "page_title": page.get("title"),
                    "error": str(e)
                })
        
        console.print(f"\n[bold green]✓ Analysis complete![/bold green]")
        
        # Display optimization metrics
        usage_report = self.api_optimizer.get_usage_report()
        console.print("\n[bold]API Usage Summary:[/bold]")
        console.print(f"  Provider: {self.ai_config['provider'].upper()}")
        console.print(f"  Model: {self.ai_config['model_info']['name']}")
        console.print(f"  Total API Calls: {usage_report['total_requests']}")
        console.print(f"  Total Tokens: {usage_report['total_tokens']:,}")
        console.print(f"  Total Cost: {usage_report['total_cost']}")
        console.print(f"  Cost Saved: {usage_report['cost_saved']}")
        console.print(f"  Cache Hit Rate: {usage_report['cache_hit_rate']}")
        
        return results


# Import os for environment variable checking
import os


if __name__ == "__main__":
    # Test the analyzer
    analyzer = UniversalAIAnalyzer()
    
    # Display provider info
    print(f"\n✅ Using: {analyzer.ai_config['model_info']['name']}")
    print(f"   Provider: {analyzer.ai_config['provider']}")
    print(f"   Cost: ${analyzer.ai_config['model_info']['cost_per_1m_input']}/1M input tokens")