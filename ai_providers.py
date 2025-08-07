"""
AI Provider Management - Support for Claude, ChatGPT, and Gemini
Ensures only one provider is active unless explicitly configured for multi-provider
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from logger_wrapper import logger


class AIProvider(Enum):
    """Supported AI providers"""

    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    GEMINI = "gemini"


class AIModel(Enum):
    """Available AI models with their providers"""

    # Claude models
    CLAUDE_3_OPUS = ("claude-3-opus-20240229", AIProvider.CLAUDE)
    CLAUDE_3_SONNET = ("claude-3-sonnet-20240229", AIProvider.CLAUDE)
    CLAUDE_3_HAIKU = ("claude-3-haiku-20240307", AIProvider.CLAUDE)

    # ChatGPT models
    GPT_4_TURBO = ("gpt-4-turbo-preview", AIProvider.CHATGPT)
    GPT_4 = ("gpt-4", AIProvider.CHATGPT)
    GPT_3_5_TURBO = ("gpt-3.5-turbo", AIProvider.CHATGPT)

    # Gemini models
    GEMINI_PRO = ("gemini-pro", AIProvider.GEMINI)
    GEMINI_PRO_VISION = ("gemini-pro-vision", AIProvider.GEMINI)
    GEMINI_ULTRA = ("gemini-ultra", AIProvider.GEMINI)


@dataclass
class ModelInfo:
    """Information about an AI model"""

    name: str
    provider: AIProvider
    cost_per_1m_input: float
    cost_per_1m_output: float
    quality_score: int  # 1-10
    speed_score: int  # 1-10
    context_window: int
    supports_json: bool
    supports_vision: bool = False


class AIProviderManager:
    """Manages AI provider selection and configuration"""

    # Model information database
    MODEL_INFO = {
        AIModel.CLAUDE_3_OPUS: ModelInfo(
            name="Claude 3 Opus",
            provider=AIProvider.CLAUDE,
            cost_per_1m_input=15.00,
            cost_per_1m_output=75.00,
            quality_score=10,
            speed_score=7,
            context_window=200000,
            supports_json=True,
        ),
        AIModel.CLAUDE_3_SONNET: ModelInfo(
            name="Claude 3 Sonnet",
            provider=AIProvider.CLAUDE,
            cost_per_1m_input=3.00,
            cost_per_1m_output=15.00,
            quality_score=9,
            speed_score=8,
            context_window=200000,
            supports_json=True,
        ),
        AIModel.CLAUDE_3_HAIKU: ModelInfo(
            name="Claude 3 Haiku",
            provider=AIProvider.CLAUDE,
            cost_per_1m_input=0.25,
            cost_per_1m_output=1.25,
            quality_score=7,
            speed_score=9,
            context_window=200000,
            supports_json=True,
        ),
        AIModel.GPT_4_TURBO: ModelInfo(
            name="GPT-4 Turbo",
            provider=AIProvider.CHATGPT,
            cost_per_1m_input=10.00,
            cost_per_1m_output=30.00,
            quality_score=9,
            speed_score=8,
            context_window=128000,
            supports_json=True,
        ),
        AIModel.GPT_4: ModelInfo(
            name="GPT-4",
            provider=AIProvider.CHATGPT,
            cost_per_1m_input=30.00,
            cost_per_1m_output=60.00,
            quality_score=9,
            speed_score=6,
            context_window=8192,
            supports_json=True,
        ),
        AIModel.GPT_3_5_TURBO: ModelInfo(
            name="GPT-3.5 Turbo",
            provider=AIProvider.CHATGPT,
            cost_per_1m_input=0.50,
            cost_per_1m_output=1.50,
            quality_score=7,
            speed_score=9,
            context_window=16385,
            supports_json=True,
        ),
        AIModel.GEMINI_PRO: ModelInfo(
            name="Gemini Pro",
            provider=AIProvider.GEMINI,
            cost_per_1m_input=0.50,
            cost_per_1m_output=1.50,
            quality_score=8,
            speed_score=9,
            context_window=32000,
            supports_json=True,
        ),
        AIModel.GEMINI_PRO_VISION: ModelInfo(
            name="Gemini Pro Vision",
            provider=AIProvider.GEMINI,
            cost_per_1m_input=0.50,
            cost_per_1m_output=1.50,
            quality_score=8,
            speed_score=8,
            context_window=32000,
            supports_json=True,
            supports_vision=True,
        ),
        AIModel.GEMINI_ULTRA: ModelInfo(
            name="Gemini Ultra",
            provider=AIProvider.GEMINI,
            cost_per_1m_input=7.00,
            cost_per_1m_output=21.00,
            quality_score=9,
            speed_score=7,
            context_window=32000,
            supports_json=True,
        ),
    }

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize provider manager"""
        self.config_file = config_file or Path("data/ai_provider_config.json")
        self.detected_providers = self._detect_available_providers()
        self.user_preferences = self._load_user_preferences()

    def _detect_available_providers(self) -> Dict[AIProvider, Dict[str, Any]]:
        """Detect which AI providers have API keys configured"""
        providers = {}

        # Load settings to get API keys from .env file
        from config import get_settings

        settings = get_settings()

        # Check Claude (Anthropic)
        if settings.anthropic_api_key:
            providers[AIProvider.CLAUDE] = {
                "api_key": settings.anthropic_api_key,
                "available_models": [
                    AIModel.CLAUDE_3_OPUS,
                    AIModel.CLAUDE_3_SONNET,
                    AIModel.CLAUDE_3_HAIKU,
                ],
                "preferred_model": self._get_preferred_claude_model(),
            }
            logger.info("✅ Claude (Anthropic) API detected")

        # Check ChatGPT (OpenAI)
        if settings.openai_api_key:
            providers[AIProvider.CHATGPT] = {
                "api_key": settings.openai_api_key,
                "available_models": [
                    AIModel.GPT_4_TURBO,
                    AIModel.GPT_4,
                    AIModel.GPT_3_5_TURBO,
                ],
                "preferred_model": self._get_preferred_openai_model(),
            }
            logger.info("✅ ChatGPT (OpenAI) API detected")

        # Check Gemini (Google)
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            providers[AIProvider.GEMINI] = {
                "api_key": api_key,
                "available_models": [
                    AIModel.GEMINI_PRO,
                    AIModel.GEMINI_PRO_VISION,
                    AIModel.GEMINI_ULTRA,
                ],
                "preferred_model": self._get_preferred_gemini_model(),
            }
            logger.info("✅ Gemini (Google) API detected")

        return providers

    def _get_preferred_claude_model(self) -> AIModel:
        """Get user's preferred Claude model from environment"""
        model_hint = os.getenv("CLAUDE_MODEL", "").lower()
        if "haiku" in model_hint:
            return AIModel.CLAUDE_3_HAIKU
        elif "sonnet" in model_hint:
            return AIModel.CLAUDE_3_SONNET
        else:
            return AIModel.CLAUDE_3_OPUS  # Default to best

    def _get_preferred_openai_model(self) -> AIModel:
        """Get user's preferred OpenAI model from environment"""
        model_hint = os.getenv("OPENAI_MODEL", "").lower()
        if "3.5" in model_hint:
            return AIModel.GPT_3_5_TURBO
        elif "gpt-4-turbo" in model_hint or "turbo" in model_hint:
            return AIModel.GPT_4_TURBO
        else:
            return AIModel.GPT_4_TURBO  # Default to turbo

    def _get_preferred_gemini_model(self) -> AIModel:
        """Get user's preferred Gemini model from environment"""
        model_hint = os.getenv("GEMINI_MODEL", "").lower()
        if "ultra" in model_hint:
            return AIModel.GEMINI_ULTRA
        elif "vision" in model_hint:
            return AIModel.GEMINI_PRO_VISION
        else:
            return AIModel.GEMINI_PRO  # Default to pro

    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences from config file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_user_preferences(self):
        """Save user preferences to config file"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump(self.user_preferences, f, indent=2)

    def select_provider(
        self,
        preferred_provider: Optional[AIProvider] = None,
        allow_multi_provider: bool = False,
        cost_priority: bool = False,
        quality_priority: bool = False,
    ) -> Tuple[AIModel, Dict[str, Any]]:
        """
        Select the best AI provider and model based on availability and preferences

        Args:
            preferred_provider: User's preferred provider (defaults to Claude)
            allow_multi_provider: Allow using multiple providers simultaneously
            cost_priority: Prioritize cost over quality
            quality_priority: Prioritize quality over cost

        Returns:
            Selected model and configuration
        """

        # Check for conflicts if not allowing multi-provider
        if not allow_multi_provider and len(self.detected_providers) > 1:
            logger.warning(
                f"⚠️ Multiple AI providers detected: {list(self.detected_providers.keys())}"
            )

            # Check for explicit preference
            if preferred_provider and preferred_provider in self.detected_providers:
                logger.info(f"📌 Using preferred provider: {preferred_provider.value}")
                selected_provider = preferred_provider
            else:
                # Ask user or use priority order
                selected_provider = self._resolve_provider_conflict()
        elif len(self.detected_providers) == 1:
            selected_provider = list(self.detected_providers.keys())[0]
            logger.info(
                f"📌 Using single available provider: {selected_provider.value}"
            )
        elif len(self.detected_providers) == 0:
            raise ValueError(
                "No AI provider API keys found! Please configure at least one."
            )
        else:
            # Multiple providers allowed
            selected_provider = preferred_provider or AIProvider.CLAUDE
            if selected_provider not in self.detected_providers:
                selected_provider = list(self.detected_providers.keys())[0]

        # Select model based on priorities
        model = self._select_best_model(
            selected_provider,
            cost_priority=cost_priority,
            quality_priority=quality_priority,
        )

        # Get configuration
        config = self._build_configuration(model)

        # Save selection
        self.user_preferences["last_selected_provider"] = selected_provider.value
        self.user_preferences["last_selected_model"] = model.name
        self._save_user_preferences()

        return model, config

    def _resolve_provider_conflict(self) -> AIProvider:
        """Resolve conflict when multiple providers are available"""

        # Check for previous selection
        if "last_selected_provider" in self.user_preferences:
            last_provider = AIProvider(self.user_preferences["last_selected_provider"])
            if last_provider in self.detected_providers:
                logger.info(
                    f"📌 Using previously selected provider: {last_provider.value}"
                )
                return last_provider

        # Check for environment preference
        env_preference = os.getenv("PREFERRED_AI_PROVIDER", "").lower()
        if env_preference:
            for provider in self.detected_providers:
                if provider.value.lower() == env_preference:
                    logger.info(
                        f"📌 Using environment preferred provider: {provider.value}"
                    )
                    return provider

        # Default priority order
        priority_order = [
            AIProvider.CLAUDE,  # Best quality/capabilities
            AIProvider.CHATGPT,  # Good alternative
            AIProvider.GEMINI,  # Newest, good value
        ]

        for provider in priority_order:
            if provider in self.detected_providers:
                logger.info(f"📌 Using default priority provider: {provider.value}")
                return provider

        # Fallback to first available
        return list(self.detected_providers.keys())[0]

    def _select_best_model(
        self,
        provider: AIProvider,
        cost_priority: bool = False,
        quality_priority: bool = False,
    ) -> AIModel:
        """Select the best model from a provider based on priorities"""

        provider_data = self.detected_providers[provider]
        available_models = provider_data["available_models"]

        # Sort models by criteria
        if cost_priority:
            # Sort by cost (cheapest first)
            sorted_models = sorted(
                available_models,
                key=lambda m: (
                    self.MODEL_INFO[m].cost_per_1m_input
                    + self.MODEL_INFO[m].cost_per_1m_output
                ),
            )
        elif quality_priority:
            # Sort by quality (best first)
            sorted_models = sorted(
                available_models, key=lambda m: -self.MODEL_INFO[m].quality_score
            )
        else:
            # Balance cost and quality
            sorted_models = sorted(
                available_models,
                key=lambda m: (
                    -(self.MODEL_INFO[m].quality_score * 10)
                    / (
                        self.MODEL_INFO[m].cost_per_1m_input
                        + self.MODEL_INFO[m].cost_per_1m_output
                        + 1
                    )
                ),
            )

        selected_model = sorted_models[0]
        logger.info(f"📌 Selected model: {self.MODEL_INFO[selected_model].name}")
        return selected_model

    def _build_configuration(self, model: AIModel) -> Dict[str, Any]:
        """Build configuration for selected model"""

        model_info = self.MODEL_INFO[model]
        provider_data = self.detected_providers[model_info.provider]

        config = {
            "provider": model_info.provider.value,
            "model": model.value[0],  # Get model string
            "api_key": provider_data["api_key"],
            "model_info": {
                "name": model_info.name,
                "cost_per_1m_input": model_info.cost_per_1m_input,
                "cost_per_1m_output": model_info.cost_per_1m_output,
                "quality_score": model_info.quality_score,
                "speed_score": model_info.speed_score,
                "context_window": model_info.context_window,
                "supports_json": model_info.supports_json,
                "supports_vision": model_info.supports_vision,
            },
        }

        return config

    def display_available_providers(self):
        """Display all available providers and models"""

        print("\n" + "=" * 60)
        print("🤖 AVAILABLE AI PROVIDERS")
        print("=" * 60)

        if not self.detected_providers:
            print("❌ No AI providers configured!")
            print("\nPlease set one of the following environment variables:")
            print("  - ANTHROPIC_API_KEY (for Claude)")
            print("  - OPENAI_API_KEY (for ChatGPT)")
            print("  - GOOGLE_API_KEY or GEMINI_API_KEY (for Gemini)")
        else:
            for provider, data in self.detected_providers.items():
                print(f"\n{provider.value.upper()}:")
                print(f"  Status: ✅ Configured")
                print(f"  Available Models:")
                for model in data["available_models"]:
                    info = self.MODEL_INFO[model]
                    cost = info.cost_per_1m_input + info.cost_per_1m_output
                    print(
                        f"    - {info.name}: ${cost:.2f}/1M tokens, Quality: {info.quality_score}/10"
                    )

        print("=" * 60)

    def check_provider_limits(self, provider: AIProvider) -> Dict[str, Any]:
        """Check rate limits and quotas for a provider"""

        # This would ideally make API calls to check actual limits
        # For now, return estimated limits

        limits = {
            AIProvider.CLAUDE: {
                "requests_per_minute": 50,
                "tokens_per_minute": 100000,
                "daily_limit": None,
            },
            AIProvider.CHATGPT: {
                "requests_per_minute": 60,
                "tokens_per_minute": 90000,
                "daily_limit": None,
            },
            AIProvider.GEMINI: {
                "requests_per_minute": 60,
                "tokens_per_minute": 60000,
                "daily_limit": 1500000,  # Free tier limit
            },
        }

        return limits.get(provider, {})


def get_ai_provider_config(
    preferred_provider: Optional[str] = None,
    cost_priority: bool = False,
    quality_priority: bool = False,
) -> Dict[str, Any]:
    """
    Get AI provider configuration

    Args:
        preferred_provider: Preferred provider name (claude, chatgpt, gemini)
        cost_priority: Prioritize cost over quality
        quality_priority: Prioritize quality over cost

    Returns:
        Configuration dictionary
    """

    manager = AIProviderManager()

    # Convert string to enum if provided
    provider_enum = None
    if preferred_provider:
        for p in AIProvider:
            if p.value.lower() == preferred_provider.lower():
                provider_enum = p
                break

    # Select provider and model
    model, config = manager.select_provider(
        preferred_provider=provider_enum,
        cost_priority=cost_priority,
        quality_priority=quality_priority,
    )

    return config


if __name__ == "__main__":
    # Test provider detection and selection
    manager = AIProviderManager()
    manager.display_available_providers()

    # Test selection
    try:
        model, config = manager.select_provider()
        print(f"\n✅ Selected: {config['model_info']['name']}")
        print(f"   Provider: {config['provider']}")
        print(
            f"   Cost: ${config['model_info']['cost_per_1m_input']:.2f} input + ${config['model_info']['cost_per_1m_output']:.2f} output per 1M tokens"
        )
    except ValueError as e:
        print(f"\n❌ {e}")
