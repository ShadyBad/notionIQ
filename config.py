"""
Configuration module for NotionIQ
Handles all environment variables and application settings
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
from enum import Enum
import os
from datetime import timedelta


class Environment(str, Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AIModel(str, Enum):
    """Supported AI models"""
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"


class Settings(BaseSettings):
    """Application settings with validation"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Notion Configuration
    notion_api_key: str = Field(
        ...,
        description="Notion integration token"
    )
    notion_inbox_database_id: str = Field(
        ...,
        description="ID of the Inbox database to process"
    )
    notion_workspace_id: Optional[str] = Field(
        None,
        description="Optional workspace ID for multi-workspace support"
    )
    notion_api_version: str = Field(
        "2022-06-28",
        description="Notion API version"
    )
    
    # AI Configuration
    anthropic_api_key: Optional[str] = Field(
        None,
        description="Anthropic API key for Claude"
    )
    claude_model: AIModel = Field(
        AIModel.CLAUDE_3_OPUS,
        description="Claude model to use"
    )
    openai_api_key: Optional[str] = Field(
        None,
        description="Optional OpenAI API key"
    )
    openai_model: str = Field(
        "gpt-4-turbo-preview",
        description="OpenAI model to use"
    )
    
    # Application Settings
    app_env: Environment = Field(
        Environment.DEVELOPMENT,
        description="Application environment"
    )
    log_level: LogLevel = Field(
        LogLevel.INFO,
        description="Logging level"
    )
    output_dir: Path = Field(
        Path("output"),
        description="Directory for output files"
    )
    data_dir: Path = Field(
        Path("data"),
        description="Directory for data files"
    )
    
    # Processing Settings
    batch_size: int = Field(
        10,
        ge=1,
        le=100,
        description="Number of pages to process in batch"
    )
    max_content_length: int = Field(
        10000,
        ge=1000,
        le=50000,
        description="Maximum content length per page"
    )
    enable_caching: bool = Field(
        True,
        description="Enable response caching"
    )
    cache_ttl_hours: int = Field(
        24,
        ge=1,
        le=168,
        description="Cache time-to-live in hours"
    )
    
    # Rate Limiting
    rate_limit_requests_per_second: float = Field(
        3.0,
        ge=0.1,
        le=10.0,
        description="API rate limit"
    )
    
    # Security Settings
    enable_encryption: bool = Field(
        False,
        description="Enable data encryption"
    )
    encryption_key: Optional[str] = Field(
        None,
        description="Encryption key for sensitive data"
    )
    
    # Feature Flags
    enable_workspace_scan: bool = Field(
        True,
        description="Enable full workspace scanning"
    )
    enable_pattern_learning: bool = Field(
        True,
        description="Enable pattern learning from user actions"
    )
    enable_auto_organization: bool = Field(
        False,
        description="Enable automatic page organization"
    )
    enable_recommendations_page: bool = Field(
        True,
        description="Create/update recommendations page in Notion"
    )
    auto_execute: bool = Field(
        False,
        description="Automatically execute recommendations without confirmation"
    )
    
    # Analytics Settings
    enable_analytics: bool = Field(
        True,
        description="Enable analytics tracking"
    )
    track_time_saved: bool = Field(
        True,
        description="Track time saved metrics"
    )
    track_accuracy: bool = Field(
        True,
        description="Track classification accuracy"
    )
    
    @field_validator("output_dir", "data_dir", mode="before")
    @classmethod
    def create_directories(cls, v):
        """Ensure directories exist"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @model_validator(mode="after")
    def validate_ai_keys(self):
        """Ensure at least one AI API key is provided and validate format"""
        if not self.anthropic_api_key and not self.openai_api_key:
            raise ValueError(
                "At least one AI API key (Anthropic or OpenAI) must be provided"
            )
        
        # Import here to avoid circular dependency
        try:
            from security import SecurityValidator
            
            # Validate API key formats if provided
            if self.notion_api_key and not SecurityValidator.validate_notion_api_key(self.notion_api_key):
                raise ValueError("Invalid Notion API key format")
            
            if self.anthropic_api_key and not SecurityValidator.validate_anthropic_api_key(self.anthropic_api_key):
                raise ValueError("Invalid Anthropic API key format")
            
            if self.openai_api_key and not SecurityValidator.validate_openai_api_key(self.openai_api_key):
                raise ValueError("Invalid OpenAI API key format")
        except ImportError:
            # Security module not available, skip validation
            pass
        
        return self
    
    @model_validator(mode="after")
    def validate_encryption(self):
        """Ensure encryption key is provided if encryption is enabled"""
        if self.enable_encryption and not self.encryption_key:
            raise ValueError(
                "Encryption key must be provided when encryption is enabled"
            )
        return self
    
    @property
    def cache_ttl(self) -> timedelta:
        """Get cache TTL as timedelta"""
        return timedelta(hours=self.cache_ttl_hours)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.app_env == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.app_env == Environment.DEVELOPMENT
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI configuration based on available keys"""
        if self.anthropic_api_key:
            return {
                "provider": "anthropic",
                "api_key": self.anthropic_api_key,
                "model": self.claude_model,
            }
        elif self.openai_api_key:
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
            }
        else:
            raise ValueError("No AI API key configured")
    
    def get_notion_headers(self) -> Dict[str, str]:
        """Get Notion API headers"""
        return {
            "Authorization": f"Bearer {self.notion_api_key}",
            "Notion-Version": self.notion_api_version,
            "Content-Type": "application/json"
        }


# Classification categories configuration
CLASSIFICATION_CONFIG = {
    "document_types": [
        "task",
        "project",
        "meeting_note",
        "idea",
        "journal",
        "reference",
        "sop",
        "goal",
        "archive"
    ],
    "urgency_levels": [
        "immediate",
        "this_week",
        "this_month",
        "someday",
        "no_deadline"
    ],
    "contexts": [
        "work",
        "personal",
        "learning",
        "creative",
        "administrative"
    ],
    "actions": [
        "move_to_database",
        "archive",
        "delete",
        "review",
        "break_down",
        "merge_with",
        "no_action"
    ],
    "confidence_thresholds": {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    }
}


# Workspace health metrics configuration  
HEALTH_METRICS_CONFIG = {
    "organization_score": {
        "factors": [
            "proper_categorization",
            "consistent_naming",
            "complete_metadata",
            "no_duplicates",
            "clear_relationships"
        ],
        "weights": {
            "proper_categorization": 0.3,
            "consistent_naming": 0.2,
            "complete_metadata": 0.2,
            "no_duplicates": 0.2,
            "clear_relationships": 0.1
        }
    },
    "efficiency_metrics": {
        "retrieval_time": "average_seconds_to_find",
        "redundancy_rate": "duplicate_content_percentage",
        "completion_rate": "tasks_completed_on_time",
        "organization_rate": "items_properly_filed"
    }
}


def get_settings() -> Settings:
    """Get application settings singleton"""
    return Settings()


if __name__ == "__main__":
    # Test configuration loading
    try:
        settings = get_settings()
        print("✅ Configuration loaded successfully!")
        print(f"Environment: {settings.app_env}")
        print(f"Notion Database: {settings.notion_inbox_database_id}")
        print(f"AI Provider: {settings.get_ai_config()['provider']}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")