"""
Tests for configuration module
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import (
    CLASSIFICATION_CONFIG,
    HEALTH_METRICS_CONFIG,
    AIModel,
    Environment,
    LogLevel,
    Settings,
    get_settings,
)


class TestSettings:
    """Test Settings class and validation"""

    def test_settings_initialization(self, mock_settings):
        """Test basic settings initialization"""
        assert mock_settings.notion_api_key == "test_notion_key"
        assert mock_settings.notion_inbox_database_id == "test_db_id"
        assert mock_settings.anthropic_api_key == "test_anthropic_key"
        assert mock_settings.batch_size == 5
        assert mock_settings.enable_caching is True

    def test_settings_from_env(self):
        """Test settings loading from environment variables"""
        env_vars = {
            "NOTION_API_KEY": "env_notion_key",
            "NOTION_INBOX_DATABASE_ID": "env_db_id",
            "ANTHROPIC_API_KEY": "env_anthropic_key",
            "BATCH_SIZE": "20",
            "ENABLE_CACHING": "false",
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings()
            assert settings.notion_api_key == "env_notion_key"
            assert settings.notion_inbox_database_id == "env_db_id"
            assert settings.batch_size == 20
            assert settings.enable_caching is False

    def test_batch_size_validation(self):
        """Test batch size validation constraints"""
        # Valid batch size
        settings = Settings(
            notion_api_key="key",
            notion_inbox_database_id="db",
            anthropic_api_key="key",
            batch_size=50,
        )
        assert settings.batch_size == 50

        # Test upper bound
        settings = Settings(
            notion_api_key="key",
            notion_inbox_database_id="db",
            anthropic_api_key="key",
            batch_size=100,
        )
        assert settings.batch_size == 100

        # Test exceeding upper bound
        with pytest.raises(ValueError):
            Settings(
                notion_api_key="key",
                notion_inbox_database_id="db",
                anthropic_api_key="key",
                batch_size=101,
            )

    def test_api_key_validation(self):
        """Test that at least one AI API key is required"""
        # Use a context manager to temporarily clear environment variables
        import os

        env_backup = os.environ.copy()

        # Clear AI API keys from environment
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
            if key in os.environ:
                del os.environ[key]

        try:
            # Valid with Anthropic key
            settings = Settings(
                notion_api_key="key",
                notion_inbox_database_id="db",
                anthropic_api_key="anthropic_key",
            )
            assert settings.anthropic_api_key == "anthropic_key"

            # Valid with OpenAI key
            settings = Settings(
                notion_api_key="key",
                notion_inbox_database_id="db",
                openai_api_key="openai_key",
            )
            assert settings.openai_api_key == "openai_key"

            # Invalid without any AI key
            with pytest.raises(ValueError, match="At least one AI API key"):
                Settings(
                    notion_api_key="key",
                    notion_inbox_database_id="db",
                    anthropic_api_key=None,
                    openai_api_key=None,
                )
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(env_backup)

    def test_encryption_key_validation(self):
        """Test encryption key validation"""
        # Valid without encryption
        settings = Settings(
            notion_api_key="key",
            notion_inbox_database_id="db",
            anthropic_api_key="key",
            enable_encryption=False,
        )
        assert settings.enable_encryption is False

        # Valid with encryption and key
        settings = Settings(
            notion_api_key="key",
            notion_inbox_database_id="db",
            anthropic_api_key="key",
            enable_encryption=True,
            encryption_key="secret_key",
        )
        assert settings.encryption_key == "secret_key"

        # Invalid with encryption but no key
        with pytest.raises(ValueError, match="Encryption key must be provided"):
            Settings(
                notion_api_key="key",
                notion_inbox_database_id="db",
                anthropic_api_key="key",
                enable_encryption=True,
                encryption_key=None,
            )

    def test_directory_creation(self):
        """Test automatic directory creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            data_dir = Path(tmpdir) / "test_data"

            settings = Settings(
                notion_api_key="key",
                notion_inbox_database_id="db",
                anthropic_api_key="key",
                output_dir=output_dir,
                data_dir=data_dir,
            )

            assert output_dir.exists()
            assert data_dir.exists()

    def test_cache_ttl_property(self, mock_settings):
        """Test cache TTL timedelta conversion"""
        from datetime import timedelta

        mock_settings.cache_ttl_hours = 24
        assert mock_settings.cache_ttl == timedelta(hours=24)

        mock_settings.cache_ttl_hours = 48
        assert mock_settings.cache_ttl == timedelta(hours=48)

    def test_environment_checks(self, mock_settings):
        """Test environment helper properties"""
        mock_settings.app_env = Environment.PRODUCTION
        assert mock_settings.is_production is True
        assert mock_settings.is_development is False

        mock_settings.app_env = Environment.DEVELOPMENT
        assert mock_settings.is_production is False
        assert mock_settings.is_development is True

    def test_get_ai_config(self, mock_settings):
        """Test AI configuration retrieval"""
        # With Anthropic key
        mock_settings.anthropic_api_key = "anthropic_key"
        mock_settings.openai_api_key = None
        config = mock_settings.get_ai_config()
        assert config["provider"] == "anthropic"
        assert config["api_key"] == "anthropic_key"
        assert config["model"] == AIModel.CLAUDE_3_OPUS

        # With OpenAI key only
        mock_settings.anthropic_api_key = None
        mock_settings.openai_api_key = "openai_key"
        config = mock_settings.get_ai_config()
        assert config["provider"] == "openai"
        assert config["api_key"] == "openai_key"

        # With no keys
        mock_settings.anthropic_api_key = None
        mock_settings.openai_api_key = None
        with pytest.raises(ValueError, match="No AI API key configured"):
            mock_settings.get_ai_config()

    def test_get_notion_headers(self, mock_settings):
        """Test Notion API headers generation"""
        headers = mock_settings.get_notion_headers()
        assert headers["Authorization"] == "Bearer test_notion_key"
        assert headers["Notion-Version"] == "2022-06-28"
        assert headers["Content-Type"] == "application/json"


class TestConfigurationConstants:
    """Test configuration constants"""

    def test_classification_config_structure(self):
        """Test CLASSIFICATION_CONFIG structure"""
        assert "document_types" in CLASSIFICATION_CONFIG
        assert "urgency_levels" in CLASSIFICATION_CONFIG
        assert "contexts" in CLASSIFICATION_CONFIG
        assert "actions" in CLASSIFICATION_CONFIG
        assert "confidence_thresholds" in CLASSIFICATION_CONFIG

        # Check document types
        assert "task" in CLASSIFICATION_CONFIG["document_types"]
        assert "project" in CLASSIFICATION_CONFIG["document_types"]
        assert "meeting_note" in CLASSIFICATION_CONFIG["document_types"]

        # Check confidence thresholds
        thresholds = CLASSIFICATION_CONFIG["confidence_thresholds"]
        assert thresholds["high"] == 0.8
        assert thresholds["medium"] == 0.6
        assert thresholds["low"] == 0.4

    def test_health_metrics_config_structure(self):
        """Test HEALTH_METRICS_CONFIG structure"""
        assert "organization_score" in HEALTH_METRICS_CONFIG
        assert "efficiency_metrics" in HEALTH_METRICS_CONFIG

        org_score = HEALTH_METRICS_CONFIG["organization_score"]
        assert "factors" in org_score
        assert "weights" in org_score

        # Check weights sum to 1.0
        weights = org_score["weights"]
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01  # Allow for floating point errors


class TestEnums:
    """Test enum definitions"""

    def test_environment_enum(self):
        """Test Environment enum values"""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"

    def test_log_level_enum(self):
        """Test LogLevel enum values"""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"

    def test_ai_model_enum(self):
        """Test AIModel enum values"""
        assert AIModel.CLAUDE_3_OPUS.value == "claude-3-opus-20240229"
        assert AIModel.CLAUDE_3_SONNET.value == "claude-3-sonnet-20240229"
        assert AIModel.CLAUDE_3_HAIKU.value == "claude-3-haiku-20240307"
        assert AIModel.GPT_4_TURBO.value == "gpt-4-turbo-preview"
        assert AIModel.GPT_4.value == "gpt-4"


class TestGetSettings:
    """Test get_settings function"""

    @patch.dict(
        os.environ,
        {
            "NOTION_API_KEY": "test_key",
            "NOTION_INBOX_DATABASE_ID": "test_db",
            "ANTHROPIC_API_KEY": "test_anthropic",
        },
    )
    def test_get_settings_singleton(self):
        """Test that get_settings returns a valid Settings instance"""
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.notion_api_key == "test_key"
        assert settings.notion_inbox_database_id == "test_db"
        assert settings.anthropic_api_key == "test_anthropic"
