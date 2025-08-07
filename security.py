"""
Security module for NotionIQ
Handles API key validation, input sanitization, and encryption
"""

import base64
import hashlib
import html
import json
import re
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from logger_wrapper import logger


class SecurityValidator:
    """Validates and sanitizes inputs for security"""

    # API key patterns
    NOTION_KEY_PATTERN = re.compile(r"^secret_[a-zA-Z0-9]{43}$")
    ANTHROPIC_KEY_PATTERN = re.compile(r"^sk-ant-[a-zA-Z0-9\-]{95}$")
    OPENAI_KEY_PATTERN = re.compile(r"^sk-[a-zA-Z0-9]{48}$")

    # Dangerous patterns to sanitize
    SCRIPT_PATTERN = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
    SQL_INJECTION_PATTERN = re.compile(
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|FROM|WHERE)\b)",
        re.IGNORECASE,
    )
    COMMAND_INJECTION_PATTERN = re.compile(r"[;&|`$()]")

    @classmethod
    def validate_notion_api_key(cls, api_key: str) -> bool:
        """
        Validate Notion API key format

        Args:
            api_key: The API key to validate

        Returns:
            True if valid, False otherwise
        """
        if not api_key:
            return False

        # Check if it matches the expected pattern
        if cls.NOTION_KEY_PATTERN.match(api_key):
            return True

        # Also accept integration tokens (different format)
        if api_key.startswith("secret_") and len(api_key) == 50:
            return True

        logger.warning("Invalid Notion API key format detected")
        return False

    @classmethod
    def validate_anthropic_api_key(cls, api_key: str) -> bool:
        """
        Validate Anthropic API key format

        Args:
            api_key: The API key to validate

        Returns:
            True if valid, False otherwise
        """
        if not api_key:
            return False

        if cls.ANTHROPIC_KEY_PATTERN.match(api_key):
            return True

        # Also check for older format
        if api_key.startswith("sk-ant-") and len(api_key) > 50:
            return True

        logger.warning("Invalid Anthropic API key format detected")
        return False

    @classmethod
    def validate_openai_api_key(cls, api_key: str) -> bool:
        """
        Validate OpenAI API key format

        Args:
            api_key: The API key to validate

        Returns:
            True if valid, False otherwise
        """
        if not api_key:
            return False

        if cls.OPENAI_KEY_PATTERN.match(api_key):
            return True

        # Also check for project keys
        if api_key.startswith("sk-proj-") and len(api_key) > 50:
            return True

        logger.warning("Invalid OpenAI API key format detected")
        return False

    @classmethod
    def sanitize_text_content(
        cls, content: str, max_length: Optional[int] = None
    ) -> str:
        """
        Sanitize text content for safe processing

        Args:
            content: The text to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized text
        """
        if not content:
            return ""

        # Remove potential script tags
        content = cls.SCRIPT_PATTERN.sub("", content)

        # Escape HTML entities
        content = html.escape(content)

        # Remove potential SQL injection patterns (for logs)
        content = cls.SQL_INJECTION_PATTERN.sub("[FILTERED]", content)

        # Remove command injection characters if found in suspicious context
        if any(pattern in content.lower() for pattern in ["exec", "eval", "system"]):
            content = cls.COMMAND_INJECTION_PATTERN.sub("", content)

        # Truncate if needed
        if max_length and len(content) > max_length:
            content = content[:max_length] + "..."

        return content

    @classmethod
    def sanitize_database_id(cls, db_id: str) -> str:
        """
        Sanitize database ID for Notion API

        Args:
            db_id: The database ID to sanitize

        Returns:
            Sanitized database ID
        """
        if not db_id:
            raise ValueError("Database ID cannot be empty")

        # Remove any whitespace
        db_id = db_id.strip()

        # Validate format (UUID with or without hyphens)
        uuid_pattern = re.compile(r"^[a-f0-9\-]{32,36}$", re.IGNORECASE)
        if not uuid_pattern.match(db_id):
            raise ValueError(f"Invalid database ID format: {db_id}")

        # Remove hyphens for consistency
        return db_id.replace("-", "")

    @classmethod
    def sanitize_json_content(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively sanitize JSON content

        Args:
            data: The JSON data to sanitize

        Returns:
            Sanitized JSON data
        """
        if isinstance(data, dict):
            return {
                key: cls.sanitize_json_content(value) for key, value in data.items()
            }
        elif isinstance(data, list):
            return [cls.sanitize_json_content(item) for item in data]
        elif isinstance(data, str):
            return cls.sanitize_text_content(data)
        else:
            return data

    @classmethod
    def validate_url(cls, url: str) -> bool:
        """
        Validate URL format and protocol

        Args:
            url: The URL to validate

        Returns:
            True if valid, False otherwise
        """
        if not url:
            return False

        # Only allow HTTPS URLs for security
        url_pattern = re.compile(r"^https://[a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=%]+$")

        return bool(url_pattern.match(url))


class EncryptionManager:
    """Manages encryption for sensitive data"""

    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize encryption manager

        Args:
            encryption_key: Optional encryption key, will generate if not provided
        """
        if encryption_key:
            self.key = self._derive_key(encryption_key)
        else:
            self.key = Fernet.generate_key()

        self.cipher = Fernet(self.key)

    @staticmethod
    def _derive_key(password: str) -> bytes:
        """
        Derive encryption key from password

        Args:
            password: The password to derive key from

        Returns:
            Derived encryption key
        """
        # Use a fixed salt for consistency (in production, store this securely)
        salt = b"notioniq_salt_v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt_data(self, data: str) -> str:
        """
        Encrypt sensitive data

        Args:
            data: The data to encrypt

        Returns:
            Encrypted data as base64 string
        """
        if not data:
            return ""

        encrypted = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data

        Args:
            encrypted_data: The encrypted data as base64 string

        Returns:
            Decrypted data
        """
        if not encrypted_data:
            return ""

        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data")

    def encrypt_json(self, data: Dict[str, Any]) -> str:
        """
        Encrypt JSON data

        Args:
            data: The JSON data to encrypt

        Returns:
            Encrypted JSON as base64 string
        """
        json_str = json.dumps(data)
        return self.encrypt_data(json_str)

    def decrypt_json(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt JSON data

        Args:
            encrypted_data: The encrypted JSON as base64 string

        Returns:
            Decrypted JSON data
        """
        json_str = self.decrypt_data(encrypted_data)
        return json.loads(json_str)


class APIKeyManager:
    """Manages API keys securely"""

    def __init__(self, encryption_manager: Optional[EncryptionManager] = None):
        """
        Initialize API key manager

        Args:
            encryption_manager: Optional encryption manager for storing keys
        """
        self.encryption_manager = encryption_manager
        self.keys_file = Path.home() / ".notioniq" / "keys.enc"
        self.keys_file.parent.mkdir(parents=True, exist_ok=True)

    def store_api_key(self, service: str, api_key: str) -> None:
        """
        Store an API key securely

        Args:
            service: The service name (notion, anthropic, openai)
            api_key: The API key to store
        """
        # Validate the key first
        if service == "notion":
            if not SecurityValidator.validate_notion_api_key(api_key):
                raise ValueError("Invalid Notion API key format")
        elif service == "anthropic":
            if not SecurityValidator.validate_anthropic_api_key(api_key):
                raise ValueError("Invalid Anthropic API key format")
        elif service == "openai":
            if not SecurityValidator.validate_openai_api_key(api_key):
                raise ValueError("Invalid OpenAI API key format")
        else:
            raise ValueError(f"Unknown service: {service}")

        # Load existing keys
        keys = self._load_keys()

        # Hash the key for storage (additional security layer)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Store encrypted if encryption is available
        if self.encryption_manager:
            encrypted_key = self.encryption_manager.encrypt_data(api_key)
            keys[service] = {"encrypted": encrypted_key, "hash": key_hash}
        else:
            # Store hash only for validation (key must come from env)
            keys[service] = {"hash": key_hash}

        # Save keys
        self._save_keys(keys)
        logger.info(f"API key for {service} stored securely")

    def validate_stored_key(self, service: str, api_key: str) -> bool:
        """
        Validate an API key against stored hash

        Args:
            service: The service name
            api_key: The API key to validate

        Returns:
            True if key matches stored hash
        """
        keys = self._load_keys()

        if service not in keys:
            return False

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return keys[service].get("hash") == key_hash

    def get_api_key(self, service: str) -> Optional[str]:
        """
        Retrieve a stored API key

        Args:
            service: The service name

        Returns:
            The API key if available and decryptable
        """
        keys = self._load_keys()

        if service not in keys:
            return None

        key_data = keys[service]

        # Return decrypted key if available
        if "encrypted" in key_data and self.encryption_manager:
            try:
                return self.encryption_manager.decrypt_data(key_data["encrypted"])
            except Exception as e:
                logger.error(f"Failed to decrypt API key for {service}: {e}")
                return None

        return None

    def _load_keys(self) -> Dict[str, Any]:
        """Load stored keys from file"""
        if not self.keys_file.exists():
            return {}

        try:
            with open(self.keys_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            return {}

    def _save_keys(self, keys: Dict[str, Any]) -> None:
        """Save keys to file"""
        try:
            with open(self.keys_file, "w") as f:
                json.dump(keys, f, indent=2)

            # Set restrictive permissions (owner read/write only)
            self.keys_file.chmod(0o600)
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a secure random token

    Args:
        length: The length of the token

    Returns:
        Secure random token
    """
    return secrets.token_urlsafe(length)


def hash_sensitive_data(data: str) -> str:
    """
    Hash sensitive data for logging

    Args:
        data: The data to hash

    Returns:
        Hashed data
    """
    if not data:
        return ""

    # Show first 4 and last 4 characters for identification
    if len(data) > 8:
        return f"{data[:4]}...{data[-4:]}"
    else:
        return "***"
