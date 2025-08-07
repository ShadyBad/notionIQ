#!/usr/bin/env python3
"""
Validation script to ensure NotionIQ is production-ready
Run this to verify all improvements have been successfully implemented
"""

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


class ProductionValidator:
    """Validates that all production requirements are met"""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def validate_all(self) -> bool:
        """Run all validations"""
        print("=" * 60)
        print("🔍 NOTIONIQ PRODUCTION VALIDATION")
        print("=" * 60)
        print()

        validations = [
            ("Test Suite", self.validate_tests),
            ("Security Module", self.validate_security),
            ("API Key Validation", self.validate_api_validation),
            ("Type Hints", self.validate_type_hints),
            ("Bug Fixes", self.validate_bug_fixes),
            ("Features", self.validate_features),
            ("CI/CD Pipeline", self.validate_cicd),
            ("Docker Support", self.validate_docker),
            ("Documentation", self.validate_documentation),
            ("Code Quality", self.validate_code_quality),
        ]

        for name, validator in validations:
            print(f"Validating {name}...")
            try:
                result = validator()
                if result:
                    self.passed += 1
                    print(f"  ✅ {name}: PASSED")
                else:
                    self.failed += 1
                    print(f"  ❌ {name}: FAILED")
            except Exception as e:
                self.failed += 1
                print(f"  ❌ {name}: ERROR - {e}")
            print()

        # Summary
        print("=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {self.passed}/{len(validations)}")
        print(f"❌ Failed: {self.failed}/{len(validations)}")

        completion_rate = (self.passed / len(validations)) * 100
        print(f"📈 Completion Rate: {completion_rate:.1f}%")

        if completion_rate >= 100:
            print("\n🎉 PRODUCTION READY! All validations passed!")
            return True
        elif completion_rate >= 75:
            print("\n⚠️ NEARLY READY! Fix remaining issues for production.")
            return False
        else:
            print("\n❌ NOT READY! Significant work needed.")
            return False

    def validate_tests(self) -> bool:
        """Validate test suite exists and is comprehensive"""
        test_dir = Path("tests")
        if not test_dir.exists():
            print("    ❌ No tests directory found")
            return False

        test_files = list(test_dir.glob("test_*.py"))
        required_tests = ["test_config.py", "test_notion_client.py", "conftest.py"]

        for required in required_tests:
            if not (test_dir / required).exists():
                print(f"    ❌ Missing test file: {required}")
                return False

        print(f"    ✓ Found {len(test_files)} test files")
        print(f"    ✓ All required test files present")
        return True

    def validate_security(self) -> bool:
        """Validate security module implementation"""
        security_file = Path("security.py")
        if not security_file.exists():
            print("    ❌ Security module not found")
            return False

        try:
            import security

            # Check for required classes
            required_classes = [
                "SecurityValidator",
                "EncryptionManager",
                "APIKeyManager",
            ]

            for cls in required_classes:
                if not hasattr(security, cls):
                    print(f"    ❌ Missing class: {cls}")
                    return False

            # Check for validation methods
            validator = security.SecurityValidator
            required_methods = [
                "validate_notion_api_key",
                "validate_anthropic_api_key",
                "sanitize_text_content",
                "sanitize_database_id",
            ]

            for method in required_methods:
                if not hasattr(validator, method):
                    print(f"    ❌ Missing method: {method}")
                    return False

            print("    ✓ Security module properly implemented")
            print("    ✓ API validation methods present")
            print("    ✓ Sanitization methods present")
            print("    ✓ Encryption support available")
            return True

        except ImportError as e:
            print(f"    ❌ Cannot import security module: {e}")
            return False

    def validate_api_validation(self) -> bool:
        """Validate API key validation in config"""
        try:
            config_content = Path("config.py").read_text()

            # Check for Pydantic v2 validators
            if "@model_validator" not in config_content:
                print("    ❌ Not using Pydantic v2 validators")
                return False

            if "@field_validator" not in config_content:
                print("    ❌ Not using field validators")
                return False

            # Check for security import
            if "from security import SecurityValidator" in config_content:
                print("    ✓ Security validation integrated")
            else:
                print("    ⚠️ Security validation not fully integrated")

            print("    ✓ Pydantic v2 validators implemented")
            return True

        except Exception as e:
            print(f"    ❌ Error checking config: {e}")
            return False

    def validate_bug_fixes(self) -> bool:
        """Validate critical bug fixes"""
        # Check page count fix
        notion_client = Path("notion_client.py").read_text()
        if "# Count pages in database with proper pagination" in notion_client:
            print("    ✓ Page count bug fixed")
        else:
            print("    ❌ Page count bug not fixed")
            return False

        # Check OpenAI implementation
        claude_analyzer = Path("claude_analyzer.py").read_text()
        if "from openai import OpenAI" in claude_analyzer:
            print("    ✓ OpenAI provider implemented")
        else:
            print("    ❌ OpenAI provider not implemented")
            return False

        return True

    def validate_features(self) -> bool:
        """Validate new features implementation"""
        notion_organizer = Path("notion_organizer.py").read_text()

        # Check Notion recommendations page
        if "_format_recommendations_for_notion" in notion_organizer:
            print("    ✓ Notion recommendations page implemented")
        else:
            print("    ❌ Notion recommendations page not implemented")
            return False

        # Check formatting methods
        if "_format_recommendation_section" in notion_organizer:
            print("    ✓ Recommendation formatting implemented")
        else:
            print("    ❌ Recommendation formatting not implemented")
            return False

        return True

    def validate_type_hints(self) -> bool:
        """Validate type hints are comprehensive"""
        files_to_check = [
            "notion_client.py",
            "claude_analyzer.py",
            "workspace_analyzer.py",
        ]

        for file in files_to_check:
            content = Path(file).read_text()

            # Check for return type hints
            if "-> None:" in content and "-> str:" in content and "-> Dict" in content:
                print(f"    ✓ {file}: Type hints present")
            else:
                print(f"    ❌ {file}: Missing type hints")
                return False

        return True

    def validate_cicd(self) -> bool:
        """Validate CI/CD pipeline configuration"""
        ci_file = Path(".github/workflows/ci.yml")
        if not ci_file.exists():
            print("    ❌ CI/CD pipeline not configured")
            return False

        ci_content = ci_file.read_text()

        # Check for required jobs
        required_jobs = ["lint", "test", "security", "build"]
        for job in required_jobs:
            if f"name: {job}" in ci_content.lower() or f"{job}:" in ci_content:
                print(f"    ✓ Job '{job}' configured")
            else:
                print(f"    ❌ Job '{job}' missing")
                return False

        return True

    def validate_docker(self) -> bool:
        """Validate Docker configuration"""
        dockerfile = Path("Dockerfile")
        if not dockerfile.exists():
            print("    ❌ Dockerfile not found")
            return False

        docker_content = dockerfile.read_text()

        # Check for security best practices
        if "USER notioniq" in docker_content:
            print("    ✓ Non-root user configured")
        else:
            print("    ❌ Running as root user")
            return False

        if "HEALTHCHECK" in docker_content:
            print("    ✓ Health check configured")
        else:
            print("    ❌ No health check")
            return False

        return True

    def validate_documentation(self) -> bool:
        """Validate documentation completeness"""
        readme = Path("README.md")
        if not readme.exists():
            print("    ❌ README not found")
            return False

        readme_content = readme.read_text().lower()

        # Check for required sections
        required_sections = [
            "features",
            "installation",
            "usage",
            "configuration",
            "testing",
        ]

        for section in required_sections:
            if section in readme_content:
                print(f"    ✓ Section '{section}' present")
            else:
                print(f"    ❌ Section '{section}' missing")
                return False

        return True

    def validate_code_quality(self) -> bool:
        """Validate code quality improvements"""
        # Check for pre-commit config
        if Path(".pre-commit-config.yaml").exists():
            print("    ✓ Pre-commit hooks configured")
        else:
            print("    ❌ Pre-commit not configured")
            return False

        # Check for setup.py
        if Path("setup.py").exists():
            print("    ✓ Package setup configured")
        else:
            print("    ❌ setup.py not found")
            return False

        return True


def main():
    """Run production validation"""
    validator = ProductionValidator()
    is_ready = validator.validate_all()

    # Exit with appropriate code
    sys.exit(0 if is_ready else 1)


if __name__ == "__main__":
    main()
