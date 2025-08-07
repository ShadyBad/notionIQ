# Contributing to NotionIQ

Thank you for your interest in contributing to NotionIQ! We welcome contributions from the community and are grateful for any help you can provide.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## 📜 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be respectful** - Treat everyone with respect and kindness
- **Be inclusive** - Welcome diverse perspectives and experiences
- **Be collaborative** - Work together to solve problems
- **Be professional** - Keep discussions focused and constructive

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/anchovy.git
   cd anchovy
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/anchovy.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 💻 Development Setup

### Prerequisites
- Python 3.11+ (3.12 recommended)
- Git
- Virtual environment tool (venv, virtualenv, or conda)

### Installation

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your test API keys
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v
```

## 🤝 How to Contribute

### Types of Contributions

#### 🐛 Bug Fixes
- Search existing issues to avoid duplicates
- Create a new issue if needed
- Reference the issue in your PR

#### ✨ New Features
- Discuss the feature in an issue first
- Get approval before implementing large changes
- Follow the existing architecture patterns

#### 📚 Documentation
- Fix typos and clarify unclear sections
- Add examples and use cases
- Update README for new features

#### 🧪 Tests
- Add tests for new features
- Improve test coverage
- Fix failing tests

#### 🎨 Code Quality
- Refactor for clarity
- Optimize performance
- Reduce technical debt

## 📏 Code Standards

### Python Style Guide

We follow PEP 8 with these additions:

```python
# Good: Descriptive names
def calculate_workspace_health_score(workspace_data: Dict[str, Any]) -> float:
    """Calculate health score for workspace organization."""
    pass

# Bad: Unclear names
def calc_score(data):
    pass
```

### Type Hints

Always use type hints:

```python
from typing import Dict, List, Optional, Any

def analyze_page(
    page_content: Dict[str, Any],
    workspace_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Analyze a page with optional context."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def process_batch(
    items: List[Dict[str, Any]],
    batch_size: int = 50
) -> List[Dict[str, Any]]:
    """Process items in batches.
    
    Args:
        items: List of items to process
        batch_size: Number of items per batch
        
    Returns:
        List of processed items
        
    Raises:
        ValueError: If batch_size is less than 1
    """
    pass
```

### Error Handling

Always handle errors gracefully:

```python
try:
    result = api_call()
except SpecificError as e:
    logger.error(f"API call failed: {e}")
    # Handle gracefully
    return default_value
```

### Logging

Use the logger wrapper:

```python
from logger_wrapper import logger

logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
```

## 🧪 Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestFeatureName:
    """Test suite for FeatureName."""
    
    def test_happy_path(self):
        """Test normal operation."""
        assert feature_function(valid_input) == expected_output
    
    def test_edge_case(self):
        """Test edge cases."""
        with pytest.raises(ValueError):
            feature_function(invalid_input)
    
    @patch('module.external_api')
    def test_with_mock(self, mock_api):
        """Test with mocked dependencies."""
        mock_api.return_value = {"status": "success"}
        result = feature_function()
        assert result["status"] == "success"
```

### Test Coverage

- Aim for 80%+ coverage
- Test all public methods
- Include edge cases
- Test error conditions

## 🔄 Pull Request Process

### Before Submitting

1. **Update from upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:
   ```bash
   pytest
   ```

3. **Check code quality**:
   ```bash
   # Format code
   black .
   
   # Check types
   mypy .
   
   # Lint
   ruff check .
   ```

4. **Update documentation** if needed

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance improvement

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings

## Related Issues
Fixes #123
```

### Review Process

1. **Automated checks** run on PR creation
2. **Code review** by maintainers
3. **Address feedback** promptly
4. **Merge** when approved

## 🐛 Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/logs
- Minimal reproducible example

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches
- Impact on existing features

## 📝 Commit Messages

Follow conventional commits:

```bash
# Format
<type>(<scope>): <subject>

# Examples
feat(analyzer): add support for custom classifiers
fix(api): handle rate limit errors gracefully
docs(readme): update installation instructions
perf(cache): optimize cache lookup performance
test(config): add tests for validation logic
refactor(client): simplify HTTP client logic
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `perf`: Performance improvement
- `test`: Testing
- `refactor`: Code refactoring
- `style`: Formatting
- `chore`: Maintenance

## 🏗️ Project Structure

```
anchovy/
├── notion_organizer.py      # Main entry point
├── config.py                # Configuration management
├── notion_client.py         # Notion API wrapper
├── claude_analyzer.py       # AI analysis engine
├── workspace_analyzer.py    # Workspace analysis
├── api_optimizer.py         # API cost optimization
├── security.py             # Security utilities
├── error_recovery.py       # Error handling (new)
├── performance_enhancer.py # Performance optimization (new)
├── tests/                  # Test suite
│   ├── conftest.py        # Test fixtures
│   ├── test_*.py          # Test modules
│   └── integration/       # Integration tests
├── docs/                   # Documentation
└── output/                # Generated reports
```

## 🎯 Development Priorities

Current focus areas:
1. **Error resilience** - Circuit breakers, retries
2. **Performance** - Async processing, caching
3. **Testing** - Increase coverage to 90%
4. **Documentation** - API docs, tutorials
5. **Security** - Enhanced validation, audit logs

## 📚 Resources

- [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)
- [Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

## 🙏 Thank You!

Your contributions make NotionIQ better for everyone. We appreciate your time and effort!

Questions? Open an issue or reach out to the maintainers.

---

**Happy coding!** 🚀