"""Setup configuration for NotionIQ"""

from setuptools import find_packages, setup
from pathlib import Path

# Read the README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="notioniq",
    version="2.0.0",
    author="NotionIQ Team",
    description="Intelligent Notion Workspace Organizer powered by Claude AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShadyBad/notionIQ",
    packages=find_packages(exclude=["tests", "tests.*"]),
    # Flat layout: source modules live at the repo root, not in a package
    # directory, so they must be declared explicitly for an editable install
    # to expose them (otherwise `import notion_organizer` fails).
    py_modules=[
        "advanced_optimizer",
        "ai_analyzer",
        "ai_providers",
        "api_auto_config",
        "api_optimizer",
        "config",
        "cost_monitor",
        "logger_wrapper",
        "notion_organizer",
        "notion_wrapper",
        "quickstart",
        "recommendation_executor",
        "security",
        "workspace_analyzer",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Office/Business",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "notioniq=notion_organizer:main",
        ],
    },
)