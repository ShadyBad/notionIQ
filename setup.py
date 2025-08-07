"""
Setup configuration for NotionIQ
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

setup(
    name="notioniq",
    version="1.0.0",
    author="NotionIQ Team",
    author_email="contact@notioniq.dev",
    description="Intelligent Notion workspace organizer powered by AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/notioniq",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Office/Business",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.1.1",
            "pytest-asyncio>=0.23.5",
            "pytest-cov>=5.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
            "mypy>=1.9.0",
            "pre-commit>=3.6.0",
        ],
        "openai": [
            "openai>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "notioniq=notion_organizer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
)
