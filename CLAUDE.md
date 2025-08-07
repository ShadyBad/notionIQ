# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NotionIQ** - An intelligent Notion workspace organizer powered by Claude AI. This system analyzes Notion workspaces, classifies content, and provides actionable recommendations for better organization.

## Repository Status

- **Project**: NotionIQ (codename "anchovy")
- **Stack**: Python 3.11+, Notion API, Anthropic Claude API
- **Status**: Core functionality implemented, ready for testing
- **License**: MIT License

## Architecture

### Core Components

1. **`config.py`** - Configuration management with Pydantic
2. **`notion_client.py`** - Enhanced Notion API wrapper with caching and rate limiting
3. **`claude_analyzer.py`** - AI-powered content classification and analysis
4. **`workspace_analyzer.py`** - Deep workspace structure analysis and health metrics
5. **`notion_organizer.py`** - Main orchestrator that coordinates all components

### Key Features

- Deep workspace scanning and relationship mapping
- AI-powered content classification with confidence scoring
- Multi-dimensional analysis (urgency, context, completeness)
- Workspace health metrics and scoring
- Actionable recommendations with reasoning
- Beautiful CLI output with Rich library
- Comprehensive JSON reports

## Development Commands

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application
```bash
# Full analysis (recommended first run)
python notion_organizer.py

# Dry run (no changes to Notion)
python notion_organizer.py --dry-run

# Process limited batch
python notion_organizer.py --batch-size 10

# Skip workspace analysis (faster)
python notion_organizer.py --skip-workspace

# Help
python notion_organizer.py --help
```

### Testing Individual Components
```bash
# Test Notion connection
python notion_client.py

# Test AI analyzer
python claude_analyzer.py

# Test workspace analyzer
python workspace_analyzer.py
```

## Configuration

Key environment variables in `.env`:
- `NOTION_API_KEY` - Notion integration token
- `NOTION_INBOX_DATABASE_ID` - ID of the Inbox database
- `ANTHROPIC_API_KEY` - Claude API key
- `BATCH_SIZE` - Number of pages to process
- `ENABLE_WORKSPACE_SCAN` - Enable deep workspace analysis
- `ENABLE_RECOMMENDATIONS_PAGE` - Create Notion recommendations page

## Output

- **JSON Reports**: Saved to `output/` directory with timestamps
- **Summary Files**: Text summaries for quick review
- **Logs**: Detailed logs in `data/notioniq.log`
- **Cache**: AI responses cached in `data/` for efficiency

## Next Development Steps

1. **Notion Recommendations Page**: Implement creation/update of recommendations page in Notion
2. **Auto-Organization**: Add capability to automatically move pages based on recommendations
3. **Web Interface**: Build web dashboard for configuration and monitoring
4. **Scheduling**: Add cron/scheduled execution support
5. **Multi-Workspace**: Support for analyzing multiple workspaces
6. **Export Formats**: Add PDF and HTML report generation
7. **Template Marketplace**: System for sharing workspace templates

## Common Issues & Solutions

- **Rate Limiting**: Adjust `RATE_LIMIT_REQUESTS_PER_SECOND` in config
- **Large Workspaces**: Use `--batch-size` to process in chunks
- **API Costs**: Enable caching with `ENABLE_CACHING=true`
- **Timeout Issues**: Increase timeout values in API calls

## Testing Approach

For production readiness:
1. Test with small batch first (`--batch-size 5`)
2. Verify classifications manually
3. Run with `--dry-run` to preview actions
4. Monitor logs for errors
5. Gradually increase batch size