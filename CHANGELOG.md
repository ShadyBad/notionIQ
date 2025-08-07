# Changelog

All notable changes to NotionIQ will be documented in this file.

## [2.0.0] - 2025-08-07

### Added
- **Full Workspace Scanning**: Now analyzes entire Notion workspace by default, not just inbox
- **Multiple Processing Modes**:
  - `workspace` mode (default) - Processes all databases and pages
  - `inbox` mode - Focuses on inbox database only  
  - `page` mode - Analyzes specific page and all children recursively
  - `databases` mode - Targets specific databases
- **Directory Processing**: Process a page and all its children/grandchildren with `--mode page`
- **Selective Database Processing**: Skip or target specific databases
- **Workspace Scanner**: New `workspace_scanner.py` for comprehensive workspace analysis
- **Enhanced CLI Options**:
  - `--mode` to select processing mode
  - `--target-databases` to specify databases to process
  - `--skip-databases` to exclude databases
  - `--target-page` for page hierarchy processing

### Changed
- Default behavior now scans entire workspace instead of just inbox
- Improved error handling for large workspaces
- Better progress reporting during database processing
- Fixed async/await issues in workspace scanning

### Fixed
- Notion recommendations page creation now uses proper parent ID
- API optimizer attribute error in workspace mode
- Workspace structure caching improvements

## [1.5.0] - 2025-08-06

### Added
- Multi-provider AI support (Claude, ChatGPT, Gemini)
- Automatic provider detection and selection
- Smart API optimization with multiple levels
- Cost monitoring and budget management
- Enhanced caching system for API cost reduction

### Changed
- Improved error recovery mechanisms
- Better security validation for API keys
- Enhanced logging with structured output

## [1.0.0] - 2025-08-01

### Initial Release
- Core NotionIQ functionality
- Inbox processing
- AI-powered content classification
- Workspace health metrics
- Recommendations generation
- JSON report generation