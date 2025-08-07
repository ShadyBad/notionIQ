#!/usr/bin/env python3
"""
NotionIQ Quickstart - Zero Configuration Required
Automatically configures and runs NotionIQ with optimal settings
"""

import os
import sys
from pathlib import Path


# Check for required API keys
def check_api_keys():
    """Check if at least one API key is configured"""
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))

    if not has_anthropic and not has_openai:
        print("❌ No API keys found!")
        print("\nPlease set one of the following environment variables:")
        print("  - ANTHROPIC_API_KEY (for Claude)")
        print("  - OPENAI_API_KEY (for GPT-4)")
        print("\nExample:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return False

    return True


def check_notion_config():
    """Check if Notion is configured"""
    has_notion_key = bool(os.getenv("NOTION_API_KEY"))
    has_database_id = bool(os.getenv("NOTION_INBOX_DATABASE_ID"))

    if not has_notion_key:
        print("❌ NOTION_API_KEY not found!")
        print("\nPlease set your Notion integration token:")
        print("  export NOTION_API_KEY='your-notion-token'")
        return False

    if not has_database_id:
        print("❌ NOTION_INBOX_DATABASE_ID not found!")
        print("\nPlease set your Notion database ID:")
        print("  export NOTION_INBOX_DATABASE_ID='your-database-id'")
        return False

    return True


def main():
    """Run NotionIQ with automatic configuration"""

    print("=" * 60)
    print("🚀 NotionIQ QUICKSTART")
    print("=" * 60)
    print()

    # Step 1: Check configuration
    print("📋 Checking configuration...")

    if not check_api_keys():
        sys.exit(1)

    if not check_notion_config():
        sys.exit(1)

    print("✅ Configuration valid!")
    print()

    # Step 2: Auto-configure API settings
    print("🔧 Auto-configuring API settings...")

    try:
        from api_auto_config import get_auto_config

        config = get_auto_config()

        print()
        print("✅ Configuration complete!")
        print(f"   Model: {config['model']}")
        print(f"   Mode: {config['optimization_level']}")
        print(f"   Est. Cost: ${config['estimated_cost_per_100']}/100 pages")
        print()

    except Exception as e:
        print(f"❌ Auto-configuration failed: {e}")
        print("Falling back to default settings...")
        print()

    # Step 3: Run NotionIQ
    print("🎯 Starting NotionIQ...")
    print("-" * 60)

    # Import and run
    try:
        import click

        from notion_organizer import main as run_organizer

        # Create context with default options (auto mode)
        ctx = click.Context(run_organizer)
        ctx.params = {
            "analyze_workspace": True,
            "process_inbox": True,
            "create_recommendations": True,
            "dry_run": False,
            "batch_size": None,
            "optimization": "auto",  # Use automatic configuration
        }

        # Run with context
        run_organizer.invoke(ctx)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("\n🌟 Welcome to NotionIQ - Intelligent Notion Organizer")
    print("   Zero configuration required - we'll handle everything!\n")

    main()
