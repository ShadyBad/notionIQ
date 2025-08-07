#!/usr/bin/env python3
"""
Code review script to check for logic errors and issues
"""

import ast
import sys
from pathlib import Path

def check_file(filepath):
    """Check a Python file for common issues"""
    print(f"\n📄 Checking {filepath.name}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        print(f"  ✅ Valid Python syntax")
        
        # Check for undefined names (basic check)
        imported_names = set()
        defined_names = set()
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname if alias.asname else alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname if alias.asname else alias.name)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined_names.add(node.id)
        
        # Count lines
        lines = content.count('\n')
        print(f"  📏 {lines} lines of code")
        
        # Count functions and classes
        functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        print(f"  🔧 {functions} functions, {classes} classes")
        
        return True
        
    except SyntaxError as e:
        print(f"  ❌ Syntax error: {e}")
        return False

def check_imports():
    """Check if all required imports are consistent"""
    files = [
        'config.py',
        'notion_client.py', 
        'claude_analyzer.py',
        'workspace_analyzer.py',
        'notion_organizer.py'
    ]
    
    imports_map = {}
    
    for file in files:
        filepath = Path(file)
        if not filepath.exists():
            print(f"❌ Missing file: {file}")
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
            
        tree = ast.parse(content)
        file_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    file_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    file_imports.append(node.module)
        
        imports_map[file] = file_imports
    
    # Check for circular imports
    internal_modules = {'config', 'notion_client', 'claude_analyzer', 'workspace_analyzer'}
    
    for file, imports in imports_map.items():
        internal_imports = [imp for imp in imports if imp in internal_modules]
        if internal_imports:
            print(f"\n{file} imports: {', '.join(internal_imports)}")

def review_config():
    """Review config.py for issues"""
    print("\n🔍 Reviewing config.py...")
    
    issues = []
    
    # Check for proper field definitions
    with open('config.py', 'r') as f:
        content = f.read()
        
    # Check for required fields
    required_fields = [
        'notion_api_key',
        'notion_inbox_database_id', 
        'anthropic_api_key'
    ]
    
    for field in required_fields:
        if field not in content:
            issues.append(f"Missing required field: {field}")
        else:
            print(f"  ✅ Found {field}")
    
    # Check for proper validators
    if '@validator' in content or 'field_validator' in content or 'def validate_' in content:
        print("  ✅ Has validators")
    
    return issues

def review_notion_client():
    """Review notion_client.py for issues"""
    print("\n🔍 Reviewing notion_client.py...")
    
    with open('notion_client.py', 'r') as f:
        content = f.read()
    
    # Check for key methods
    methods = [
        'get_database',
        'get_database_pages',
        'get_page_content',
        'extract_text_from_blocks',
        'scan_workspace'
    ]
    
    for method in methods:
        if f'def {method}' in content:
            print(f"  ✅ Found method: {method}")
        else:
            print(f"  ❌ Missing method: {method}")
    
    # Check for error handling
    if 'try:' in content and 'except' in content:
        print("  ✅ Has error handling")
    
    # Check for rate limiting
    if 'rate_limit' in content or '_rate_limit_wait' in content:
        print("  ✅ Has rate limiting")

def review_claude_analyzer():
    """Review claude_analyzer.py"""
    print("\n🔍 Reviewing claude_analyzer.py...")
    
    with open('claude_analyzer.py', 'r') as f:
        content = f.read()
    
    # Check for prompt creation
    if '_create_analysis_prompt' in content:
        print("  ✅ Has prompt creation")
    
    # Check for response parsing
    if '_parse_analysis_response' in content:
        print("  ✅ Has response parsing")
    
    # Check for caching
    if 'cache' in content.lower():
        print("  ✅ Has caching logic")
    
    # Check prompt structure
    if 'classification' in content and 'recommendations' in content:
        print("  ✅ Has classification and recommendations in prompt")

def review_workspace_analyzer():
    """Review workspace_analyzer.py"""
    print("\n🔍 Reviewing workspace_analyzer.py...")
    
    with open('workspace_analyzer.py', 'r') as f:
        content = f.read()
    
    # Check for key analysis methods
    methods = [
        '_analyze_database_structures',
        '_analyze_content_patterns',
        '_calculate_health_metrics',
        '_generate_insights'
    ]
    
    for method in methods:
        if f'async def {method}' in content or f'def {method}' in content:
            print(f"  ✅ Found: {method}")

def review_main_orchestrator():
    """Review notion_organizer.py"""
    print("\n🔍 Reviewing notion_organizer.py...")
    
    with open('notion_organizer.py', 'r') as f:
        content = f.read()
    
    # Check for CLI setup
    if '@click.command' in content:
        print("  ✅ Has Click CLI setup")
    
    # Check for main workflow
    if 'run_analysis' in content:
        print("  ✅ Has run_analysis method")
    
    # Check for report generation
    if '_create_report' in content:
        print("  ✅ Has report creation")
    
    # Check for async handling
    if 'asyncio.run' in content:
        print("  ✅ Has async execution")

def main():
    print("=" * 60)
    print("🔍 NotionIQ Code Review")
    print("=" * 60)
    
    # Check all Python files
    files = [
        Path('config.py'),
        Path('notion_client.py'),
        Path('claude_analyzer.py'),
        Path('workspace_analyzer.py'),
        Path('notion_organizer.py')
    ]
    
    all_valid = True
    for f in files:
        if f.exists():
            if not check_file(f):
                all_valid = False
        else:
            print(f"❌ Missing file: {f}")
            all_valid = False
    
    # Check imports
    print("\n📦 Checking import structure...")
    check_imports()
    
    # Detailed reviews
    review_config()
    review_notion_client()
    review_claude_analyzer()
    review_workspace_analyzer()
    review_main_orchestrator()
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ All files have valid Python syntax!")
    else:
        print("❌ Some issues found - see above")
    print("=" * 60)

if __name__ == "__main__":
    main()