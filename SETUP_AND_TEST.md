# NotionIQ Setup & Testing Guide

## 🚀 Quick Setup

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
```env
# Required fields:
NOTION_API_KEY=secret_abc123...  # Your Notion Integration Token
NOTION_INBOX_DATABASE_ID=abc123def456...  # Your Inbox Database ID
ANTHROPIC_API_KEY=sk-ant-api03-...  # Your Claude API Key
```

## 📋 Pre-Flight Checklist

### Getting Your Notion Credentials

1. **Create Notion Integration:**
   - Go to https://www.notion.so/my-integrations
   - Click "New integration"
   - Name it "NotionIQ"
   - Select your workspace
   - Copy the Internal Integration Token

2. **Get Database ID:**
   - Open your Notion database in browser
   - Copy the ID from URL: `notion.so/workspace/[DATABASE_ID]?v=...`
   - It's the part between the last `/` and the `?`

3. **Grant Access:**
   - In Notion, open your database
   - Click "..." menu → "Add connections"
   - Select your NotionIQ integration

### Getting Your Anthropic API Key

1. Go to https://console.anthropic.com/
2. Navigate to API Keys
3. Create a new key
4. Copy and save securely

## 🧪 Testing Steps

### Step 1: Test Configuration
```bash
python -c "from config import get_settings; s = get_settings(); print('✅ Config loaded!')"
```

Expected: Should load without errors if `.env` is configured.

### Step 2: Test Notion Connection
```bash
python -c "
from notion_client import NotionAdvancedClient
from config import get_settings
settings = get_settings()
client = NotionAdvancedClient(settings)
db = client.get_database(settings.notion_inbox_database_id)
print('✅ Connected to Notion!')
"
```

### Step 3: Test Individual Components
```bash
# Test each module individually
python notion_client.py
python claude_analyzer.py  # Will test with sample data
python workspace_analyzer.py
```

### Step 4: Dry Run Test
```bash
# Test with small batch, no changes (inbox only)
python notion_organizer.py --mode inbox --batch-size 1 --dry-run

# Test workspace mode with limit
python notion_organizer.py --mode workspace --batch-size 2 --dry-run
```

### Step 5: Full Test Run
```bash
# Process 5 pages per database (full workspace)
python notion_organizer.py --batch-size 5

# Process specific databases only
python notion_organizer.py --mode databases --target-databases "Inbox" --batch-size 10

# Process a page and its children
python notion_organizer.py --mode page --target-page "your-page-id"
```

## 🔍 Verification Steps

### Check Output Files
After running, verify these files are created:
- `output/analysis_report_[timestamp].json`
- `output/summary_[timestamp].txt`
- `data/notioniq.log`

### Review the Summary
```bash
# View the latest summary
ls -la output/summary_*.txt
cat output/summary_*.txt  # Read the latest one
```

### Check Logs for Errors
```bash
# Check for any errors
grep ERROR data/notioniq.log
grep WARNING data/notioniq.log
```

## 🐛 Common Issues & Solutions

### Issue: "Module not found" Error
```bash
# Solution: Ensure virtual environment is activated
which python  # Should show venv path
pip list  # Check installed packages
```

### Issue: "Invalid API Key"
```bash
# Solution: Verify .env file
cat .env | grep API_KEY  # Check keys are set (be careful not to expose them)
```

### Issue: "Database not found"
- Ensure the database ID is correct
- Verify the integration has access to the database
- Check if database is not archived

### Issue: Rate Limiting
```bash
# Solution: Adjust rate limit in .env
RATE_LIMIT_REQUESTS_PER_SECOND=1.0  # Slower but safer
```

## ✅ Success Indicators

You'll know everything is working when:
1. No Python errors during execution
2. Output files are created in `output/` directory
3. Console shows colorful progress bars
4. Health score is calculated (0-100)
5. Recommendations are generated
6. JSON report contains analyzed pages

## 📊 Understanding the Output

### Console Output
- Green checkmarks (✅) = Success
- Yellow warnings (⚠️) = Attention needed
- Red errors (❌) = Issues to fix

### Health Score Interpretation
- **90-100**: Excellent workspace organization
- **70-89**: Good, minor improvements possible
- **50-69**: Moderate, several improvements recommended
- **Below 50**: Significant reorganization beneficial

### Confidence Levels
- **High (>0.8)**: Very confident in classification
- **Medium (0.6-0.8)**: Reasonably confident
- **Low (<0.6)**: Manual review suggested

## 🎯 Next Steps

Once testing is successful:
1. Run on full inbox: `python notion_organizer.py`
2. Review recommendations in JSON report
3. Implement high-confidence suggestions
4. Iterate and refine

## 📝 Test Checklist

- [ ] Virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] .env file configured with valid credentials
- [ ] Notion integration has database access
- [ ] Config module loads successfully
- [ ] Notion connection test passes
- [ ] Dry run completes without errors
- [ ] Output files are generated
- [ ] No errors in log file
- [ ] At least one page analyzed successfully

## 💡 Pro Tips

1. **Start Small**: Always test with `--batch-size 1` first
2. **Use Dry Run**: Preview changes with `--dry-run`
3. **Check Logs**: Review `data/notioniq.log` for details
4. **Enable Caching**: Reduces API calls on subsequent runs
5. **Monitor Costs**: Check your Anthropic usage dashboard

---

If all tests pass, your NotionIQ system is ready for production use! 🎉