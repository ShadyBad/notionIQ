# Agent Usage Policy for NotionIQ

This file defines when Claude Code should automatically use specialized agents for this project.

## 🚨 MANDATORY AGENT USAGE

Claude Code MUST use these agents for the following scenarios:

### Development Tasks

| Task | Required Agent | When to Use |
|------|---------------|-------------|
| Writing new features | `fullstack-developer` or `backend-api-developer` | Any new functionality |
| Debugging errors | `debugger-specialist` | When errors occur or tests fail |
| Code review | `code-reviewer` | After writing >20 lines of code |
| Performance issues | `performance-engineer` | When optimizing or fixing slow code |
| Database work | `database-optimizer` | For any database-related tasks |
| API development | `api-designer` | When creating or modifying APIs |

### Infrastructure & DevOps

| Task | Required Agent | When to Use |
|------|---------------|-------------|
| CI/CD issues | `devops-engineer` | GitHub Actions or pipeline problems |
| Docker problems | `devops-engineer` | Dockerfile or container issues |
| Build failures | `build-engineer` | Package, bundling, or build errors |
| Deployment | `deployment-engineer` | Any deployment-related tasks |
| Infrastructure | `platform-engineer` | Platform or infrastructure changes |

### Quality & Documentation

| Task | Required Agent | When to Use |
|------|---------------|-------------|
| Writing tests | `test-automator` | Creating or modifying tests |
| Documentation | `documentation-engineer` | Updating any .md files |
| Security review | `security-engineer` | Security or compliance checks |
| Architecture review | `architect-reviewer` | Major architectural changes |

## 🤖 AUTOMATIC TRIGGERS

These agents should be launched automatically:

1. **After any code changes**
   - Automatically run `code-reviewer` agent
   - Example: "I've updated the API endpoints" → Launch code-reviewer

2. **When errors occur**
   - Automatically run `debugger-specialist` agent
   - Example: "Tests are failing" → Launch debugger-specialist

3. **For performance issues**
   - Automatically run `performance-engineer` agent
   - Example: "The app is running slowly" → Launch performance-engineer

4. **When updating documentation**
   - Automatically run `documentation-engineer` agent
   - Example: "Update the README" → Launch documentation-engineer

## 📋 WORKFLOW EXAMPLES

### Example 1: Adding a New Feature
```
1. User: "Add a new export feature"
2. Claude: Launch `fullstack-developer` agent to implement
3. Claude: Launch `test-automator` agent to add tests
4. Claude: Launch `code-reviewer` agent to review
5. Claude: Launch `documentation-engineer` to update docs
```

### Example 2: Fixing CI/CD Issues
```
1. User: "The GitHub Actions are failing"
2. Claude: Launch `devops-engineer` agent to diagnose
3. Claude: Launch `debugger-specialist` if needed
4. Claude: Launch `test-automator` to fix tests
```

### Example 3: Performance Optimization
```
1. User: "The workspace scan is too slow"
2. Claude: Launch `performance-engineer` agent
3. Claude: Launch `database-optimizer` if DB-related
4. Claude: Launch `code-reviewer` after changes
```

## ⚙️ CONFIGURATION

Add this to your initial prompt or conversation:
```
"Always use specialized agents as defined in AGENTS.md for this project"
```

## 📊 AGENT PERFORMANCE METRICS

Track agent usage to ensure compliance:
- Minimum 80% of applicable tasks should use agents
- Code changes >20 lines must use code-reviewer
- All debugging must use debugger-specialist
- All CI/CD fixes must use devops-engineer

## 🎯 BENEFITS OF AGENT USAGE

1. **Higher Quality**: Specialized expertise for each domain
2. **Faster Resolution**: Agents are optimized for specific tasks
3. **Better Practices**: Agents follow industry best practices
4. **Comprehensive**: Agents consider aspects you might miss
5. **Learning**: Agents can identify patterns and improvements

---

**Remember**: When in doubt, use an agent! It's better to over-use agents than to miss important optimizations or best practices.