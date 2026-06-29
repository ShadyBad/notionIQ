# NotionIQ Overhaul — Design Spec

- **Date:** 2026-06-28
- **Status:** approved
- **Risk tier:** HIGH (multi-file; model/cost/behavior changes; reversible — dry-run exists, no data migration)
- **Owner:** Brandon

## Problem

NotionIQ classifies Notion pages via Claude. It bleeds cost, carries dead weight, and the run experience is noisy. Concretely (from codebase map):

- Default model is `claude-3-opus-20240229` — stale and ~15× pricier than the task needs. Classification is Haiku-class work.
- No Anthropic prompt caching, despite a large static system/taxonomy prompt re-sent on every page call.
- Pricing constants are stale Opus-3 rates duplicated in 3 places; reported cost is wrong even when a cheaper model is selected.
- `SmartCache` returns *another page's* analysis on ≥0.85 Jaccard similarity — a correctness bug that ships wrong classifications to save pennies.
- Caches key on content hash only (not Notion `last_edited_time`), so re-runs re-fetch and can re-pay.
- ~2,170 lines of dead modules (`claude_analyzer.py`, `error_recovery.py`, `performance_enhancer.py`, `workspace_scanner.py`) imported by nothing.
- Workspace scan double-fetches every page (full payload) only to count it.
- No `console_scripts` entry — users must run `python notion_organizer.py`.
- Six-step narrative output is noisy, not tasteful.

## Hypothesis

Modern model + prompt caching + incremental skip cuts per-page cost ≥90% with classification quality held. Removing the similarity-cache fixes a real correctness bug. Cleanup and a one-command CLI make it pleasant to use.

## Scope (12 items, 3 slices)

### Slice 1 — Lean Engine (cost → near-zero)
1. **Modern models + routing.** Default `claude-haiku-4-5`. Update `ai_providers.py` MODEL_INFO to current Haiku 4.5 / Sonnet 4.6 / Opus 4.8 with correct pricing (source: claude-api skill). Routing logic preserved; `--optimization full` → Sonnet/Opus.
2. **Prompt caching.** `cache_control: ephemeral` on the static system/taxonomy block (`ai_analyzer.py:253`). Bump `anthropic` SDK from 0.21.3 to current.
3. **Single correct pricing source.** Remove the 3 duplicated stale Opus-rate constants; `calculate_cost` uses the actually-selected model.
4. **Kill similarity cache.** Remove the ≥0.85-Jaccard return path (`api_optimizer.py:269`). Keep exact content-hash cache.
5. **Incremental skip.** Cache key includes Notion `last_edited_time`; skip pages unchanged since last run.

### Slice 2 — Tidy (efficiency)
6. **Delete dead modules:** `claude_analyzer.py`, `error_recovery.py`, `performance_enhancer.py`, `workspace_scanner.py`.
7. **Consolidate caches** — three layers to one persistent content+version cache.
8. **No double-fetch** — scan fetches each page once, reused for analysis (`notion_wrapper.py:324`).
9. **`notioniq` command** — `console_scripts` entry; bump Notion API version.

### Slice 3 — Taste (UX)
10. **Elegant summary** — one composed final Rich panel (pages, classifications, cost, cache-hit %, time) replacing the 6-step narrative.
11. **`notioniq init` wizard** — interactive setup: prompt keys, validate, write `.env`. Replaces `quickstart.py`.
12. **Smart defaults + markdown export** — no-args run Just Works; `--format markdown` beside JSON.

## Non-goals (deferred)

- **Batch API.** 50% off but 24h async turnaround breaks run-and-see UX; caching + Haiku already reach near-zero.
- Web dashboard, scheduling, multi-workspace, template marketplace.
- Auto-organization / auto-execute behavior changes.

## Constraints

- Preserve existing behavior (classify + recommend), CLI options, dry-run.
- No new required env vars without defaults.
- Mock API calls in tests — no live spend in CI.

## Risks (adversarial — ways this could be wrong)

- **Haiku quality regression.** Haiku-4.5 may classify worse than Opus-3 on ambiguous pages. Mitigation: hold a sample run for quality comparison before declaring success; routing still allows `--optimization full`.
- **Prompt caching needs prompt restructure.** `cache_control` only pays off if the static block is large, contiguous, and ordered first. May need to reorder the prompt; if static portion is small, savings shrink — Haiku switch still dominates.
- **SDK bump breaks call signatures.** anthropic 0.21.3 → current is a major jump; `messages.create` params and response shape may differ. Mitigation: pin a known-good current version, smoke-test the call path.
- **Removing similarity cache raises cost** for near-duplicate-heavy workspaces. Accepted — correctness over pennies; incremental skip recovers most of it.
- **`last_edited_time` skip can mask real reclassification needs** if taxonomy changes between runs. Mitigation: cache key also includes a prompt/taxonomy version hash so a prompt change invalidates the cache.

## Success criteria

1. Per-page cost ≥90% lower on a sample run (measured by the cost panel, with corrected pricing).
2. Classification quality held on that sample (manual spot check, no obvious regressions).
3. `notioniq` one-command run works after `pip install -e .`.
4. Tests green; new unit tests for pricing calc, cache key (incl. `last_edited_time` + taxonomy version), model routing; CLI-entry smoke test.

## Plan sketch

Slice 1 first (verifies the cost thesis), then Slice 2 cleanup, then Slice 3 UX. Each slice independently testable. Single ship.
