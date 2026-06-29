# NotionIQ Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut NotionIQ per-page Claude cost ≥90%, remove a correctness bug and ~2,170 lines of dead code, and make the tool one-command, beautiful, and easy.

**Architecture:** NotionIQ is a flat-layout Python CLI (`click`) that scans a Notion workspace, classifies each page via the Anthropic SDK (`ai_analyzer.py` is the live path), and emits JSON + an optional Notion recommendations page. Cost is driven by one Claude call per page with no caching on a stale, expensive model. We modernize the model + pricing, add prompt caching, kill an unsafe similarity cache, delete dead modules, then polish the UX.

**Tech Stack:** Python 3.11+, `anthropic` SDK, `click`, `rich`, `pydantic`/`pydantic-settings`, `pytest`, `notion-client`.

## Global Constraints

- Default Claude model is `claude-haiku-4-5` (classification is Haiku-class work). Routing may escalate to `claude-sonnet-4-6` / `claude-opus-4-8`.
- Current pricing per 1M tokens (source: claude-api skill, cached 2026-06-04): Haiku 4.5 = `$1.00` in / `$5.00` out; Sonnet 4.6 = `$3.00` / `$15.00`; Opus 4.8 = `$5.00` / `$25.00`.
- **Model ID strings are exact — never append date suffixes.** Use `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-8` verbatim.
- `effort` / `output_config` params **error on Haiku 4.5** — do not send them. `temperature` is still accepted on Haiku 4.5 — keep `temperature=0.3`.
- Anthropic prompt-cache minimum prefix on Haiku 4.5 is **4096 tokens**; below that, `cache_control` silently no-ops (no error). The model swap is the must-win; caching is bonus.
- Tests mock the Anthropic client — **no live API spend in CI.**
- Preserve existing behavior (classify + recommend), all CLI options, and `--dry-run`.
- Run `uv run pytest` (or `python -m pytest`) and `uv run ruff check .` after each task; both must stay green.

## File Structure

| File | Change |
|---|---|
| `config.py` | `AIModel` enum → modern IDs; `claude_model` default → Haiku; add `prompt_version` setting |
| `ai_providers.py` | `AIModel` enum + `MODEL_INFO` table → modern models/pricing; routing default → Haiku |
| `api_optimizer.py` | `TokenOptimizer` pricing → per-model injected rates; **delete** `SmartCache` (Task 9); remove similarity path |
| `ai_analyzer.py` | Claude call uses `system` + `cache_control` + real `usage`; cache key gains `last_edited_time` + `prompt_version`; drop smart-cache calls |
| `notion_wrapper.py` | Scan stops double-fetching (Task 10); bump Notion API version |
| `requirements.txt` | Bump `anthropic` (Task 7) |
| `setup.py` | `console_scripts` entry `notioniq` (Task 11) |
| `notion_organizer.py` | Elegant summary panel; smart defaults; `--format markdown` (Tasks 12, 14) |
| `quickstart.py` → `notioniq init` | Interactive setup wizard wired to CLI (Task 13) |
| `claude_analyzer.py`, `error_recovery.py`, `performance_enhancer.py`, `workspace_scanner.py` | **Delete** (Task 8) |
| `tests/test_models.py`, `tests/test_pricing.py`, `tests/test_cache_key.py`, `tests/test_routing.py` | New |

---

## SLICE 1 — Lean Engine (cost → near-zero)

### Task 1: Modern model IDs, registry, and pricing

**Files:**
- Modify: `config.py:34-42` (`AIModel` enum), `config.py:68-70` (`claude_model` default)
- Modify: `ai_providers.py:24-40` (`AIModel` enum), `ai_providers.py:62-92` (`MODEL_INFO` Claude rows), `ai_providers.py:180-211` region (`available_models` lists for Claude)
- Test: `tests/test_models.py`, `tests/test_pricing.py`

**Interfaces:**
- Produces: `config.AIModel.CLAUDE_HAIKU == "claude-haiku-4-5"`, `.CLAUDE_SONNET == "claude-sonnet-4-6"`, `.CLAUDE_OPUS == "claude-opus-4-8"`; `ai_providers.AIModel.CLAUDE_HAIKU/SONNET/OPUS` members keyed in `MODEL_INFO` with current pricing.

- [ ] **Step 1: Write the failing test** — `tests/test_models.py`

```python
from config import AIModel as CfgModel
from ai_providers import AIModel as ProvModel, AIProviderManager


def test_config_model_ids_are_current():
    assert CfgModel.CLAUDE_HAIKU.value == "claude-haiku-4-5"
    assert CfgModel.CLAUDE_SONNET.value == "claude-sonnet-4-6"
    assert CfgModel.CLAUDE_OPUS.value == "claude-opus-4-8"
    # No retired claude-3 IDs remain
    assert all("claude-3" not in m.value for m in CfgModel if m.value.startswith("claude"))


def test_provider_registry_has_current_claude_models():
    info = AIProviderManager.MODEL_INFO
    assert info[ProvModel.CLAUDE_HAIKU].name == "Claude Haiku 4.5"
    assert ProvModel.CLAUDE_HAIKU.value[0] == "claude-haiku-4-5"
```

- [ ] **Step 2: Run it, confirm FAIL** — `python -m pytest tests/test_models.py -v` → fails (`AttributeError: CLAUDE_HAIKU`).

- [ ] **Step 3: Update `config.py`** — replace the Claude members in `AIModel` (lines 37-39) and the default (line 68-70):

```python
class AIModel(str, Enum):
    """Supported AI models"""

    CLAUDE_HAIKU = "claude-haiku-4-5"
    CLAUDE_SONNET = "claude-sonnet-4-6"
    CLAUDE_OPUS = "claude-opus-4-8"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
```

```python
    claude_model: AIModel = Field(
        AIModel.CLAUDE_HAIKU, description="Claude model to use (Haiku is the cost-optimal default)"
    )
```

- [ ] **Step 4: Update `ai_providers.py` enum** (lines 27-30):

```python
    # Claude models
    CLAUDE_OPUS = ("claude-opus-4-8", AIProvider.CLAUDE)
    CLAUDE_SONNET = ("claude-sonnet-4-6", AIProvider.CLAUDE)
    CLAUDE_HAIKU = ("claude-haiku-4-5", AIProvider.CLAUDE)
```

- [ ] **Step 5: Update `ai_providers.py` `MODEL_INFO`** Claude rows (lines 63-92) to:

```python
        AIModel.CLAUDE_OPUS: ModelInfo(
            name="Claude Opus 4.8",
            provider=AIProvider.CLAUDE,
            cost_per_1m_input=5.00,
            cost_per_1m_output=25.00,
            quality_score=10,
            speed_score=6,
            context_window=1000000,
            supports_json=True,
        ),
        AIModel.CLAUDE_SONNET: ModelInfo(
            name="Claude Sonnet 4.6",
            provider=AIProvider.CLAUDE,
            cost_per_1m_input=3.00,
            cost_per_1m_output=15.00,
            quality_score=9,
            speed_score=8,
            context_window=1000000,
            supports_json=True,
        ),
        AIModel.CLAUDE_HAIKU: ModelInfo(
            name="Claude Haiku 4.5",
            provider=AIProvider.CLAUDE,
            cost_per_1m_input=1.00,
            cost_per_1m_output=5.00,
            quality_score=8,
            speed_score=10,
            context_window=200000,
            supports_json=True,
        ),
```

- [ ] **Step 6: Fix remaining `CLAUDE_3_*` references** — `grep -rn "CLAUDE_3_" .` (ignore deleted modules). Update `available_models` lists (~line 185) and `_get_preferred_claude_model` (lines 216-221) to the new member names. Leave `_get_preferred_claude_model` default returning `AIModel.CLAUDE_HAIKU` (changed in Task 2).

- [ ] **Step 7: Update `.env.example`** — `CLAUDE_MODEL=claude-haiku-4-5`.

- [ ] **Step 8: Run tests + lint** — `python -m pytest tests/test_models.py tests/test_config.py -v && ruff check .` → PASS.

- [ ] **Step 9: Commit** — `feat(models): modernize Claude models + pricing to Haiku 4.5 / Sonnet 4.6 / Opus 4.8`

---

### Task 2: Default routing to Haiku, guard effort param

**Files:**
- Modify: `ai_providers.py:213-221` (`_get_preferred_claude_model` default)
- Test: `tests/test_routing.py`

**Interfaces:**
- Consumes: `AIProviderManager` from Task 1.
- Produces: with no `CLAUDE_MODEL` hint and no priority flags, selected Claude model is `claude-haiku-4-5`.

- [ ] **Step 1: Write the failing test** — `tests/test_routing.py`

```python
import os
from ai_providers import AIModel, AIProviderManager


def test_preferred_claude_model_defaults_to_haiku(monkeypatch):
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    mgr = AIProviderManager.__new__(AIProviderManager)  # no __init__ side effects
    assert mgr._get_preferred_claude_model() == AIModel.CLAUDE_HAIKU


def test_sonnet_hint_routes_to_sonnet(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "sonnet")
    mgr = AIProviderManager.__new__(AIProviderManager)
    assert mgr._get_preferred_claude_model() == AIModel.CLAUDE_SONNET
```

- [ ] **Step 2: Run it, confirm FAIL** — `python -m pytest tests/test_routing.py -v` (default still returns OPUS).

- [ ] **Step 3: Update default** (`ai_providers.py:216-221`):

```python
        model_hint = os.getenv("CLAUDE_MODEL", "").lower()
        if "opus" in model_hint:
            return AIModel.CLAUDE_OPUS
        elif "sonnet" in model_hint:
            return AIModel.CLAUDE_SONNET
        else:
            return AIModel.CLAUDE_HAIKU  # cost-optimal default
```

- [ ] **Step 4: Confirm the Claude call sends no `effort`/`output_config`** — read `ai_analyzer.py:250-266`. The call already only sets `model`, `max_tokens`, `temperature`, `messages`. Leave as-is (Haiku 4.5 rejects `effort`). No change needed beyond a code comment: `# Haiku 4.5 rejects effort/output_config; temperature is supported`.

- [ ] **Step 5: Run tests** — `python -m pytest tests/test_routing.py -v` → PASS.

- [ ] **Step 6: Commit** — `feat(routing): default Claude model selection to Haiku`

---

### Task 3: Accurate per-model cost from real API usage (single pricing source)

**Files:**
- Modify: `api_optimizer.py:53-79` (`TokenOptimizer.__init__` + remove hardcoded class constants), `api_optimizer.py:175-179` (`calculate_cost`)
- Modify: `ai_analyzer.py:195-212` (use `message.usage` instead of tiktoken counts), `ai_analyzer.py:250-266` (return usage)
- Test: `tests/test_pricing.py`

**Interfaces:**
- Produces: `TokenOptimizer(input_cost_per_1m, output_cost_per_1m).calculate_cost(in, out, cache_read=0)` returns USD using the **selected model's** rates, charging `cache_read` tokens at `0.1×` input rate.

- [ ] **Step 1: Write the failing test** — `tests/test_pricing.py`

```python
from api_optimizer import TokenOptimizer


def test_cost_uses_injected_rates_not_opus():
    opt = TokenOptimizer(input_cost_per_1m=1.00, output_cost_per_1m=5.00)  # Haiku 4.5
    # 1M in + 1M out = $1 + $5 = $6 (NOT the old $90 Opus-3 figure)
    assert round(opt.calculate_cost(1_000_000, 1_000_000), 2) == 6.00


def test_cache_reads_billed_at_one_tenth_input():
    opt = TokenOptimizer(input_cost_per_1m=1.00, output_cost_per_1m=5.00)
    # 1M cache-read tokens = $1.00 * 0.1 = $0.10
    assert round(opt.calculate_cost(0, 0, cache_read_tokens=1_000_000), 2) == 0.10
```

- [ ] **Step 2: Run it, confirm FAIL** — `python -m pytest tests/test_pricing.py -v`.

- [ ] **Step 3: Make `TokenOptimizer` take rates** — replace class constants (`api_optimizer.py:56-58`) and `__init__` (60-71):

```python
class TokenOptimizer:
    """Optimizes content to minimize token usage"""

    def __init__(
        self,
        optimization_level: OptimizationLevel = OptimizationLevel.MINIMAL,
        input_cost_per_1m: float = 1.00,
        output_cost_per_1m: float = 5.00,
    ):
        """Initialize token optimizer with the selected model's pricing"""
        self.optimization_level = optimization_level
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self.encoder = None
        if tiktoken:
            try:
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                logger.warning("Failed to initialize tiktoken encoder")
```

- [ ] **Step 4: Update `calculate_cost`** (`api_optimizer.py:175-179`):

```python
    def calculate_cost(
        self, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0
    ) -> float:
        """Calculate API cost in USD using the selected model's rates"""
        input_cost = (input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_cost_per_1m
        cache_cost = (cache_read_tokens / 1_000_000) * self.input_cost_per_1m * 0.1
        return input_cost + output_cost + cache_cost
```

- [ ] **Step 5: Wire model rates into the optimizer** — wherever `TokenOptimizer`/`APIOptimizer` is constructed in `ai_analyzer.py` (`__init__`, near the `ai_config` setup), pass the selected model's costs:

```python
input_rate = self.ai_config["model_info"]["cost_per_1m_input"]
output_rate = self.ai_config["model_info"]["cost_per_1m_output"]
# pass input_rate/output_rate down into TokenOptimizer(...)
```
Read `ai_analyzer.py:40-110` to find the exact construction site and `ai_config` shape; thread the two rates through `APIOptimizer` to `TokenOptimizer`. If `model_info` is a `ModelInfo` dataclass rather than a dict, use attribute access (`.cost_per_1m_input`).

- [ ] **Step 6: Use real API usage for metrics + cost** — in `_get_claude_response` (`ai_analyzer.py:250-266`) return the `message.usage` alongside text, and in `analyze_page` (lines 195-212) prefer real counts:

```python
# _get_claude_response: return both text and usage
response_text = message.content[0].text
usage = message.usage  # input_tokens, output_tokens, cache_read_input_tokens
return response_text, usage
```
Update `_get_ai_response` and `analyze_page` so that, for the Claude path, `input_tokens`/`output_tokens` come from `usage` and cost uses `calculate_cost(usage.input_tokens, usage.output_tokens, getattr(usage, "cache_read_input_tokens", 0) or 0)`. Keep the tiktoken fallback for non-Claude providers that don't return usage. Log `cache_read_input_tokens` so caching is observable.

- [ ] **Step 7: Delete the stale duplicate pricing constants** — `grep -rn "INPUT_COST_PER_1M\|OUTPUT_COST_PER_1M\|75.00\|cost.*15.00" cost_monitor.py api_optimizer.py`. In `cost_monitor.py`, replace any hardcoded Claude input/output constants with values passed in from the selected model (or import from `ai_providers.MODEL_INFO`). Do not leave a second source of truth.

- [ ] **Step 8: Run tests + lint** — `python -m pytest tests/test_pricing.py -v && ruff check .` → PASS.

- [ ] **Step 9: Commit** — `feat(cost): per-model pricing from real API usage; bill cache reads at 0.1x`

---

### Task 4: Prompt caching via cached system prompt

**Files:**
- Modify: `ai_analyzer.py` `_create_analysis_prompt` (read it first — ~lines 290-360) and `_get_claude_response` (250-266)

**Interfaces:**
- Consumes: real-usage plumbing from Task 3.
- Produces: Claude request sends a stable `system` block with `cache_control: {"type": "ephemeral"}` (static taxonomy/instructions) and a per-page `user` message (dynamic content).

- [ ] **Step 1: Read `_create_analysis_prompt`** and split its output into two strings: `system_text` (the fixed taxonomy, classification rules, JSON schema instruction — identical across pages) and `user_text` (this page's title/content/properties). Change its signature to return `(system_text, user_text)`.

- [ ] **Step 2: Update `analyze_page`** (line 189) to receive both: `system_text, user_text = self._create_analysis_prompt(prepared_content, workspace_context)`. Apply MINIMAL optimization only to `user_text` (line 192-193).

- [ ] **Step 3: Update `_get_claude_response`** to send the cached system block:

```python
    def _get_claude_response(self, system_text: str, user_text: str):
        """Get response from Claude with a cached system prompt"""
        try:
            message = self.client.messages.create(
                model=self.ai_config["model"],
                max_tokens=(
                    500
                    if self.optimization_level == OptimizationLevel.MINIMAL
                    else 2000
                ),
                temperature=0.3,  # Haiku 4.5 supports temperature; no effort/output_config
                system=[
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_text}],
            )
            return message.content[0].text, message.usage
        except Exception as e:
            logger.error(f"Error getting Claude response: {e}")
            raise
```
Thread `system_text`/`user_text` through `_get_ai_response`. For OpenAI/Gemini, keep their existing single-prompt shape (concatenate `system_text + "\n\n" + user_text`, or pass `system_text` as their system message — OpenAI already has a system message at line 279).

- [ ] **Step 4: Add a test with a mocked client** — `tests/test_caching.py`: construct the analyzer with a stubbed `self.client` whose `messages.create` records kwargs, call `_get_claude_response("SYS", "USR")`, assert the call included `system=[{... "cache_control": {"type": "ephemeral"} ...}]` and `messages=[{"role":"user","content":"USR"}]`. Stub `message.usage` with `input_tokens`/`output_tokens`/`cache_read_input_tokens`.

- [ ] **Step 5: Run tests** — `python -m pytest tests/test_caching.py -v` → PASS.

- [ ] **Step 6: Note the conditional payoff** — add a one-line comment above the `system` block: `# Cache pays off only when system_text >= 4096 tokens (Haiku 4.5 min prefix); verify via usage.cache_read_input_tokens`.

- [ ] **Step 7: Commit** — `feat(cache): cache static system prompt with cache_control ephemeral`

---

### Task 5: Kill the similarity cache (correctness fix)

**Files:**
- Modify: `ai_analyzer.py:165-171` (remove `check_cache_and_similarity` call) and `:230-231` (remove `smart_cache.store`)
- Modify: `api_optimizer.py:502-514` (`check_cache_and_similarity`) — make it a no-op stub or remove callers
- Test: `tests/test_no_similarity.py`

**Interfaces:**
- Produces: `analyze_page` never returns another page's analysis; only the exact content-hash `response_cache` (line 177) serves cache hits.

- [ ] **Step 1: Write the failing test** — `tests/test_no_similarity.py`: build two pages with ≥0.85 Jaccard-similar text but different titles; stub the Claude client to return a deterministic classification keyed off the title; analyze both; assert each result's classification matches its own page (i.e. page B did **not** receive page A's cached analysis). With the similarity cache active this fails.

- [ ] **Step 2: Run it, confirm FAIL** — `python -m pytest tests/test_no_similarity.py -v`.

- [ ] **Step 3: Remove the similarity read** — delete `ai_analyzer.py:165-171` (the `check_cache_and_similarity` block). Keep the exact-hash cache at lines 173-180.

- [ ] **Step 4: Remove the similarity write** — delete `ai_analyzer.py:230-231` (`self.api_optimizer.smart_cache.store(...)`). Keep `self.response_cache[cache_key] = analysis` + `_save_cache()`.

- [ ] **Step 5: Neutralize `check_cache_and_similarity`** — in `api_optimizer.py:502-514`, since `analyze_page` no longer calls it, remove the method and any now-unused `smart_cache` attribute references on `APIOptimizer` (full `SmartCache` deletion happens in Task 9).

- [ ] **Step 6: Run tests** — `python -m pytest tests/test_no_similarity.py -v` → PASS.

- [ ] **Step 7: Commit** — `fix(cache): remove similarity cache that returned other pages' classifications`

---

### Task 6: Cache key includes last_edited_time + prompt version

**Files:**
- Modify: `config.py` (add `prompt_version` setting), `ai_analyzer.py:448-475` (`_generate_cache_key`)
- Test: `tests/test_cache_key.py`

**Interfaces:**
- Consumes: `settings.prompt_version`.
- Produces: cache key changes when a page's `last_edited_time` changes or when `prompt_version` changes (so a taxonomy edit invalidates stale analyses).

- [ ] **Step 1: Write the failing test** — `tests/test_cache_key.py`

```python
def test_cache_key_changes_with_last_edited_time(analyzer):
    base = {"title": "X", "content": "hello", "properties": {}, "last_edited_time": "2026-01-01T00:00:00Z"}
    edited = {**base, "last_edited_time": "2026-06-01T00:00:00Z"}
    assert analyzer._generate_cache_key(base) != analyzer._generate_cache_key(edited)


def test_cache_key_changes_with_prompt_version(analyzer, monkeypatch):
    page = {"title": "X", "content": "hello", "properties": {}, "last_edited_time": "2026-01-01T00:00:00Z"}
    k1 = analyzer._generate_cache_key(page)
    monkeypatch.setattr(analyzer.settings, "prompt_version", "v2")
    assert analyzer._generate_cache_key(page) != k1
```
Provide an `analyzer` fixture that builds the analyzer with a stubbed client (no network).

- [ ] **Step 2: Run it, confirm FAIL** — `python -m pytest tests/test_cache_key.py -v`.

- [ ] **Step 3: Add the setting** — `config.py`, in the Processing block (~line 92):

```python
    prompt_version: str = Field(
        "v1", description="Bump to invalidate cached analyses when the prompt/taxonomy changes"
    )
```

- [ ] **Step 4: Include both in the hashed dict** — `ai_analyzer.py:463-469`:

```python
            content_str = json.dumps(
                {
                    "title": page_content.get("title"),
                    "content": page_content.get("content"),
                    "properties": props,
                    "last_edited_time": page_content.get("last_edited_time"),
                    "prompt_version": self.settings.prompt_version,
                },
                sort_keys=True,
```

- [ ] **Step 5: Run tests** — `python -m pytest tests/test_cache_key.py -v` → PASS.

- [ ] **Step 6: Commit** — `feat(cache): key analyses on last_edited_time + prompt_version`

---

### Task 7: Bump the Anthropic SDK

**Files:** Modify `requirements.txt`

- [ ] **Step 1: Find the pin** — `grep -n anthropic requirements.txt` (currently `anthropic==0.21.3`).
- [ ] **Step 2: Bump** — set `anthropic>=0.40,<1.0` (a current release supporting `cache_control` + `message.usage.cache_read_input_tokens`). If a `uv.lock`/`requirements.lock` exists, regenerate.
- [ ] **Step 3: Install + smoke** — `uv sync` (or `pip install -r requirements.txt`), then `python -c "import anthropic, ai_analyzer"` → no import error.
- [ ] **Step 4: Run full suite** — `python -m pytest -q` → green.
- [ ] **Step 5: Commit** — `build: bump anthropic SDK for prompt caching + usage fields`

---

## SLICE 2 — Tidy (efficiency)

### Task 8: Delete dead modules

**Files:** Delete `claude_analyzer.py`, `error_recovery.py`, `performance_enhancer.py`, `workspace_scanner.py`

- [ ] **Step 1: Prove they are unimported** — `grep -rn "import claude_analyzer\|from claude_analyzer\|import error_recovery\|from error_recovery\|import performance_enhancer\|from performance_enhancer\|import workspace_scanner\|from workspace_scanner" --include=*.py .` → expect zero hits outside the files themselves and their own tests.
- [ ] **Step 2: Delete the four files** — `git rm claude_analyzer.py error_recovery.py performance_enhancer.py workspace_scanner.py`.
- [ ] **Step 3: Remove orphaned references** — re-grep; delete any docs/`__init__` references that mention them.
- [ ] **Step 4: Run suite + import check** — `python -m pytest -q && python -c "import notion_organizer"` → green.
- [ ] **Step 5: Commit** — `refactor: delete ~2,170 lines of unimported dead modules`

---

### Task 9: Remove the SmartCache class (consolidate to one cache)

**Files:** Modify `api_optimizer.py` (delete `SmartCache` class lines ~182-300 and its construction in `APIOptimizer`)

- [ ] **Step 1: Confirm no remaining callers** — after Task 5, `grep -rn "SmartCache\|smart_cache\|get_similar_cached\|_calculate_similarity" --include=*.py .` → only definitions remain.
- [ ] **Step 2: Delete `SmartCache`** and remove its instantiation/attribute on `APIOptimizer`; drop now-unused imports (`pickle`, `hashlib` if unused elsewhere — verify with grep before removing).
- [ ] **Step 3: Run suite + lint** — `python -m pytest -q && ruff check .` → green.
- [ ] **Step 4: Commit** — `refactor(cache): remove SmartCache; rely on exact content-hash cache`

---

### Task 10: Stop the workspace scan double-fetch

**Files:** Modify `notion_wrapper.py:286-339` (`scan_workspace`), bump `notion_api_version`

- [ ] **Step 1: Read `scan_workspace`** (lines 286-345). It paginates full `databases.query` payloads only to `len()` pages (lines 324-339), then analysis re-fetches each page.
- [ ] **Step 2: Choose the cheaper path** — either (a) count via `page_size=1` + the API's total where available, or (b) fetch each page **once** in the scan and return the page payloads for reuse downstream instead of re-fetching in `notion_organizer.py`. Prefer (b): return the already-fetched page dicts so analysis consumes them directly. Keep behavior identical (same pages analyzed).
- [ ] **Step 3: Add a regression test** — `tests/test_scan.py`: stub the Notion client to count `databases.query` / `pages.retrieve` calls; assert each page is fetched at most once across scan + analyze for a small fixture workspace.
- [ ] **Step 4: Bump Notion API version** — `config.py:62` → a current `notion_api_version` (e.g. `"2025-09-03"`); verify the pinned `notion-client` accepts it, else leave a comment and keep `2022-06-28`.
- [ ] **Step 5: Run tests** — `python -m pytest tests/test_scan.py -v` → PASS.
- [ ] **Step 6: Commit** — `perf(notion): scan fetches each page once; reuse for analysis`

---

### Task 11: `notioniq` command + packaging

**Files:** Modify `setup.py:1-45`

- [ ] **Step 1: Add the entry point** — in `setup()`:

```python
    entry_points={
        "console_scripts": [
            "notioniq=notion_organizer:main",
        ],
    },
```
Confirm `notion_organizer.py` exposes `main` (the `@click.command` at line 1004/1072).

- [ ] **Step 2: Install editable + smoke** — `pip install -e .` then `notioniq --help` → prints the CLI help.
- [ ] **Step 3: Update README/CLAUDE.md run commands** — replace `python notion_organizer.py` with `notioniq` in the primary examples (keep one note that the module form still works).
- [ ] **Step 4: Commit** — `feat(cli): add notioniq console entry point`

---

## SLICE 3 — Taste (UX)

### Task 12: Elegant run summary panel

**Files:** Modify `notion_organizer.py` (replace the 6-step narrative ~lines 129-198 and the trailing summaries ~lines 962-985)

- [ ] **Step 1: Read the current output flow** (lines 120-200, 950-990).
- [ ] **Step 2: Replace the 6-step narrative with quiet progress** — keep a single `rich` `Progress` spinner for the page loop; drop the per-step "Step N: …" banners. Keep per-page classification lines only at `--verbose` (add the flag if absent; default off).
- [ ] **Step 3: Compose one final summary `Panel`** — a single bordered panel titled `NotionIQ` showing: pages analyzed, cache-hit rate, top classifications (compact table), total cost (from corrected pricing), tokens, elapsed time. Pull numbers from `api_optimizer.metrics`. Keep it under ~12 lines, aligned, no emoji-noise.
- [ ] **Step 4: Manual verify** — `notioniq --dry-run --batch-size 2` (mock or small DB) renders one clean panel, no 6-step spam.
- [ ] **Step 5: Commit** — `feat(ux): single elegant run-summary panel`

---

### Task 13: `notioniq init` setup wizard

**Files:** Modify `notion_organizer.py` (add a `click` sub-command group or an `init` command), fold in `quickstart.py` logic; then delete `quickstart.py`

- [ ] **Step 1: Decide the surface** — convert the single `@click.command` into a `@click.group()` with `run` (current default behavior) and `init`, OR add `--init` handling. Prefer a group with a default command so bare `notioniq` still runs analysis. Read `quickstart.py:1-123` for the existing interactive flow to reuse.
- [ ] **Step 2: Implement `init`** — interactively prompt (via `rich.prompt.Prompt`) for `NOTION_API_KEY`, `NOTION_INBOX_DATABASE_ID`, `ANTHROPIC_API_KEY`; validate non-empty (reuse `security.py` validators if present); write/update `.env`; print a success panel with the next command (`notioniq`).
- [ ] **Step 3: Wire entry point** — ensure `notioniq init` resolves through the `console_scripts` `main`.
- [ ] **Step 4: Delete `quickstart.py`** once its logic is folded in; update any references.
- [ ] **Step 5: Manual verify** — run `notioniq init` in a temp dir → writes a valid `.env`; `notioniq init` again offers to update without clobbering unrelated keys.
- [ ] **Step 6: Commit** — `feat(cli): notioniq init setup wizard (replaces quickstart.py)`

---

### Task 14: Smart defaults + markdown export

**Files:** Modify `notion_organizer.py` (options + report writing ~lines 590-600)

- [ ] **Step 1: Confirm cost-safe defaults** — `--optimization` stays `minimal`, `--daily-budget` stays `$5`, model defaults to Haiku (Task 1). Ensure bare `notioniq` (no flags) runs a sensible workspace analysis end-to-end.
- [ ] **Step 2: Add `--format`** — `@click.option("--format", "output_format", type=click.Choice(["json", "markdown", "both"]), default="json")`.
- [ ] **Step 3: Implement markdown writer** — alongside the JSON report (line 592-596), when `markdown`/`both`, write `output/analysis_report_{timestamp}.md`: a human-readable summary (counts, per-database recommendations table, top actions). Add a small `_write_markdown_report(report) -> Path` helper.
- [ ] **Step 4: Add a test** — `tests/test_report.py`: feed a sample report dict to `_write_markdown_report`, assert the file exists and contains expected headers/rows.
- [ ] **Step 5: Run tests** — `python -m pytest tests/test_report.py -v` → PASS.
- [ ] **Step 6: Commit** — `feat(report): --format markdown export + Just-Works defaults`

---

## Self-Review (completed)

- **Spec coverage:** All 12 spec items map to tasks — 1:Task1, 2:Task4, 3:Task3, 4:Task5, 5:Task6, 6:Task8, 7:Task9, 8:Task10, 9:Task11, 10:Task12, 11:Task13, 12:Task14. Model routing (item 1) split across Task1+Task2; SDK bump (item 2 dependency) is Task7. ✅
- **Batch API:** correctly excluded (spec non-goal). ✅
- **Type consistency:** `_get_claude_response` returns `(text, usage)` in Tasks 3 & 4; `calculate_cost(input, output, cache_read_tokens=0)` consistent across Tasks 3 & 4; `_generate_cache_key` signature unchanged (dict in, str out) in Task 6. ✅
- **Open reads flagged:** Tasks 3, 4, 10, 13 require reading a function body before editing (construction site, `_create_analysis_prompt`, `scan_workspace`, `quickstart.py`) — each step says so explicitly rather than inventing code. ✅
- **Test isolation:** every test stubs the Anthropic/Notion client — no live spend. ✅

## Execution order

Slice 1 (Tasks 1→7) first — verifies the cost thesis. Then Slice 2 (8→11) cleanup. Then Slice 3 (12→14) UX. Tasks within a slice are mostly independent; Task 4 depends on Task 3's `(text, usage)` return, and Task 9 depends on Task 5.
