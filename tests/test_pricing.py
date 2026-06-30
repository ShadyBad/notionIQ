"""Tests for accurate per-model pricing sourced from real API usage."""

from types import SimpleNamespace

from api_optimizer import APIOptimizer, OptimizationLevel, TokenOptimizer


def test_cost_uses_injected_rates_not_opus():
    opt = TokenOptimizer(input_cost_per_1m=1.00, output_cost_per_1m=5.00)  # Haiku 4.5
    # 1M in + 1M out = $1 + $5 = $6 (NOT the old $90 Opus-3 figure)
    assert round(opt.calculate_cost(1_000_000, 1_000_000), 2) == 6.00


def test_cache_reads_billed_at_one_tenth_input():
    opt = TokenOptimizer(input_cost_per_1m=1.00, output_cost_per_1m=5.00)
    # 1M cache-read tokens = $1.00 * 0.1 = $0.10
    assert round(opt.calculate_cost(0, 0, cache_read_tokens=1_000_000), 2) == 0.10


def test_record_api_usage_threads_cache_read_into_total_cost(monkeypatch):
    # Regression: record_api_usage must pass cache_read_tokens to calculate_cost,
    # otherwise metrics.total_cost (shown in the summary panel) is understated.
    monkeypatch.setattr(APIOptimizer, "_load_metrics", lambda self: None)
    monkeypatch.setattr(APIOptimizer, "_save_metrics", lambda self: None)
    opt = APIOptimizer(
        SimpleNamespace(),
        OptimizationLevel.MINIMAL,
        input_cost_per_1m=1.00,
        output_cost_per_1m=5.00,
    )
    opt.record_api_usage(1_000_000, 0, from_cache=False, cache_read_tokens=1_000_000)
    # input $1.00 + cache-read (1M * $1.00 * 0.1) $0.10 = $1.10
    assert round(opt.metrics.total_cost, 2) == 1.10
