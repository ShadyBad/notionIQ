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
