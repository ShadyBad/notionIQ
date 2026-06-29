"""Tests for prompt caching via a cached system prompt block.

The Claude request must send the static taxonomy/instructions as a ``system``
block carrying ``cache_control: {"type": "ephemeral"}`` and only the per-page
content as the ``user`` message. ``_get_claude_response`` must still return
``(text, usage)``.
"""

from types import SimpleNamespace

from ai_analyzer import UniversalAIAnalyzer
from api_optimizer import OptimizationLevel


class _RecordingMessages:
    """Stub for ``client.messages`` that records the create() kwargs."""

    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        usage = SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
        )
        message = SimpleNamespace(
            content=[SimpleNamespace(text="OK")],
            usage=usage,
        )
        return message


def _make_analyzer():
    """Build an analyzer without running __init__ (no network/client setup)."""
    analyzer = UniversalAIAnalyzer.__new__(UniversalAIAnalyzer)
    analyzer.optimization_level = OptimizationLevel.MINIMAL
    analyzer.provider_type = "claude"
    analyzer.ai_config = {"model": "claude-haiku-4-5"}
    analyzer.client = SimpleNamespace(messages=_RecordingMessages())
    return analyzer


def test_claude_response_sends_cached_system_block():
    analyzer = _make_analyzer()

    text, usage = analyzer._get_claude_response("SYS", "USR")

    assert text == "OK"
    assert usage.input_tokens == 10
    assert usage.cache_read_input_tokens == 0

    kwargs = analyzer.client.messages.kwargs
    assert kwargs["system"] == [
        {
            "type": "text",
            "text": "SYS",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert kwargs["messages"] == [{"role": "user", "content": "USR"}]


def test_get_ai_response_threads_system_and_user_for_claude():
    analyzer = _make_analyzer()

    text, usage = analyzer._get_ai_response("SYS", "USR")

    assert text == "OK"
    assert usage is not None
    kwargs = analyzer.client.messages.kwargs
    assert kwargs["system"][0]["text"] == "SYS"
    assert kwargs["messages"][0]["content"] == "USR"
