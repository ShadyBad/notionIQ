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
