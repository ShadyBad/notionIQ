"""The CLI must fail gracefully (not dump a traceback) when unconfigured."""

from click.testing import CliRunner
from pydantic import BaseModel, ValidationError

from notion_organizer import main


def _missing_config_error(*args, **kwargs):
    class _Req(BaseModel):
        notion_api_key: str
        notion_inbox_database_id: str

    _Req()  # raises ValidationError with type="missing" for both fields


def test_run_without_config_shows_friendly_message(monkeypatch):
    # get_settings raising ValidationError simulates a first run with no .env/keys.
    monkeypatch.setattr("notion_organizer.get_settings", _missing_config_error)

    result = CliRunner().invoke(main, ["run", "--dry-run"])

    assert result.exit_code == 1
    assert "isn't configured" in result.output
    assert "notioniq init" in result.output
    # The raw pydantic dump / loguru backtrace must NOT leak to the user.
    assert "Traceback" not in result.output
    assert "validation error" not in result.output.lower()


def test_missing_config_error_helper_actually_raises():
    # Guard: the helper must raise a ValidationError carrying missing fields.
    try:
        _missing_config_error()
    except ValidationError as exc:
        types = {e["type"] for e in exc.errors()}
        assert "missing" in types
    else:  # pragma: no cover
        raise AssertionError("expected ValidationError")
