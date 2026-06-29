"""Tests for the `notioniq init` setup wizard and its .env writer."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

import notion_organizer as no


def test_update_env_preserves_unrelated_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# existing config\nBATCH_SIZE=10\nNOTION_API_KEY=old\nENABLE_CACHING=true\n"
    )

    no._update_env_file(
        env,
        {
            "NOTION_API_KEY": "new_notion",
            "NOTION_INBOX_DATABASE_ID": "abc123",
            "ANTHROPIC_API_KEY": "sk-ant-xyz",
        },
    )

    lines = env.read_text().splitlines()
    # Comment and unrelated keys are preserved untouched.
    assert "# existing config" in lines
    assert "BATCH_SIZE=10" in lines
    assert "ENABLE_CACHING=true" in lines
    # Targeted key is updated in place, not duplicated.
    assert lines.count("NOTION_API_KEY=new_notion") == 1
    assert not any(line == "NOTION_API_KEY=old" for line in lines)
    # New keys are appended.
    assert "NOTION_INBOX_DATABASE_ID=abc123" in lines
    assert "ANTHROPIC_API_KEY=sk-ant-xyz" in lines


def test_update_env_creates_file_when_absent(tmp_path):
    env = tmp_path / ".env"
    no._update_env_file(env, {"NOTION_API_KEY": "k"})
    assert env.exists()
    assert env.read_text() == "NOTION_API_KEY=k\n"


def test_init_command_writes_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    answers = iter(["secret_tok", "deadbeef1234", "sk-ant-key"])
    with patch("notion_organizer.Prompt.ask", side_effect=lambda *a, **k: next(answers)):
        result = CliRunner().invoke(no.main, ["init"])

    assert result.exit_code == 0, result.output
    content = (tmp_path / ".env").read_text()
    assert "NOTION_API_KEY=secret_tok" in content
    assert "NOTION_INBOX_DATABASE_ID=deadbeef1234" in content
    assert "ANTHROPIC_API_KEY=sk-ant-key" in content
