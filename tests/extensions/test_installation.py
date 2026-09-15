from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qcodes.extensions.installation import register_station_schema_with_vscode

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from pytest_mock import MockerFixture


def test_register_station_schema_with_unparsable_settings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, mocker: MockerFixture
) -> None:
    """A settings file that is not valid JSON is reported and left untouched."""
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}", encoding="utf-8")

    settings_path = tmp_path / "Code" / "User" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_content = '{\n    // comments are not valid json\n    "a": 1\n}'
    settings_path.write_text(settings_content, encoding="utf-8")

    mocker.patch("sys.platform", "win32")
    mocker.patch("qcodes.extensions.installation.SCHEMA_PATH", str(schema_path))
    mocker.patch(
        "qcodes.extensions.installation.os.path.expandvars",
        return_value=str(settings_path),
    )

    with caplog.at_level(logging.WARNING, logger="qcodes.extensions.installation"):
        register_station_schema_with_vscode()

    assert "Could not parse VSCode settings file" in caplog.text
    assert settings_path.read_text(encoding="utf-8") == settings_content
    assert not (tmp_path / "Code" / "User" / "settings.json_backup").exists()
