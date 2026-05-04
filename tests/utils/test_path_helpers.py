"""
Tests for qcodes.utils.path_helpers - path utility functions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from qcodes.utils.path_helpers import (
    QCODES_USER_PATH_ENV,
    get_qcodes_path,
    get_qcodes_user_path,
)


def test_get_qcodes_path_returns_string() -> None:
    """Test that get_qcodes_path returns a string."""
    result = get_qcodes_path()
    assert isinstance(result, str)


def test_get_qcodes_path_ends_with_separator() -> None:
    """Test that get_qcodes_path returns a path ending with os.sep."""
    result = get_qcodes_path()
    assert result.endswith(os.sep)


def test_get_qcodes_path_contains_qcodes() -> None:
    """Test that the returned path contains 'qcodes'."""
    result = get_qcodes_path()
    assert "qcodes" in result.lower()


def test_get_qcodes_path_with_subfolder() -> None:
    """Test that get_qcodes_path appends a subfolder."""
    result = get_qcodes_path("subdir")
    assert result.endswith("subdir" + os.sep)


def test_get_qcodes_path_with_nested_subfolders() -> None:
    """Test that get_qcodes_path appends multiple subfolder parts."""
    result = get_qcodes_path("subdir", "nested")
    assert "subdir" in result
    assert result.endswith("nested" + os.sep)


def test_get_qcodes_user_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_qcodes_user_path returns ~/.qcodes by default."""
    monkeypatch.delenv(QCODES_USER_PATH_ENV, raising=False)
    result = get_qcodes_user_path()
    expected = os.path.join(str(Path.home()), ".qcodes")
    assert result == expected


def test_get_qcodes_user_path_respects_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that get_qcodes_user_path uses QCODES_USER_PATH env var."""
    custom_path = str(tmp_path / "custom_qcodes")
    monkeypatch.setenv(QCODES_USER_PATH_ENV, custom_path)
    result = get_qcodes_user_path()
    assert result == custom_path


def test_get_qcodes_user_path_appends_file_parts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that get_qcodes_user_path appends file parts."""
    custom_path = str(tmp_path / "custom_qcodes")
    monkeypatch.setenv(QCODES_USER_PATH_ENV, custom_path)
    result = get_qcodes_user_path("config.json")
    assert result == os.path.join(custom_path, "config.json")


def test_get_qcodes_user_path_appends_nested_parts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that get_qcodes_user_path appends multiple nested parts."""
    custom_path = str(tmp_path / "custom_qcodes")
    monkeypatch.setenv(QCODES_USER_PATH_ENV, custom_path)
    result = get_qcodes_user_path("subdir", "file.txt")
    assert result == os.path.join(custom_path, "subdir", "file.txt")


def test_qcodes_user_path_env_constant() -> None:
    """Test that the env variable constant has the expected value."""
    assert QCODES_USER_PATH_ENV == "QCODES_USER_PATH"
