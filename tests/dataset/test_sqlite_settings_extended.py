"""
Extended tests for qcodes.dataset.sqlite.settings beyond the minimal
checks in test_sqlitesettings.py.
"""

from __future__ import annotations

from qcodes.dataset.sqlite.settings import SQLiteSettings, _read_settings

# --------------- _read_settings returns a 2-tuple of dicts ---------------


def test_read_settings_returns_two_dicts() -> None:
    result = _read_settings()
    assert isinstance(result, tuple)
    assert len(result) == 2
    limits, settings = result
    assert isinstance(limits, dict)
    assert isinstance(settings, dict)


# --------------- settings dict ---------------


def test_settings_contains_version_key() -> None:
    assert "VERSION" in SQLiteSettings.settings


def test_settings_version_is_string() -> None:
    assert isinstance(SQLiteSettings.settings["VERSION"], str)


def test_settings_dict_is_non_empty() -> None:
    assert len(SQLiteSettings.settings) >= 1


# --------------- limits dict ---------------


EXPECTED_LIMIT_KEYS = {
    "MAX_ATTACHED",
    "MAX_COLUMN",
    "MAX_COMPOUND_SELECT",
    "MAX_EXPR_DEPTH",
    "MAX_FUNCTION_ARG",
    "MAX_LENGTH",
    "MAX_LIKE_PATTERN_LENGTH",
    "MAX_PAGE_COUNT",
    "MAX_SQL_LENGTH",
    "MAX_VARIABLE_NUMBER",
}


def test_limits_contains_expected_keys() -> None:
    assert EXPECTED_LIMIT_KEYS.issubset(SQLiteSettings.limits.keys())


def test_each_limit_value_is_int_or_str() -> None:
    for key, value in SQLiteSettings.limits.items():
        assert isinstance(value, (int, str)), (
            f"Limit {key!r} should be int or str, got {type(value)}"
        )


def test_limits_has_ten_entries() -> None:
    assert len(SQLiteSettings.limits) == 10
