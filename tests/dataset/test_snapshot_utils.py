from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qcodes.dataset.snapshot_utils import diff_param_snapshots
from qcodes.utils import ParameterDiff


def _make_mock_dataset(run_id: int, snapshot: dict | None) -> MagicMock:
    ds = MagicMock()
    ds.run_id = run_id
    ds.snapshot = snapshot
    return ds


def test_diff_param_snapshots_both_have_snapshots() -> None:
    left_snapshot = {
        "station": {
            "parameters": {
                "p1": {"value": 1.0},
                "p2": {"value": 2.0},
            }
        }
    }
    right_snapshot = {
        "station": {
            "parameters": {
                "p1": {"value": 1.0},
                "p3": {"value": 3.0},
            }
        }
    }
    left = _make_mock_dataset(1, left_snapshot)
    right = _make_mock_dataset(2, right_snapshot)

    result = diff_param_snapshots(left, right)

    assert isinstance(result, ParameterDiff)
    assert result.left_only == {"p2": 2.0}
    assert result.right_only == {"p3": 3.0}
    assert result.changed == {}


def test_diff_param_snapshots_identical_snapshots() -> None:
    snapshot = {
        "station": {
            "parameters": {
                "p1": {"value": 1.0},
            }
        }
    }
    left = _make_mock_dataset(1, snapshot)
    right = _make_mock_dataset(2, snapshot)

    result = diff_param_snapshots(left, right)

    assert result.left_only == {}
    assert result.right_only == {}
    assert result.changed == {}


def test_diff_param_snapshots_changed_values() -> None:
    left_snapshot = {
        "station": {
            "parameters": {
                "p1": {"value": 1.0},
            }
        }
    }
    right_snapshot = {
        "station": {
            "parameters": {
                "p1": {"value": 99.0},
            }
        }
    }
    left = _make_mock_dataset(1, left_snapshot)
    right = _make_mock_dataset(2, right_snapshot)

    result = diff_param_snapshots(left, right)

    assert result.changed == {"p1": (1.0, 99.0)}


def test_diff_param_snapshots_raises_when_left_snapshot_is_none() -> None:
    left = _make_mock_dataset(run_id=5, snapshot=None)
    right = _make_mock_dataset(
        run_id=6,
        snapshot={"station": {"parameters": {"p1": {"value": 1.0}}}},
    )

    with pytest.raises(RuntimeError, match="5"):
        diff_param_snapshots(left, right)


def test_diff_param_snapshots_raises_when_right_snapshot_is_none() -> None:
    left = _make_mock_dataset(
        run_id=7,
        snapshot={"station": {"parameters": {"p1": {"value": 1.0}}}},
    )
    right = _make_mock_dataset(run_id=8, snapshot=None)

    with pytest.raises(RuntimeError, match="8"):
        diff_param_snapshots(left, right)


def test_diff_param_snapshots_raises_when_both_snapshots_are_none() -> None:
    left = _make_mock_dataset(run_id=10, snapshot=None)
    right = _make_mock_dataset(run_id=11, snapshot=None)

    # When both are None, the left dataset is identified as the empty one
    with pytest.raises(RuntimeError, match="10"):
        diff_param_snapshots(left, right)


def test_diff_param_snapshots_error_message_includes_run_id() -> None:
    left = _make_mock_dataset(run_id=42, snapshot=None)
    right = _make_mock_dataset(
        run_id=99,
        snapshot={"station": {"parameters": {}}},
    )

    with pytest.raises(RuntimeError) as exc_info:
        diff_param_snapshots(left, right)

    assert "42" in str(exc_info.value)
    assert "snapshot" in str(exc_info.value).lower()
