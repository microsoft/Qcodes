from __future__ import annotations

from typing import TYPE_CHECKING

from qcodes.utils import ParameterDiff, diff_param_values

from .data_set import load_by_id

if TYPE_CHECKING:
    from .data_set_protocol import DataSetProtocol


def diff_param_snapshots(
    left: DataSetProtocol, right: DataSetProtocol
) -> ParameterDiff:
    """
    Given two datasets, returns the differences between
    parameter values in each of their snapshots.
    """
    left_snapshot = left.snapshot
    right_snapshot = right.snapshot

    if left_snapshot is None or right_snapshot is None:
        if left_snapshot is None:
            empty = left
        else:
            empty = right
        raise RuntimeError(
            f"Tried to compare two snapshots"
            f"but the snapshot of {empty.run_id} "
            f"is empty."
        )

    return diff_param_values(left_snapshot, right_snapshot)


def diff_param_values_by_id(left_id: int, right_id: int) -> ParameterDiff:
    """
    Given the IDs of two datasets, returns the differences between
    parameter values in each of their snapshots.
    """
    return diff_param_snapshots(load_by_id(left_id), load_by_id(right_id))
