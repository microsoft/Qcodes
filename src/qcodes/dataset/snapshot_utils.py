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


def diff_start_end_snapshot(dataset: DataSetProtocol) -> ParameterDiff:
    """
    Given a dataset, returns the differences between the parameter values in
    the snapshot taken at the start of the measurement and the snapshot taken
    at the end of the measurement.

    Note that the snapshot at the end of a measurement is only taken if
    snapshotting at the end is enabled. See the ``snapshot_at_end`` key in the
    ``dataset`` section of the QCoDeS config.

    Args:
        dataset: the dataset to compare the start and end snapshots of.

    Returns:
        The differences between the start and the end snapshot where the start
        snapshot is the left hand side and the end snapshot the right hand side.

    Raises:
        RuntimeError: if the dataset does not contain both a start and an end
            snapshot.

    """
    start_snapshot = dataset.snapshot
    end_snapshot = dataset.end_snapshot

    if start_snapshot is None:
        raise RuntimeError(
            f"Tried to compare the start and end snapshot of run "
            f"{dataset.run_id} but the snapshot taken at the start of the "
            f"measurement is empty."
        )
    if end_snapshot is None:
        raise RuntimeError(
            f"Tried to compare the start and end snapshot of run "
            f"{dataset.run_id} but the snapshot taken at the end of the "
            f"measurement is empty."
        )

    return diff_param_values(start_snapshot, end_snapshot)


def diff_start_end_snapshot_by_id(run_id: int) -> ParameterDiff:
    """
    Given the ID of a dataset, returns the differences between the parameter
    values in the snapshot taken at the start of the measurement and the
    snapshot taken at the end of the measurement.
    """
    return diff_start_end_snapshot(load_by_id(run_id))
