from __future__ import annotations

from typing import TYPE_CHECKING

from qcodes.dataset.linked_datasets.links import links_to_str
from qcodes.dataset.sqlite.queries import (
    _rewrite_timestamps,
    create_run,
    mark_run_complete,
)

if TYPE_CHECKING:
    from qcodes.dataset.data_set_protocol import DataSetProtocol
    from qcodes.dataset.sqlite.connection import ConnectionPlus


def _add_run_to_runs_table(
    dataset: DataSetProtocol,
    target_conn: ConnectionPlus,
    target_exp_id: int,
    create_run_table: bool = True,
) -> tuple[int, int, str | None]:
    metadata = dataset.metadata
    snapshot_raw = dataset._snapshot_raw
    captured_run_id = dataset.captured_run_id
    captured_counter = dataset.captured_counter
    parent_dataset_links = links_to_str(dataset.parent_dataset_links)
    target_counter, target_run_id, target_table_name = create_run(
        target_conn,
        target_exp_id,
        name=dataset.name,
        guid=dataset.guid,
        metadata=metadata,
        captured_run_id=captured_run_id,
        captured_counter=captured_counter,
        parent_dataset_links=parent_dataset_links,
        create_run_table=create_run_table,
        snapshot_raw=snapshot_raw,
        description=dataset.description,
    )
    mark_run_complete(target_conn, target_run_id)
    _rewrite_timestamps(
        target_conn,
        target_run_id,
        dataset.run_timestamp_raw,
        dataset.completed_timestamp_raw,
    )
    return target_counter, target_run_id, target_table_name
