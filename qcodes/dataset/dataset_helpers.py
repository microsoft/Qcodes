from typing import Optional

from qcodes.dataset.data_set_protocol import DataSetProtocol
from qcodes.dataset.descriptions.versioning.converters import new_to_old
from qcodes.dataset.linked_datasets.links import links_to_str
from qcodes.dataset.sqlite.connection import ConnectionPlus
from qcodes.dataset.sqlite.queries import (
    _rewrite_timestamps,
    create_run,
    mark_run_complete,
)


def _add_run_to_runs_table(
    dataset: DataSetProtocol,
    target_conn: ConnectionPlus,
    target_exp_id: int,
    create_run_table: bool = True,
) -> Optional[str]:
    if dataset._parameters is not None:
        param_names = dataset._parameters.split(",")
    else:
        param_names = []
    parspecs_dict = {
        p.name: p for p in new_to_old(dataset.description.interdeps).paramspecs
    }
    parspecs = [parspecs_dict[p] for p in param_names]
    metadata = dataset.metadata
    snapshot_raw = dataset._snapshot_raw
    captured_run_id = dataset.captured_run_id
    captured_counter = dataset.captured_counter
    parent_dataset_links = links_to_str(dataset.parent_dataset_links)
    _, target_run_id, target_table_name = create_run(
        target_conn,
        target_exp_id,
        name=dataset.name,
        guid=dataset.guid,
        parameters=parspecs,
        metadata=metadata,
        captured_run_id=captured_run_id,
        captured_counter=captured_counter,
        parent_dataset_links=parent_dataset_links,
        create_run_table=create_run_table,
        snapshot_raw=snapshot_raw,
    )
    mark_run_complete(target_conn, target_run_id)
    _rewrite_timestamps(
        target_conn,
        target_run_id,
        dataset.run_timestamp_raw,
        dataset.completed_timestamp_raw,
    )
    return target_table_name
