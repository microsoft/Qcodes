import json
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.linked_datasets.links import Link, str_to_links
from qcodes.dataset.sqlite.connection import ConnectionPlus
from qcodes.dataset.sqlite.queries import (
    ExperimentAttributeDict,
    get_raw_run_attributes,
    raw_time_to_str_time,
)

from .descriptions.versioning import serialization


class RunAttributesDict(TypedDict):
    run_id: int
    counter: int
    captured_run_id: int
    captured_counter: int
    experiment: ExperimentAttributeDict
    name: str
    run_timestamp: Optional[str]
    completed_timestamp: Optional[str]
    metadata: Dict[str, Any]
    parent_dataset_links: List[Link]
    run_description: RunDescriber
    snapshot: Optional[Dict[str, Any]]


def get_run_attributes(conn: ConnectionPlus, guid: str) -> Optional[RunAttributesDict]:
    """
    Look up all information and metadata about a given dataset captured
    in the database.

    Args:
        conn: Connection to the database
        guid: GUID of the dataset to look up

    Returns:
        Dictionary of information about the dataset.
    """
    raw_attributes = get_raw_run_attributes(conn, guid)

    if raw_attributes is None:
        return None

    attributes: RunAttributesDict = {
        "run_id": raw_attributes["run_id"],
        "counter": raw_attributes["counter"],
        "captured_run_id": raw_attributes["captured_run_id"],
        "captured_counter": raw_attributes["captured_counter"],
        "experiment": raw_attributes["experiment"],
        "name": raw_attributes["name"],
        "run_timestamp": raw_time_to_str_time(raw_attributes["run_timestamp"]),
        "completed_timestamp": raw_time_to_str_time(
            raw_attributes["completed_timestamp"]
        ),
        "metadata": raw_attributes["metadata"],
        "parent_dataset_links": str_to_links(raw_attributes["parent_dataset_links"]),
        "run_description": serialization.from_json_to_current(
            raw_attributes["run_description"]
        ),
        "snapshot": json.loads(raw_attributes["snapshot"])
        if raw_attributes["snapshot"] is not None
        else None,
    }
    return attributes
