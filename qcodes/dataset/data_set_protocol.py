from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Mapping,
                    Optional, Sequence, Sized, Union)

import numpy as np
from typing_extensions import Protocol, runtime_checkable

from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
from qcodes.dataset.sqlite.connection import ConnectionPlus
from qcodes.dataset.sqlite.query_helpers import VALUES
from qcodes.dataset.linked_datasets.links import Link
from qcodes.dataset.export_config import DataExportType
from qcodes.dataset.descriptions.rundescriber import RunDescriber

SPECS = List[ParamSpec]
# Transition period type: SpecsOrInterDeps. We will allow both as input to
# the DataSet constructor for a while, then deprecate SPECS and finally remove
# the ParamSpec class
SpecsOrInterDeps = Union[SPECS, InterDependencies_]


if TYPE_CHECKING:
    from qcodes.station import Station


@runtime_checkable
class DataSetProtocol(Protocol, Sized):

    def __init__(self, path_to_db: Optional[str] = None,
                 run_id: Optional[int] = None,
                 conn: Optional[ConnectionPlus] = None,
                 exp_id: Optional[int] = None,
                 name: Optional[str] = None,
                 specs: Optional[SpecsOrInterDeps] = None,
                 values: Optional[VALUES] = None,
                 metadata: Optional[Mapping[str, Any]] = None,
                 shapes: Optional[Shapes] = None,
                 in_memory_cache: bool = True) -> None:
        pass

    def prepare(self,
                station: "Optional[Station]",
                interdeps: InterDependencies_,
                write_in_background: bool,
                shapes: Shapes = None,
                parent_datasets: Sequence[Dict[Any, Any]] = ()) -> None:
        pass

    def _enqueue_results(
            self,
            result_dict: Mapping[ParamSpecBase, np.ndarray]
    ) -> None:
        pass

    def get_metadata(self, tag: str) -> str:
        pass

    def add_metadata(self, tag: str, metadata: Any) -> None:
        pass

    @property
    def run_id(self) -> int:
        pass

    @property
    def captured_run_id(self) -> int:
        pass

    @property
    def captured_counter(self) -> int:
        pass

    @property
    def guid(self) -> str:
        pass

    def add_snapshot(self, snapshot: str, overwrite: bool = False) -> None:
        pass

    def mark_completed(self) -> None:
        pass

    @property
    def parent_dataset_links(self) -> List[Link]:
        pass

    def export(self,
               export_type: Optional[Union[DataExportType, str]] = None,
               path: Optional[str] = None,
               prefix: Optional[str] = None) -> None:
        pass

    @property
    def number_of_results(self) -> int:
        pass

    def _flush_data_to_database(self, block: bool = False) -> None:
        pass

    @property
    def paramspecs(self) -> Dict[str, ParamSpec]:
        pass

    @property
    def exp_name(self) -> str:
        pass

    @property
    def sample_name(self) -> str:
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def snapshot_raw(self) -> Optional[str]:
        pass

    @property
    def run_timestamp_raw(self) -> Optional[float]:
        pass

    def run_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
        pass

    @property
    def completed_timestamp_raw(self) -> Optional[float]:
        pass

    def completed_timestamp(
            self,
            fmt: str = "%Y-%m-%d %H:%M:%S"
    ) -> Optional[str]:
        pass

    @property
    def completed(self) -> bool:
        pass

    @completed.setter
    def completed(self, value: bool) -> None:
        pass

    @property
    def description(self) -> RunDescriber:
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        pass


@runtime_checkable
class DataSetWithSubscriberProtocol(DataSetProtocol, Protocol, Sized):

    def subscribe(self,
                  callback: Callable[[Any, int, Optional[Any]], None],
                  min_wait: int = 0,
                  min_count: int = 1,
                  state: Optional[Any] = None,
                  callback_kwargs: Optional[Mapping[str, Any]] = None
                  ) -> str:
        pass

    def subscribe_from_config(self, name: str) -> str:
        pass

    def unsubscribe_all(self) -> None:
        pass
