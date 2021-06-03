from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
)

import numpy as np
from typing_extensions import Protocol, runtime_checkable

from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
from qcodes.dataset.export_config import DataExportType
from qcodes.dataset.linked_datasets.links import Link

SPECS = List[ParamSpec]
# Transition period type: SpecsOrInterDeps. We will allow both as input to
# the DataSet constructor for a while, then deprecate SPECS and finally remove
# the ParamSpec class
SpecsOrInterDeps = Union[SPECS, InterDependencies_]


if TYPE_CHECKING:

    from .data_set_cache import DataSetCache


@runtime_checkable
class DataSetProtocol(Protocol, Sized):
    def prepare(
        self,
        *,
        snapshot: Dict[Any, Any],
        interdeps: InterDependencies_,
        shapes: Shapes = None,
        parent_datasets: Sequence[Mapping[Any, Any]] = (),
        write_in_background: bool = False,
    ) -> None:
        pass

    @property
    def completed(self) -> bool:
        pass

    # todo do we really need both of these?

    @completed.setter
    def completed(self, value: bool) -> None:
        pass

    def mark_completed(self) -> None:
        pass

    # dataset attributes

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

    @property
    def number_of_results(self) -> int:
        pass

    @property
    def paramspecs(self) -> Dict[str, ParamSpec]:
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def exp_name(self) -> str:
        pass

    @property
    def sample_name(self) -> str:
        pass

    def run_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
        pass

    @property
    def run_timestamp_raw(self) -> Optional[float]:
        pass

    def completed_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
        pass

    @property
    def completed_timestamp_raw(self) -> Optional[float]:
        pass

    # snapshot and metadata

    def add_snapshot(self, snapshot: str, overwrite: bool = False) -> None:
        pass

    @property
    def snapshot_raw(self) -> Optional[str]:
        pass

    def get_metadata(self, tag: str) -> str:
        pass

    def add_metadata(self, tag: str, metadata: Any) -> None:
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        pass

    # dataset description and links

    @property
    def description(self) -> RunDescriber:
        pass

    @property
    def dependent_parameters(self) -> Tuple[ParamSpecBase, ...]:
        pass

    @property
    def parent_dataset_links(self) -> List[Link]:
        pass

    # data related members

    def export(
        self,
        export_type: Optional[Union[DataExportType, str]] = None,
        path: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> None:
        pass

    @property
    def cache(self) -> DataSetCache[DataSetProtocol]:
        pass

    # private members called doing measurement

    def _enqueue_results(self, result_dict: Mapping[ParamSpecBase, np.ndarray]) -> None:
        pass

    def _flush_data_to_database(self, block: bool = False) -> None:
        pass
