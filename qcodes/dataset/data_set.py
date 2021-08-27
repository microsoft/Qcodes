import importlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Sized,
    Tuple,
    Union,
)

import numpy
import pandas as pd

import qcodes
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning.converters import new_to_old, old_to_new
from qcodes.dataset.descriptions.versioning.rundescribertypes import Shapes
from qcodes.dataset.descriptions.versioning.v0 import InterDependencies
from qcodes.dataset.experiment_settings import get_default_experiment_id
from qcodes.dataset.export_config import (
    DataExportType,
    get_data_export_path,
    get_data_export_prefix,
    get_data_export_type,
)
from qcodes.dataset.guids import filter_guids_by_parts, generate_guid, parse_guid
from qcodes.dataset.linked_datasets.links import Link, links_to_str, str_to_links
from qcodes.dataset.sqlite.connection import (
    ConnectionPlus,
    atomic,
    atomic_transaction,
    transaction,
)
from qcodes.dataset.sqlite.database import (
    conn_from_dbpath_or_conn,
    connect,
    get_DB_location,
)
from qcodes.dataset.sqlite.queries import (
    add_meta_data,
    add_parameter,
    completed,
    create_run,
    get_completed_timestamp_from_run_id,
    get_experiment_name_from_experiment_id,
    get_guid_from_run_id,
    get_guids_from_run_spec,
    get_last_experiment,
    get_metadata,
    get_metadata_from_run_id,
    get_parameter_data,
    get_parent_dataset_links,
    get_run_description,
    get_run_timestamp_from_run_id,
    get_runid_from_guid,
    get_sample_name_from_experiment_id,
    mark_run_complete,
    remove_trigger,
    run_exists,
    set_run_timestamp,
    update_parent_datasets,
    update_run_description,
)
from qcodes.dataset.sqlite.query_helpers import (
    VALUE,
    VALUES,
    insert_many_values,
    length,
    one,
    select_one_where,
)
from qcodes.instrument.parameter import _BaseParameter
from qcodes.utils.deprecate import deprecate
from qcodes.utils.helpers import NumpyJSONEncoder

from .data_set_cache import DataSetCacheWithDBBackend
from .descriptions.versioning import serialization as serial
from .exporters.export_info import ExportInfo
from .exporters.export_to_csv import dataframe_to_csv
from .exporters.export_to_pandas import (
    load_to_concatenated_dataframe,
    load_to_dataframe_dict,
)
from .exporters.export_to_xarray import (
    load_to_xarray_dataarray_dict,
    load_to_xarray_dataset,
)
from .subscriber import _Subscriber

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr


log = logging.getLogger(__name__)


# TODO: storing parameters in separate table as an extension (dropping
# the column parametenrs would be much nicer

# TODO: metadata split between well known columns and maybe something else is
# not such a good idea. The problem is if we allow for specific columns then
# how do the user/us know which are metatadata?  I THINK the only sane solution
# is to store JSON in a column called metadata

# TODO: fixix  a subset of metadata that we define well known (and create them)
# i.e. no dynamic creation of metadata columns, but add stuff to
# a json inside a 'metadata' column

array_like_types = (tuple, list, numpy.ndarray)
scalar_res_types = Union[str, complex,
                         numpy.integer, numpy.floating, numpy.complexfloating]
values_type = Union[scalar_res_types, numpy.ndarray,
                    Sequence[scalar_res_types]]
res_type = Tuple[Union[_BaseParameter, str],
                 values_type]
setpoints_type = Sequence[Union[str, _BaseParameter]]
SPECS = List[ParamSpec]
# Transition period type: SpecsOrInterDeps. We will allow both as input to
# the DataSet constructor for a while, then deprecate SPECS and finally remove
# the ParamSpec class
SpecsOrInterDeps = Union[SPECS, InterDependencies_]
ParameterData = Dict[str, Dict[str, numpy.ndarray]]


class CompletedError(RuntimeError):
    pass


class _BackgroundWriter(Thread):
    """
    Write the results from the DataSet's dataqueue in a new thread
    """

    def __init__(self, queue: "Queue[Any]", conn: ConnectionPlus):
        super().__init__(daemon=True)
        self.queue = queue
        self.path = conn.path_to_dbfile
        self.keep_writing = True

    def run(self) -> None:

        self.conn = connect(self.path)

        while self.keep_writing:

            item = self.queue.get()
            if item['keys'] == 'stop':
                self.keep_writing = False
                self.conn.close()
            elif item['keys'] == 'finalize':
                _WRITERS[self.path].active_datasets.remove(item['values'])
            else:
                self.write_results(
                    item['keys'], item['values'], item['table_name'])
            self.queue.task_done()

    def write_results(self, keys: Sequence[str],
                      values: Sequence[List[Any]],
                      table_name: str) -> None:
        insert_many_values(self.conn, table_name, keys, values)

    def shutdown(self) -> None:
        """
        Send a termination signal to the data writing queue, wait for the
        queue to empty and the thread to join.

        If the background writing thread is not alive this will do nothing.
        """
        if self.is_alive():
            self.queue.put({'keys': 'stop', 'values': []})
            self.queue.join()
            self.join()


@dataclass
class _WriterStatus:
    bg_writer: Optional[_BackgroundWriter]
    write_in_background: Optional[bool]
    data_write_queue: "Queue[Any]"
    active_datasets: Set[int]


_WRITERS: Dict[str, _WriterStatus] = {}


class DataSet(Sized):

    # the "persistent traits" are the attributes/properties of the DataSet
    # that are NOT tied to the representation of the DataSet in any particular
    # database
    persistent_traits = ('name', 'guid', 'number_of_results',
                         'parameters', 'paramspecs', 'exp_name', 'sample_name',
                         'completed', 'snapshot', 'run_timestamp_raw',
                         'description', 'completed_timestamp_raw', 'metadata',
                         'dependent_parameters', 'parent_dataset_links',
                         'captured_run_id', 'captured_counter')
    background_sleep_time = 1e-3

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
        """
        Create a new :class:`.DataSet` object. The object can either hold a new run or
        an already existing run. If a ``run_id`` is provided, then an old run is
        looked up, else a new run is created.

        Args:
            path_to_db: path to the sqlite file on disk. If not provided, the
                path will be read from the config.
            run_id: provide this when loading an existing run, leave it
                as None when creating a new run
            conn: connection to the DB; if provided and ``path_to_db`` is
                provided as well, then a ``ValueError`` is raised (this is to
                prevent the possibility of providing a connection to a DB
                file that is different from ``path_to_db``)
            exp_id: the id of the experiment in which to create a new run.
                Ignored if ``run_id`` is provided.
            name: the name of the dataset. Ignored if ``run_id`` is provided.
            specs: paramspecs belonging to the dataset or an ``InterDependencies_``
                object that describes the dataset. Ignored if ``run_id``
                is provided.
            values: values to insert into the dataset. Ignored if ``run_id`` is
                provided.
            metadata: metadata to insert into the dataset. Ignored if ``run_id``
                is provided.
            shapes:
                An optional dict from names of dependent parameters to the shape
                of the data captured as a list of integers. The list is in the
                same order as the interdependencies or paramspecs provided.
                Ignored if ``run_id`` is provided.
            in_memory_cache: Should measured data be keep in memory
                and available as part of the `dataset.cache` object.

        """
        self.conn = conn_from_dbpath_or_conn(conn, path_to_db)

        self._debug = False
        self.subscribers: Dict[str, _Subscriber] = {}
        self._parent_dataset_links: List[Link]
        #: In memory representation of the data in the dataset.
        self._cache: DataSetCacheWithDBBackend = DataSetCacheWithDBBackend(self)
        self._results: List[Dict[str, VALUE]] = []
        self._in_memory_cache = in_memory_cache
        self._export_path: Optional[str] = None

        if run_id is not None:
            if not run_exists(self.conn, run_id):
                raise ValueError(f"Run with run_id {run_id} does not exist in "
                                 f"the database")
            self._run_id = run_id
            self._completed = completed(self.conn, self.run_id)
            run_desc = self._get_run_description_from_db()
            self._rundescriber = run_desc
            self._metadata = get_metadata_from_run_id(self.conn, self.run_id)
            self._started = self.run_timestamp_raw is not None
            self._parent_dataset_links = str_to_links(
                get_parent_dataset_links(self.conn, self.run_id)
            )
            self._export_info = ExportInfo.from_str(
                self.metadata.get("export_info", "")
            )
        else:
            # Actually perform all the side effects needed for the creation
            # of a new dataset. Note that a dataset is created (in the DB)
            # with no parameters; they are written to disk when the dataset
            # is marked as started
            if exp_id is None:
                exp_id = get_default_experiment_id(self.conn)
            name = name or "dataset"
            _, run_id, __ = create_run(self.conn, exp_id, name,
                                       generate_guid(),
                                       parameters=None,
                                       values=values,
                                       metadata=metadata)
            # this is really the UUID (an ever increasing count in the db)
            self._run_id = run_id
            self._completed = False
            self._started = False

            if isinstance(specs, InterDependencies_):
                interdeps = specs
            elif specs is not None:
                interdeps = old_to_new(InterDependencies(*specs))
            else:
                interdeps = InterDependencies_()

            self.set_interdependencies(
                interdeps=interdeps,
                shapes=shapes)

            self._metadata = get_metadata_from_run_id(self.conn, self.run_id)
            self._parent_dataset_links = []
            self._export_info = ExportInfo({})

        if _WRITERS.get(self.path_to_db) is None:
            queue: "Queue[Any]" = Queue()
            ws: _WriterStatus = _WriterStatus(
                bg_writer=None,
                write_in_background=None,
                data_write_queue=queue,
                active_datasets=set())
            _WRITERS[self.path_to_db] = ws

    def prepare(
        self,
        *,
        snapshot: Mapping[Any, Any],
        interdeps: InterDependencies_,
        shapes: Shapes = None,
        parent_datasets: Sequence[Mapping[Any, Any]] = (),
        write_in_background: bool = False,
    ) -> None:

        self.add_snapshot(json.dumps({"station": snapshot}, cls=NumpyJSONEncoder))

        if interdeps == InterDependencies_():
            raise RuntimeError("No parameters supplied")

        self.set_interdependencies(interdeps, shapes)
        links = [Link(head=self.guid, **pdict) for pdict in parent_datasets]
        self.parent_dataset_links = links
        self.mark_started(start_bg_writer=write_in_background)

    @property
    def cache(self) -> DataSetCacheWithDBBackend:
        return self._cache

    @property
    def run_id(self) -> int:
        return self._run_id

    @property
    def captured_run_id(self) -> int:
        return select_one_where(self.conn, "runs",
                                "captured_run_id", "run_id", self.run_id)

    @property
    def path_to_db(self) -> str:
        return self.conn.path_to_dbfile

    @property
    def name(self) -> str:
        return select_one_where(self.conn, "runs",
                                "name", "run_id", self.run_id)

    @property
    def table_name(self) -> str:
        return select_one_where(self.conn, "runs",
                                "result_table_name", "run_id", self.run_id)

    @property
    def guid(self) -> str:
        return get_guid_from_run_id(self.conn, self.run_id)

    @property
    def snapshot(self) -> Optional[Dict[str, Any]]:
        """Snapshot of the run as dictionary (or None)"""
        snapshot_json = self.snapshot_raw
        if snapshot_json is not None:
            return json.loads(snapshot_json)
        else:
            return None

    @property
    def _snapshot_raw(self) -> Optional[str]:
        """Snapshot of the run as a JSON-formatted string (or None)"""
        return select_one_where(self.conn, "runs", "snapshot",
                                "run_id", self.run_id)

    @property
    def snapshot_raw(self) -> Optional[str]:
        """Snapshot of the run as a JSON-formatted string (or None)"""
        return self._snapshot_raw

    @property
    def number_of_results(self) -> int:
        sql = f'SELECT COUNT(*) FROM "{self.table_name}"'
        cursor = atomic_transaction(self.conn, sql)
        return one(cursor, 'COUNT(*)')

    @property
    def counter(self) -> int:
        return select_one_where(self.conn, "runs",
                                "result_counter", "run_id", self.run_id)

    @property
    def captured_counter(self) -> int:
        return select_one_where(self.conn, "runs",
                                "captured_counter", "run_id", self.run_id)

    @property
    def parameters(self) -> str:
        if self.pristine:
            psnames = [ps.name for ps in self.description.interdeps.paramspecs]
            return ','.join(psnames)
        else:
            return select_one_where(self.conn, "runs",
                                    "parameters", "run_id", self.run_id)

    @property
    def paramspecs(self) -> Dict[str, ParamSpec]:
        return {ps.name: ps
                for ps in self.get_parameters()}

    @property
    def dependent_parameters(self) -> Tuple[ParamSpecBase, ...]:
        """
        Return all the parameters that explicitly depend on other parameters
        """
        return tuple(self._rundescriber.interdeps.dependencies.keys())

    @property
    def exp_id(self) -> int:
        return select_one_where(self.conn, "runs",
                                "exp_id", "run_id", self.run_id)

    @property
    def exp_name(self) -> str:
        return get_experiment_name_from_experiment_id(self.conn, self.exp_id)

    @property
    def sample_name(self) -> str:
        return get_sample_name_from_experiment_id(self.conn, self.exp_id)

    @property
    def run_timestamp_raw(self) -> Optional[float]:
        """
        Returns run timestamp as number of seconds since the Epoch

        The run timestamp is the moment when the measurement for this run
        started.
        """
        return get_run_timestamp_from_run_id(self.conn, self.run_id)

    @property
    def description(self) -> RunDescriber:
        return self._rundescriber

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def parent_dataset_links(self) -> List[Link]:
        """
        Return a list of Link objects. Each Link object describes a link from
        this dataset to one of its parent datasets
        """
        return self._parent_dataset_links

    @parent_dataset_links.setter
    def parent_dataset_links(self, links: List[Link]) -> None:
        """
        Assign one or more links to parent datasets to this dataset. It is an
        error to assign links to a non-pristine dataset

        Args:
            links: The links to assign to this dataset
        """
        if not self.pristine:
            raise RuntimeError('Can not set parent dataset links on a dataset '
                               'that has been started.')

        if not all(isinstance(link, Link) for link in links):
            raise ValueError('Invalid input. Did not receive a list of Links')

        for link in links:
            if link.head != self.guid:
                raise ValueError(
                    'Invalid input. All links must point to this dataset. '
                    'Got link(s) with head(s) pointing to another dataset.')

        self._parent_dataset_links = links

    @property
    def _writer_status(self) -> _WriterStatus:
        return _WRITERS[self.path_to_db]

    def the_same_dataset_as(self, other: 'DataSet') -> bool:
        """
        Check if two datasets correspond to the same run by comparing
        all their persistent traits. Note that this method
        does not compare the data itself.

        This function raises if the GUIDs match but anything else doesn't

        Args:
            other: the dataset to compare self to
        """

        if not isinstance(other, DataSet):
            return False

        guids_match = self.guid == other.guid

        # note that the guid is in itself a persistent trait of the DataSet.
        # We therefore do not need to handle the case of guids not equal
        # but all persistent traits equal, as this is not possible.
        # Thus, if all persistent traits are the same we can safely return True
        for attr in DataSet.persistent_traits:
            if getattr(self, attr) != getattr(other, attr):
                if guids_match:
                    raise RuntimeError('Critical inconsistency detected! '
                                       'The two datasets have the same GUID, '
                                       f'but their "{attr}" differ.')
                else:
                    return False

        return True

    def run_timestamp(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
        """
        Returns run timestamp in a human-readable format

        The run timestamp is the moment when the measurement for this run
        started. If the run has not yet been started, this function returns
        None.

        Consult with :func:`time.strftime` for information about the format.
        """
        if self.run_timestamp_raw is None:
            return None
        else:
            return time.strftime(fmt, time.localtime(self.run_timestamp_raw))

    @property
    def completed_timestamp_raw(self) -> Optional[float]:
        """
        Returns timestamp when measurement run was completed
        as number of seconds since the Epoch

        If the run (or the dataset) is not completed, then returns None.
        """
        return get_completed_timestamp_from_run_id(self.conn, self.run_id)

    def completed_timestamp(self,
                            fmt: str = "%Y-%m-%d %H:%M:%S") -> Optional[str]:
        """
        Returns timestamp when measurement run was completed
        in a human-readable format

        If the run (or the dataset) is not completed, then returns None.

        Consult with ``time.strftime`` for information about the format.
        """
        completed_timestamp_raw = self.completed_timestamp_raw

        if completed_timestamp_raw:
            completed_timestamp: Optional[str] = time.strftime(
                fmt, time.localtime(completed_timestamp_raw))
        else:
            completed_timestamp = None

        return completed_timestamp

    def _get_run_description_from_db(self) -> RunDescriber:
        """
        Look up the run_description from the database
        """
        desc_str = get_run_description(self.conn, self.run_id)
        return serial.from_json_to_current(desc_str)

    def toggle_debug(self) -> None:
        """
        Toggle debug mode, if debug mode is on all the queries made are
        echoed back.
        """
        self._debug = not self._debug
        self.conn.close()
        self.conn = connect(self.path_to_db, self._debug)

    def set_interdependencies(self,
                              interdeps: InterDependencies_,
                              shapes: Shapes = None) -> None:
        """
        Set the interdependencies object (which holds all added
        parameters and their relationships) of this dataset and
        optionally the shapes object that holds information about
        the shape of the data to be measured.
        """
        if not isinstance(interdeps, InterDependencies_):
            raise TypeError('Wrong input type. Expected InterDepencies_, '
                            f'got {type(interdeps)}')

        if not self.pristine:
            mssg = ('Can not set interdependencies on a DataSet that has '
                    'been started.')
            raise RuntimeError(mssg)
        self._rundescriber = RunDescriber(interdeps, shapes=shapes)

    def get_parameters(self) -> SPECS:
        old_interdeps = new_to_old(self.description.interdeps)
        return list(old_interdeps.paramspecs)

    def add_metadata(self, tag: str, metadata: Any) -> None:
        """
        Adds metadata to the :class:`.DataSet`. The metadata is stored under the
        provided tag. Note that None is not allowed as a metadata value, and the
        tag has to be a valid python identified (e.g. containing alphanumeric
        characters and underscores).

        Args:
            tag: represents the key in the metadata dictionary
            metadata: actual metadata
        """

        self._metadata[tag] = metadata
        # `add_meta_data` is not atomic by itself, hence using `atomic`
        with atomic(self.conn) as conn:
            add_meta_data(conn, self.run_id, {tag: metadata})

    def add_snapshot(self, snapshot: str, overwrite: bool = False) -> None:
        """
        Adds a snapshot to this run

        Args:
            snapshot: the raw JSON dump of the snapshot
            overwrite: force overwrite an existing snapshot
        """
        if self.snapshot is None or overwrite:
            add_meta_data(self.conn, self.run_id, {'snapshot': snapshot})
        elif self.snapshot is not None and not overwrite:
            log.warning('This dataset already has a snapshot. Use overwrite'
                        '=True to overwrite that')

    @property
    def pristine(self) -> bool:
        """
        Is this :class:`.DataSet` pristine? A pristine :class:`.DataSet` has not yet been started,
        meaning that parameters can still be added and removed, but results
        can not be added.
        """
        return not(self._started or self._completed)

    @property
    def running(self) -> bool:
        """
        Is this :class:`.DataSet` currently running? A running :class:`.DataSet` has been started,
        but not yet completed.
        """
        return self._started and not self._completed

    @property
    def started(self) -> bool:
        """
        Has this :class:`.DataSet` been started? A :class:`.DataSet` not started can not have any
        results added to it.
        """
        return self._started

    @property
    def completed(self) -> bool:
        """
        Is this :class:`.DataSet` completed? A completed :class:`.DataSet` may not be modified in
        any way.
        """
        return self._completed

    @completed.setter
    def completed(self, value: bool) -> None:
        self._completed = value
        if value:
            mark_run_complete(self.conn, self.run_id)

    def mark_started(self, start_bg_writer: bool = False) -> None:
        """
        Mark this :class:`.DataSet` as started. A :class:`.DataSet` that has been started can not
        have its parameters modified.

        Calling this on an already started :class:`.DataSet` is a NOOP.

        Args:
            start_bg_writer: If True, the add_results method will write to the
                database in a separate thread.
        """
        if not self._started:
            self._perform_start_actions(start_bg_writer=start_bg_writer)
            self._started = True

    def _perform_start_actions(self, start_bg_writer: bool) -> None:
        """
        Perform the actions that must take place once the run has been started
        """
        paramspecs = new_to_old(self._rundescriber.interdeps).paramspecs

        for spec in paramspecs:
            add_parameter(self.conn, self.table_name, spec)

        desc_str = serial.to_json_for_storage(self.description)

        update_run_description(self.conn, self.run_id, desc_str)

        set_run_timestamp(self.conn, self.run_id)

        pdl_str = links_to_str(self._parent_dataset_links)
        update_parent_datasets(self.conn, self.run_id, pdl_str)

        writer_status = self._writer_status

        write_in_background_status = writer_status.write_in_background
        if write_in_background_status is not None and write_in_background_status != start_bg_writer:
            raise RuntimeError("All datasets written to the same database must "
                               "be written either in the background or in the "
                               "main thread. You cannot mix.")
        if start_bg_writer:
            writer_status.write_in_background = True
            if writer_status.bg_writer is None:
                writer_status.bg_writer = _BackgroundWriter(
                    writer_status.data_write_queue, self.conn)
            if not writer_status.bg_writer.is_alive():
                writer_status.bg_writer.start()
        else:
            writer_status.write_in_background = False

        writer_status.active_datasets.add(self.run_id)

    def mark_completed(self) -> None:
        """
        Mark :class:`.DataSet` as complete and thus read only and notify the subscribers
        """
        if self.completed:
            return
        if self.pristine:
            raise RuntimeError('Can not mark DataSet as complete before it '
                               'has been marked as started.')

        self._perform_completion_actions()
        self.completed = True

    def _perform_completion_actions(self) -> None:
        """
        Perform the necessary clean-up
        """
        for sub in self.subscribers.values():
            sub.done_callback()
        self._ensure_dataset_written()

    def add_results(self, results: Sequence[Mapping[str, VALUE]]) -> None:
        """
        Adds a sequence of results to the :class:`.DataSet`.

        Args:
            results: list of name-value dictionaries where each dictionary
                provides the values for the parameters in that result. If some
                parameters are missing the corresponding values are assumed
                to be None

        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this :class:`.DataSet`.

        It is an error to add results to a completed :class:`.DataSet`.
        """

        self._raise_if_not_writable()

        expected_keys = frozenset.union(*(frozenset(d) for d in results))
        values = [[d.get(k, None) for k in expected_keys] for d in results]

        writer_status = self._writer_status

        if writer_status.write_in_background:
            item = {'keys': list(expected_keys), 'values': values,
                    "table_name": self.table_name}
            writer_status.data_write_queue.put(item)
        else:
            insert_many_values(self.conn, self.table_name, list(expected_keys),
                               values)

    def _raise_if_not_writable(self) -> None:
        if self.pristine:
            raise RuntimeError('This DataSet has not been marked as started. '
                               'Please mark the DataSet as started before '
                               'adding results to it.')
        if self.completed:
            raise CompletedError('This DataSet is complete, no further '
                                 'results can be added to it.')

    def _ensure_dataset_written(self) -> None:
        writer_status = self._writer_status

        if writer_status.write_in_background:
            writer_status.data_write_queue.put(
                {'keys': 'finalize', 'values': self.run_id})
            while self.run_id in writer_status.active_datasets:
                time.sleep(self.background_sleep_time)
        else:
            if self.run_id in writer_status.active_datasets:
                writer_status.active_datasets.remove(self.run_id)
        if len(writer_status.active_datasets) == 0:
            writer_status.write_in_background = None
            if writer_status.bg_writer is not None:
                writer_status.bg_writer.shutdown()
                writer_status.bg_writer = None

    @staticmethod
    def _validate_parameters(*params: Union[str, ParamSpec, _BaseParameter]
                             ) -> List[str]:
        """
        Validate that the provided parameters have a name and return those
        names as a list.
        The Parameters may be a mix of strings, ParamSpecs or ordinary
        QCoDeS parameters.
        """

        valid_param_names = []
        for maybeParam in params:
            if isinstance(maybeParam, str):
                valid_param_names.append(maybeParam)
                continue
            else:
                try:
                    maybeParam = maybeParam.name
                except Exception as e:
                    raise ValueError(
                        "This parameter does not have  a name") from e
                valid_param_names.append(maybeParam)
        return valid_param_names

    def get_parameter_data(
            self,
            *params: Union[str, ParamSpec, _BaseParameter],
            start: Optional[int] = None,
            end: Optional[int] = None) -> ParameterData:
        """
        Returns the values stored in the :class:`.DataSet` for the specified parameters
        and their dependencies. If no parameters are supplied the values will
        be returned for all parameters that are not them self dependencies.

        The values are returned as a dictionary with names of the requested
        parameters as keys and values consisting of dictionaries with the
        names of the parameters and its dependencies as keys and numpy arrays
        of the data as values. If the dataset has a shape recorded
        in its metadata and the number of datapoints recorded matches the
        expected number of points the data will be returned as numpy arrays
        in this shape. If there are less datapoints recorded than expected
        from the metadata the dataset will be returned as is. This could happen
        if you call `get_parameter_data` on an incomplete dataset. See
        :py:meth:`dataset.cache.data <.DataSetCache.data>` for an implementation that
        returns the data with the expected shape using `NaN` or zeros as
        placeholders.

        If there are more datapoints than expected the dataset will be returned
        as is and a warning raised.

        If some of the parameters are stored as arrays
        the remaining parameters are expanded to the same shape as these.

        If provided, the start and end arguments select a range of results
        by result count (index). If the range is empty - that is, if the end is
        less than or equal to the start, or if start is after the current end
        of the :class:`.DataSet` – then a list of empty arrays is returned.

        Args:
            *params: string parameter names, QCoDeS Parameter objects, and
                ParamSpec objects. If no parameters are supplied data for
                all parameters that are not a dependency of another
                parameter will be returned.
            start: start value of selection range (by result count); ignored
                if None
            end: end value of selection range (by results count); ignored if
                None

        Returns:
            Dictionary from requested parameters to Dict of parameter names
            to numpy arrays containing the data points of type numeric,
            array or string.
        """
        if len(params) == 0:
            valid_param_names = [ps.name
                                 for ps in self._rundescriber.interdeps.non_dependencies]
        else:
            valid_param_names = self._validate_parameters(*params)
        return get_parameter_data(self.conn, self.table_name,
                                  valid_param_names, start, end)

    def to_pandas_dataframe_dict(self,
                                 *params: Union[str,
                                                ParamSpec,
                                                _BaseParameter],
                                 start: Optional[int] = None,
                                 end: Optional[int] = None) ->\
            Dict[str, "pd.DataFrame"]:
        """
        Returns the values stored in the :class:`.DataSet` for the specified parameters
        and their dependencies as a dict of :py:class:`pandas.DataFrame` s
        Each element in the dict is indexed by the names of the requested
        parameters.

        Each DataFrame contains a column for the data and is indexed by a
        :py:class:`pandas.MultiIndex` formed from all the setpoints
        of the parameter.

        If no parameters are supplied data will be be
        returned for all parameters in the :class:`.DataSet` that are not them self
        dependencies of other parameters.

        If provided, the start and end arguments select a range of results
        by result count (index). If the range is empty - that is, if the end is
        less than or equal to the start, or if start is after the current end
        of the :class:`.DataSet` – then a dict of empty :py:class:`pandas.DataFrame` s is
        returned.

        Args:
            *params: string parameter names, QCoDeS Parameter objects, and
                ParamSpec objects. If no parameters are supplied data for
                all parameters that are not a dependency of another
                parameter will be returned.
            start: start value of selection range (by result count); ignored
                if None
            end: end value of selection range (by results count); ignored if
                None

        Returns:
            Dictionary from requested parameter names to
            :py:class:`pandas.DataFrame` s with the requested parameter as
            a column and a indexed by a :py:class:`pandas.MultiIndex` formed
            by the dependencies.
        """
        datadict = self.get_parameter_data(*params,
                                           start=start,
                                           end=end)
        dfs_dict = load_to_dataframe_dict(datadict)
        return dfs_dict

    @deprecate(reason='This method will be removed due to inconcise naming, please '
               'use the renamed method to_pandas_dataframe_dict',
               alternative='to_pandas_dataframe_dict')
    def get_data_as_pandas_dataframe(self,
                                     *params: Union[str,
                                                    ParamSpec,
                                                    _BaseParameter],
                                     start: Optional[int] = None,
                                     end: Optional[int] = None) -> \
            Dict[str, "pd.DataFrame"]:
        """
        Returns the values stored in the :class:`.DataSet` for the specified parameters
        and their dependencies as a dict of :py:class:`pandas.DataFrame` s
        Each element in the dict is indexed by the names of the requested
        parameters.

        Each DataFrame contains a column for the data and is indexed by a
        :py:class:`pandas.MultiIndex` formed from all the setpoints
        of the parameter.

        If no parameters are supplied data will be be
        returned for all parameters in the :class:`.DataSet` that are not them self
        dependencies of other parameters.

        If provided, the start and end arguments select a range of results
        by result count (index). If the range is empty - that is, if the end is
        less than or equal to the start, or if start is after the current end
        of the :class:`.DataSet` – then a dict of empty :py:class:`pandas.DataFrame` s is
        returned.

        Args:
            *params: string parameter names, QCoDeS Parameter objects, and
                ParamSpec objects. If no parameters are supplied data for
                all parameters that are not a dependency of another
                parameter will be returned.
            start: start value of selection range (by result count); ignored
                if None
            end: end value of selection range (by results count); ignored if
                None

        Returns:
            Dictionary from requested parameter names to
            :py:class:`pandas.DataFrame` s with the requested parameter as
            a column and a indexed by a :py:class:`pandas.MultiIndex` formed
            by the dependencies.
        """
        return self.to_pandas_dataframe_dict(*params, start=start, end=end)

    def to_pandas_dataframe(self,
                            *params: Union[str,
                                           ParamSpec,
                                           _BaseParameter],
                            start: Optional[int] = None,
                            end: Optional[int] = None) -> "pd.DataFrame":
        """
        Returns the values stored in the :class:`.DataSet` for the specified parameters
        and their dependencies as a concatenated :py:class:`pandas.DataFrame` s

        The DataFrame contains a column for the data and is indexed by a
        :py:class:`pandas.MultiIndex` formed from all the setpoints
        of the parameter.

        If no parameters are supplied data will be be
        returned for all parameters in the :class:`.DataSet` that are not them self
        dependencies of other parameters.

        If provided, the start and end arguments select a range of results
        by result count (index). If the range is empty - that is, if the end is
        less than or equal to the start, or if start is after the current end
        of the :class:`.DataSet` – then a dict of empty :py:class:`pandas.DataFrame` s is
        returned.

        Args:
            *params: string parameter names, QCoDeS Parameter objects, and
                ParamSpec objects. If no parameters are supplied data for
                all parameters that are not a dependency of another
                parameter will be returned.
            start: start value of selection range (by result count); ignored
                if None
            end: end value of selection range (by results count); ignored if
                None

        Returns:
            :py:class:`pandas.DataFrame` s with the requested parameter as
            a column and a indexed by a :py:class:`pandas.MultiIndex` formed
            by the dependencies.

        Example:
            Return a pandas DataFrame with
                df = ds.to_pandas_dataframe()
        """
        datadict = self.get_parameter_data(*params,
                                           start=start,
                                           end=end)
        return load_to_concatenated_dataframe(datadict)

    def to_xarray_dataarray_dict(self,
                                 *params: Union[str,
                                                ParamSpec,
                                                _BaseParameter],
                                 start: Optional[int] = None,
                                 end: Optional[int] = None) -> \
            Dict[str, "xr.DataArray"]:
        """
        Returns the values stored in the :class:`.DataSet` for the specified parameters
        and their dependencies as a dict of :py:class:`xr.DataArray` s
        Each element in the dict is indexed by the names of the requested
        parameters.

        If no parameters are supplied data will be be
        returned for all parameters in the :class:`.DataSet` that are not them self
        dependencies of other parameters.

        If provided, the start and end arguments select a range of results
        by result count (index). If the range is empty - that is, if the end is
        less than or equal to the start, or if start is after the current end
        of the :class:`.DataSet` – then a dict of empty :py:class:`xr.DataArray` s is
        returned.

        The dependent parameters of the Dataset are normally used as coordinates of the
        XArray dataframe. However if non unique values are found for the dependent parameter
        values we will fall back to using an index as coordinates.

        Args:
            *params: string parameter names, QCoDeS Parameter objects, and
                ParamSpec objects. If no parameters are supplied data for
                all parameters that are not a dependency of another
                parameter will be returned.
            start: start value of selection range (by result count); ignored
                if None
            end: end value of selection range (by results count); ignored if
                None

        Returns:
            Dictionary from requested parameter names to :py:class:`xr.DataArray` s
            with the requested parameter(s) as a column(s) and coordinates
            formed by the dependencies.

        Example:
            Return a dict of xr.DataArray with

                dataarray_dict = ds.to_xarray_dataarray_dict()
        """
        data = self.get_parameter_data(*params,
                                       start=start,
                                       end=end)
        datadict = load_to_xarray_dataarray_dict(self, data)

        return datadict

    def to_xarray_dataset(self, *params: Union[str,
                                               ParamSpec,
                                               _BaseParameter],
                          start: Optional[int] = None,
                          end: Optional[int] = None) -> "xr.Dataset":
        """
        Returns the values stored in the :class:`.DataSet` for the specified parameters
        and their dependencies as a :py:class:`xr.Dataset` object.

        If no parameters are supplied data will be be
        returned for all parameters in the :class:`.DataSet` that are not then self
        dependencies of other parameters.

        If provided, the start and end arguments select a range of results
        by result count (index). If the range is empty - that is, if the end is
        less than or equal to the start, or if start is after the current end
        of the :class:`.DataSet` – then a empty :py:class:`xr.Dataset` s is
        returned.

        The dependent parameters of the Dataset are normally used as coordinates of the
        XArray dataframe. However if non unique values are found for the dependent parameter
        values we will fall back to using an index as coordinates.

        Args:
            *params: string parameter names, QCoDeS Parameter objects, and
                ParamSpec objects. If no parameters are supplied data for
                all parameters that are not a dependency of another
                parameter will be returned.
            start: start value of selection range (by result count); ignored
                if None
            end: end value of selection range (by results count); ignored if
                None

        Returns:
            :py:class:`xr.Dataset` with the requested parameter(s) data as
            :py:class:`xr.DataArray` s and coordinates formed by the dependencies.

        Example:
            Return a concatenated xr.Dataset with

                xds = ds.to_xarray_dataset()
        """
        data = self.get_parameter_data(*params,
                                       start=start,
                                       end=end)

        return load_to_xarray_dataset(self, data)

    def write_data_to_text_file(self, path: str,
                                single_file: bool = False,
                                single_file_name: Optional[str] = None) -> None:
        """
        An auxiliary function to export data to a text file. When the data with more
        than one dependent variables, say "y(x)" and "z(x)", is concatenated to a single file
        it reads:

                    x1  y1(x1)  z1(x1)
                    x2  y2(x2)  z2(x2)
                    ..    ..      ..
                    xN  yN(xN)  zN(xN)

        For each new independent variable, say "k", the expansion is in the y-axis:

                    x1  y1(x1)  z1(x1)
                    x2  y2(x2)  z2(x2)
                    ..    ..      ..
                    xN  yN(xN)  zN(xN)
                    k1  y1(k1)  z1(k1)
                    k2  y2(k2)  z2(k2)
                    ..    ..      ..
                    kN  yN(kN)  zN(kN)

        Args:
            path: User defined path where the data to be exported
            single_file: If true, merges the data of same length of multiple
                         dependent parameters to a single file.
            single_file_name: User defined name for the data to be concatenated.
                              If no extension is passed (.dat, .csv or .txt),
                              .dat is automatically appended.

        Raises:
            DataLengthException: If the data of multiple parameters have not same
                                 length and wanted to be merged in a single file.
            DataPathException: If the data of multiple parameters are wanted to be merged
                               in a single file but no filename provided.
        """
        dfdict = self.to_pandas_dataframe_dict()
        dataframe_to_csv(
            dfdict=dfdict,
            path=path,
            single_file=single_file,
            single_file_name=single_file_name,
        )

    def subscribe(self,
                  callback: Callable[[Any, int, Optional[Any]], None],
                  min_wait: int = 0,
                  min_count: int = 1,
                  state: Optional[Any] = None,
                  callback_kwargs: Optional[Mapping[str, Any]] = None
                  ) -> str:
        subscriber_id = uuid.uuid4().hex
        subscriber = _Subscriber(self, subscriber_id, callback, state,
                                 min_wait, min_count, callback_kwargs)
        self.subscribers[subscriber_id] = subscriber
        subscriber.start()
        return subscriber_id

    def subscribe_from_config(self, name: str) -> str:
        """
        Subscribe a subscriber defined in the `qcodesrc.json` config file to
        the data of this :class:`.DataSet`. The definition can be found at
        ``subscription.subscribers`` in the ``qcodesrc.json`` config file.

        Args:
            name: identifier of the subscriber. Equal to the key of the entry
                in ``qcodesrc.json::subscription.subscribers``.
        """
        subscribers = qcodes.config.subscription.subscribers
        try:
            subscriber_info = getattr(subscribers, name)
        # the dot dict behind the config does not convert the error and
        # actually raises a `KeyError`
        except (AttributeError, KeyError):
            keys = ','.join(subscribers.keys())
            raise RuntimeError(
                f'subscribe_from_config: failed to subscribe "{name}" to '
                f'DataSet from list of subscribers in `qcodesrc.json` '
                f'(subscriptions.subscribers). Chose one of: {keys}')
        # get callback from string
        parts = subscriber_info.factory.split('.')
        import_path, type_name = '.'.join(parts[:-1]), parts[-1]
        module = importlib.import_module(import_path)
        factory = getattr(module, type_name)

        kwargs = {k: v for k, v in subscriber_info.subscription_kwargs.items()}
        kwargs['callback'] = factory(self, **subscriber_info.factory_kwargs)
        kwargs['state'] = {}
        return self.subscribe(**kwargs)

    def unsubscribe(self, uuid: str) -> None:
        """
        Remove subscriber with the provided uuid
        """
        with atomic(self.conn) as conn:
            sub = self.subscribers[uuid]
            remove_trigger(conn, sub.trigger_id)
            sub.schedule_stop()
            sub.join()
            del self.subscribers[uuid]

    def unsubscribe_all(self) -> None:
        """
        Remove all subscribers
        """
        sql = "select * from sqlite_master where type = 'trigger';"
        triggers = atomic_transaction(self.conn, sql).fetchall()
        with atomic(self.conn) as conn:
            for trigger in triggers:
                remove_trigger(conn, trigger['name'])
            for sub in self.subscribers.values():
                sub.schedule_stop()
                sub.join()
            self.subscribers.clear()

    def get_metadata(self, tag: str) -> str:
        return get_metadata(self.conn, tag, self.table_name)

    def __len__(self) -> int:
        return length(self.conn, self.table_name)

    def __repr__(self) -> str:
        out = []
        heading = f"{self.name} #{self.run_id}@{self.path_to_db}"
        out.append(heading)
        out.append("-" * len(heading))
        ps = self.get_parameters()
        if len(ps) > 0:
            for p in ps:
                out.append(f"{p.name} - {p.type}")

        return "\n".join(out)

    def _enqueue_results(
            self, result_dict: Mapping[ParamSpecBase, numpy.ndarray]) -> None:
        """
        Enqueue the results into self._results

        Before we can enqueue the results, all values of the results dict
        must have the same length. We enqueue each parameter tree separately,
        effectively mimicking making one call to add_results per parameter
        tree.

        Deal with 'numeric' type parameters. If a 'numeric' top level parameter
        has non-scalar shape, it must be unrolled into a list of dicts of
        single values (database).
        """
        self._raise_if_not_writable()
        interdeps = self._rundescriber.interdeps

        toplevel_params = (set(interdeps.dependencies)
                           .intersection(set(result_dict)))
        if self._in_memory_cache:
            new_results: Dict[str, Dict[str, numpy.ndarray]] = {}
        for toplevel_param in toplevel_params:
            inff_params = set(interdeps.inferences.get(toplevel_param, ()))
            deps_params = set(interdeps.dependencies.get(toplevel_param, ()))
            all_params = (inff_params
                          .union(deps_params)
                          .union({toplevel_param}))

            if self._in_memory_cache:
                new_results[toplevel_param.name] = {}
                new_results[toplevel_param.name][toplevel_param.name] = self._reshape_array_for_cache(
                    toplevel_param,
                    result_dict[toplevel_param]
                )
                for param in all_params:
                    if param is not toplevel_param:
                        new_results[toplevel_param.name][param.name] = self._reshape_array_for_cache(
                            param,
                            result_dict[param]
                        )

            if toplevel_param.type == 'array':
                res_list = self._finalize_res_dict_array(
                    result_dict, all_params)
            elif toplevel_param.type in ('numeric', 'text', 'complex'):
                res_list = self._finalize_res_dict_numeric_text_or_complex(
                    result_dict, toplevel_param,
                    inff_params, deps_params)
            else:
                res_dict: Dict[str, VALUE] = {ps.name: result_dict[ps] for ps in all_params}
                res_list = [res_dict]
            self._results += res_list

        # Finally, handle standalone parameters

        standalones = (set(interdeps.standalones)
                       .intersection(set(result_dict)))

        if standalones:
            stdln_dict = {st: result_dict[st] for st in standalones}
            self._results += self._finalize_res_dict_standalones(stdln_dict)
            if self._in_memory_cache:
                for st in standalones:
                    new_results[st.name] = {
                        st.name: self._reshape_array_for_cache(st, result_dict[st])
                    }

        if self._in_memory_cache:
            self.cache.add_data(new_results)

    @staticmethod
    def _reshape_array_for_cache(
            param: ParamSpecBase,
            param_data: numpy.ndarray
    ) -> numpy.ndarray:
        """
        Shape cache data so it matches data read from database.
        This means:

        - Add an extra singleton dim to array data
        - flatten non array data into a linear array.
        """
        param_data = numpy.atleast_1d(param_data)
        if param.type == "array":
            new_data = numpy.reshape(
                param_data,
                (1,) + param_data.shape
            )
        else:
            new_data = param_data.ravel()
        return new_data

    @staticmethod
    def _finalize_res_dict_array(
            result_dict: Mapping[ParamSpecBase, values_type],
            all_params: Set[ParamSpecBase]) -> List[Dict[str, VALUE]]:
        """
        Make a list of res_dicts out of the results for a 'array' type
        parameter. The results are assumed to already have been validated for
        type and shape
        """

        def reshaper(val: Any, ps: ParamSpecBase) -> VALUE:
            paramtype = ps.type
            if paramtype == 'numeric':
                return float(val)
            elif paramtype == 'text':
                return str(val)
            elif paramtype == 'complex':
                return complex(val)
            elif paramtype == 'array':
                if val.shape:
                    return val
                else:
                    return numpy.reshape(val, (1,))
            else:
                raise ValueError(f'Cannot handle unknown paramtype '
                                 f'{paramtype!r} of {ps!r}.')

        res_dict = {ps.name: reshaper(result_dict[ps], ps)
                    for ps in all_params}

        return [res_dict]

    @staticmethod
    def _finalize_res_dict_numeric_text_or_complex(
            result_dict: Mapping[ParamSpecBase, numpy.ndarray],
            toplevel_param: ParamSpecBase,
            inff_params: Set[ParamSpecBase],
            deps_params: Set[ParamSpecBase]) -> List[Dict[str, VALUE]]:
        """
        Make a res_dict in the format expected by DataSet.add_results out
        of the results for a 'numeric' or text type parameter. This includes
        replicating and unrolling values as needed and also handling the corner
        case of np.array(1) kind of values
        """

        res_list: List[Dict[str, VALUE]] = []
        all_params = inff_params.union(deps_params).union({toplevel_param})

        t_map = {'numeric': float, 'text': str, 'complex': complex}

        toplevel_shape = result_dict[toplevel_param].shape
        if toplevel_shape == ():
            # In the case of a single value, life is reasonably simple
            res_list = [{ps.name: t_map[ps.type](result_dict[ps])
                         for ps in all_params}]
        else:
            # We first massage all values into np.arrays of the same
            # shape
            flat_results: Dict[str, numpy.ndarray] = {}

            toplevel_val = result_dict[toplevel_param]
            flat_results[toplevel_param.name] = toplevel_val.ravel()
            N = len(flat_results[toplevel_param.name])
            for dep in deps_params:
                if result_dict[dep].shape == ():
                    flat_results[dep.name] = numpy.repeat(result_dict[dep], N)
                else:
                    flat_results[dep.name] = result_dict[dep].ravel()
            for inff in inff_params:
                if numpy.shape(result_dict[inff]) == ():
                    flat_results[inff.name] = numpy.repeat(result_dict[dep], N)
                else:
                    flat_results[inff.name] = result_dict[inff].ravel()

            # And then put everything into the list

            res_list = [{p.name: flat_results[p.name][ind] for p in all_params}
                        for ind in range(N)]

        return res_list

    @staticmethod
    def _finalize_res_dict_standalones(
            result_dict: Mapping[ParamSpecBase, numpy.ndarray]
    ) -> List[Dict[str, VALUE]]:
        """
        Massage all standalone parameters into the correct shape
        """
        res_list: List[Dict[str, VALUE]] = []
        for param, value in result_dict.items():
            if param.type == 'text':
                if value.shape:
                    res_list += [{param.name: str(val)} for val in value]
                else:
                    res_list += [{param.name: str(value)}]
            elif param.type == 'numeric':
                if value.shape:
                    res_list += [{param.name: number} for number in value]
                else:
                    res_list += [{param.name: float(value)}]
            elif param.type == 'complex':
                if value.shape:
                    res_list += [{param.name: number} for number in value]
                else:
                    res_list += [{param.name: complex(value)}]
            else:
                res_list += [{param.name: value}]

        return res_list

    def _flush_data_to_database(self, block: bool = False) -> None:
        """
        Write the in-memory results to the database.

        Args:
            block: If writing using a background thread block until the
                background thread has written all data to disc. The
                argument has no effect if not using a background thread.

        """

        log.debug('Flushing to database')
        writer_status = self._writer_status
        if len(self._results) > 0:
            try:

                self.add_results(self._results)
                if writer_status.write_in_background:
                    log.debug(f"Succesfully enqueued result for write thread")
                else:
                    log.debug(f'Successfully wrote result to disk')
                self._results = []
            except Exception as e:
                if writer_status.write_in_background:
                    log.warning(f"Could not enqueue result; {e}")
                else:
                    log.warning(f'Could not commit to database; {e}')
        else:
            log.debug('No results to flush')

        if writer_status.write_in_background and block:
            log.debug(f"Waiting for write queue to empty.")
            writer_status.data_write_queue.join()

    def _export_file_name(self, prefix: str, export_type: DataExportType) -> str:
        """Get export file name"""
        extension = export_type.value
        return f"{prefix}{self.run_id}.{extension}"

    def _export_as_netcdf(self, path: str, file_name: str) -> str:
        """Export data as netcdf to a given path with file prefix"""
        file_path = os.path.join(path, file_name)
        xarr_dataset = self.to_xarray_dataset()
        data_var_kinds = [
            xarr_dataset.data_vars[data_var].dtype.kind
            for data_var in xarr_dataset.data_vars
        ]
        coord_kinds = [
            xarr_dataset.coords[coord].dtype.kind for coord in xarr_dataset.coords
        ]
        if "c" in data_var_kinds or "c" in coord_kinds:
            # see http://xarray.pydata.org/en/stable/howdoi.html
            # for how to export complex numbers
            xarr_dataset.to_netcdf(
                path=file_path, engine="h5netcdf", invalid_netcdf=True
            )
        else:
            xarr_dataset.to_netcdf(path=file_path, engine="h5netcdf")
        return file_path

    def _export_as_csv(self, path: str, file_name: str) -> str:
        """Export data as csv to a given path with file prefix"""
        self.write_data_to_text_file(path=path, single_file=True, single_file_name=file_name)
        return os.path.join(path, file_name)

    def _export_data(self,
                     export_type: DataExportType,
                     path: Optional[str] = None,
                     prefix: Optional[str] = None
                     ) -> Optional[str]:
        """Export data to disk with file name {prefix}{run_id}.{ext}.
        Values for the export type, path and prefix can also be set in the qcodes
        "dataset" config.

        Args:
            export_type: Data export type, e.g. DataExportType.NETCDF
            path: Export path, defaults to value set in config
            prefix: File prefix, e.g. "qcodes_", defaults to value set in config.

        Returns:
            str: Path file was saved to, returns None if no file was saved.
        """
        # Set defaults to values in config if the value was not set
        # (defaults to None)
        path = path if path is not None else get_data_export_path()
        prefix = prefix if prefix is not None else get_data_export_prefix()

        if DataExportType.NETCDF == export_type:
            file_name = self._export_file_name(
                prefix=prefix, export_type=DataExportType.NETCDF)
            return self._export_as_netcdf(path=path, file_name=file_name)

        elif DataExportType.CSV == export_type:
            file_name = self._export_file_name(
                prefix=prefix, export_type=DataExportType.CSV)
            return self._export_as_csv(path=path, file_name=file_name)

        else:
            return None

    def export(self,
               export_type: Optional[Union[DataExportType, str]] = None,
               path: Optional[str] = None,
               prefix: Optional[str] = None) -> None:
        """Export data to disk with file name {prefix}{run_id}.{ext}.
        Values for the export type, path and prefix can also be set in the "dataset"
        section of qcodes config.

        Args:
            export_type: Data export type, e.g. "netcdf" or ``DataExportType.NETCDF``,
                defaults to a value set in qcodes config
            path: Export path, defaults to value set in config
            prefix: File prefix, e.g. ``qcodes_``, defaults to value set in config.

        Raises:
            ValueError: If the export data type is not specified, raise an error
        """
        export_type = get_data_export_type(export_type)

        if export_type is None:
            raise ValueError(
                "No data export type specified. Please set the export data type "
                "by using ``qcodes.dataset.export_config.set_data_export_type`` or "
                "give an explicit export_type when calling ``dataset.export`` manually."
            )

        self._export_path = self._export_data(
            export_type=export_type,
            path=path,
            prefix=prefix
        )
        export_info = self.export_info
        if self._export_path is not None:
            export_info.export_paths[export_type.value] = self._export_path

        self._set_export_info(export_info)

    @property
    def export_path(self) -> Optional[str]:
        return self._export_path

    @property
    def export_info(self) -> ExportInfo:
        return self._export_info

    def _set_export_info(self, export_info: ExportInfo) -> None:
        self.add_metadata("export_info", export_info.to_str())
        self._export_info = export_info


# public api
def load_by_id(run_id: int, conn: Optional[ConnectionPlus] = None) -> DataSet:
    """
    Load a dataset by run id

    If no connection is provided, lookup is performed in the database file that
    is specified in the config.

    Note that the ``run_id`` used in this function in not preserved when copying
    data to another db file. We recommend using :func:`.load_by_run_spec` which
    does not have this issue and is significantly more flexible.

    Args:
        run_id: run id of the dataset
        conn: connection to the database to load from

    Returns:
        :class:`.DataSet` with the given run id
    """
    if run_id is None:
        raise ValueError('run_id has to be a positive integer, not None.')

    conn = conn or connect(get_DB_location())

    d = DataSet(conn=conn, run_id=run_id)
    return d


def load_by_run_spec(*,
                     captured_run_id: Optional[int] = None,
                     captured_counter: Optional[int] = None,
                     experiment_name: Optional[str] = None,
                     sample_name: Optional[str] = None,
                     # guid parts
                     sample_id: Optional[int] = None,
                     location: Optional[int] = None,
                     work_station: Optional[int] = None,
                     conn: Optional[ConnectionPlus] = None) -> DataSet:
    """
    Load a run from one or more pieces of runs specification. All
    fields are optional but the function will raise an error if more than one
    run matching the supplied specification is found. Along with the error
    specs of the runs found will be printed.

    Args:
        captured_run_id: The ``run_id`` that was originally assigned to this
          at the time of capture.
        captured_counter: The counter that was originally assigned to this
          at the time of capture.
        experiment_name: name of the experiment that the run was captured
        sample_name: The name of the sample given when creating the experiment.
        sample_id: The sample_id assigned as part of the GUID.
        location: The location code assigned as part of GUID.
        work_station: The workstation assigned as part of the GUID.
        conn: An optional connection to the database. If no connection is
          supplied a connection to the default database will be opened.

    Raises:
        NameError: if no run or more than one run with the given specification
         exists in the database

    Returns:
        :class:`.DataSet` matching the provided specification.
    """
    conn = conn or connect(get_DB_location())
    guids = get_guids_from_run_spec(conn,
                                    captured_run_id=captured_run_id,
                                    captured_counter=captured_counter,
                                    experiment_name=experiment_name,
                                    sample_name=sample_name)

    matched_guids = filter_guids_by_parts(guids, location, sample_id,
                                          work_station)

    if len(matched_guids) == 1:
        return load_by_guid(matched_guids[0], conn)
    elif len(matched_guids) > 1:
        print(generate_dataset_table(matched_guids, conn=conn))
        raise NameError("More than one matching dataset found. "
                        "Please supply more information to uniquely"
                        "identify a dataset")
    else:
        raise NameError(f'No run matching the supplied information '
                        f'found.')


def load_by_guid(guid: str, conn: Optional[ConnectionPlus] = None) -> DataSet:
    """
    Load a dataset by its GUID

    If no connection is provided, lookup is performed in the database file that
    is specified in the config.

    Args:
        guid: guid of the dataset
        conn: connection to the database to load from

    Returns:
        :class:`.DataSet` with the given guid

    Raises:
        NameError: if no run with the given GUID exists in the database
        RuntimeError: if several runs with the given GUID are found
    """
    conn = conn or connect(get_DB_location())

    # this function raises a RuntimeError if more than one run matches the GUID
    run_id = get_runid_from_guid(conn, guid)

    if run_id == -1:
        raise NameError(f'No run with GUID: {guid} found in database.')

    return DataSet(run_id=run_id, conn=conn)


def load_by_counter(counter: int, exp_id: int,
                    conn: Optional[ConnectionPlus] = None) -> DataSet:
    """
    Load a dataset given its counter in a given experiment

    Lookup is performed in the database file that is specified in the config.

    Note that the `counter` used in this function in not preserved when copying
    data to another db file. We recommend using :func:`.load_by_run_spec` which
    does not have this issue and is significantly more flexible.

    Args:
        counter: counter of the dataset within the given experiment
        exp_id: id of the experiment where to look for the dataset
        conn: connection to the database to load from. If not provided, a
          connection to the DB file specified in the config is made

    Returns:
        :class:`.DataSet` of the given counter in the given experiment
    """
    conn = conn or connect(get_DB_location())
    sql = """
    SELECT run_id
    FROM
      runs
    WHERE
      result_counter= ? AND
      exp_id = ?
    """
    c = transaction(conn, sql, counter, exp_id)
    run_id = one(c, 'run_id')

    d = DataSet(conn=conn, run_id=run_id)
    return d


def new_data_set(name: str,
                 exp_id: Optional[int] = None,
                 specs: Optional[SPECS] = None,
                 values: Optional[VALUES] = None,
                 metadata: Optional[Any] = None,
                 conn: Optional[ConnectionPlus] = None,
                 in_memory_cache: bool = True,
                 ) -> DataSet:
    """
    Create a new dataset in the currently active/selected database.

    If ``exp_id`` is not specified, the last experiment will be loaded by default.

    Args:
        name: the name of the new dataset
        exp_id: the id of the experiments this dataset belongs to, defaults
            to the last experiment
        specs: list of parameters to create this dataset with
        values: the values to associate with the parameters
        metadata: the metadata to associate with the dataset
        in_memory_cache: Should measured data be keep in memory
            and available as part of the `dataset.cache` object.

    Return:
        the newly created :class:`.DataSet`
    """
    # note that passing `conn` is a secret feature that is unfortunately used
    # in `Runner` to pass a connection from an existing `Experiment`.
    d = DataSet(path_to_db=None, run_id=None, conn=conn,
                name=name, specs=specs, values=values,
                metadata=metadata, exp_id=exp_id, in_memory_cache=in_memory_cache)

    return d


def generate_dataset_table(guids: Sequence[str],
                           conn: Optional[ConnectionPlus] = None) -> str:
    """
    Generate an ASCII art table of information about the runs attached to the
    supplied guids.

    Args:
        guids: Sequence of one or more guids
        conn: A ConnectionPlus object with a connection to the database.

    Returns: ASCII art table of information about the supplied guids.
    """
    from tabulate import tabulate
    headers = ["captured_run_id", "captured_counter", "experiment_name",
               "sample_name",
               "sample_id", "location", "work_station"]
    table = []
    for guid in guids:
        ds = load_by_guid(guid, conn=conn)
        parsed_guid = parse_guid(guid)
        table.append([ds.captured_run_id, ds.captured_counter, ds.exp_name,
                      ds.sample_name,
                      parsed_guid['sample'], parsed_guid['location'],
                      parsed_guid['work_station']])
    return tabulate(table, headers=headers)
