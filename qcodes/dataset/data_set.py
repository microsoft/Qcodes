import functools
import importlib
import json
import logging
import os
import time
import uuid
from queue import Empty, Queue
from threading import Thread
from typing import (Any, Callable, Dict, List, Optional, Sequence, Sized,
                    Tuple, Union, TYPE_CHECKING, Mapping)

if TYPE_CHECKING:
    import pandas as pd

import numpy

import qcodes.config
import qcodes.dataset.descriptions.versioning.serialization as serial
from qcodes.dataset.descriptions.dependencies import (DependencyError,
                                                      InterDependencies_)
from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.descriptions.versioning.converters import (new_to_old,
                                                               old_to_new,
                                                               v1_to_v0)
from qcodes.dataset.descriptions.versioning.v0 import InterDependencies
from qcodes.dataset.guids import (
    filter_guids_by_parts, generate_guid, parse_guid)
from qcodes.dataset.linked_datasets.links import (Link, links_to_str,
                                                  str_to_links)
from qcodes.dataset.sqlite.connection import (ConnectionPlus, atomic,
                                              atomic_transaction,
                                              transaction)
from qcodes.dataset.sqlite.database import (
    connect, get_DB_location, conn_from_dbpath_or_conn)
from qcodes.dataset.sqlite.queries import (
    add_meta_data, add_parameter, completed, create_run,
    get_completed_timestamp_from_run_id, get_data,
    get_experiment_name_from_experiment_id, get_experiments,
    get_guid_from_run_id, get_guids_from_run_spec,
    get_last_experiment, get_metadata, get_metadata_from_run_id,
    get_parameter_data, get_parent_dataset_links, get_run_description,
    get_run_timestamp_from_run_id, get_runid_from_guid,
    get_sample_name_from_experiment_id, get_setpoints, get_values,
    mark_run_complete, remove_trigger, run_exists, set_run_timestamp,
    update_parent_datasets, update_run_description)
from qcodes.dataset.sqlite.query_helpers import (VALUE, insert_many_values,
                                                 insert_values, length, one,
                                                 select_one_where, VALUES)
from qcodes.instrument.parameter import _BaseParameter
from qcodes.utils.deprecate import deprecate

log = logging.getLogger(__name__)


# TODO: as of now every time a result is inserted with add_result the db is
# saved same for add_results. IS THIS THE BEHAVIOUR WE WANT?

# TODO: storing parameters in separate table as an extension (dropping
# the column parametenrs would be much nicer

# TODO: metadata split between well known columns and maybe something else is
# not such a good idea. The problem is if we allow for specific columns then
# how do the user/us know which are metatadata?  I THINK the only sane solution
# is to store JSON in a column called metadata

# TODO: fixix  a subset of metadata that we define well known (and create them)
# i.e. no dynamic creation of metadata columns, but add stuff to
# a json inside a 'metadata' column


SPECS = List[ParamSpec]
# Transition period type: SpecsOrInterDeps. We will allow both as input to
# the DataSet constructor for a while, then deprecate SPECS and finally remove
# the ParamSpec class
SpecsOrInterDeps = Union[SPECS, InterDependencies_]


class CompletedError(RuntimeError):
    pass

class DataLengthException(Exception):
    pass

class DataPathException(Exception):
    pass


class _Subscriber(Thread):
    """
    Class to add a subscriber to a :class:`.DataSet`. The subscriber gets called every
    time an insert is made to the results_table.

    The _Subscriber is not meant to be instantiated directly, but rather used
    via the 'subscribe' method of the :class:`.DataSet`.

    NOTE: A subscriber should be added *after* all parameters have been added.

    NOTE: Special care shall be taken when using the *state* object: it is the
    user's responsibility to operate with it in a thread-safe way.
    """
    def __init__(self,
                 dataSet: 'DataSet',
                 id_: str,
                 callback: Callable[..., None],
                 state: Optional[Any] = None,
                 loop_sleep_time: int = 0,  # in milliseconds
                 min_queue_length: int = 1,
                 callback_kwargs: Optional[Mapping[str, Any]] = None
                 ) -> None:
        super().__init__()

        self._id = id_

        self.dataSet = dataSet
        self.table_name = dataSet.table_name
        self._data_set_len = len(dataSet)

        self.state = state

        self.data_queue: Queue = Queue()
        self._queue_length: int = 0
        self._stop_signal: bool = False
        # convert milliseconds to seconds
        self._loop_sleep_time = loop_sleep_time / 1000
        self.min_queue_length = min_queue_length

        if callback_kwargs is None or len(callback_kwargs) == 0:
            self.callback = callback
        else:
            self.callback = functools.partial(callback, **callback_kwargs)

        self.callback_id = f"callback{self._id}"
        self.trigger_id = f"sub{self._id}"

        conn = dataSet.conn

        conn.create_function(self.callback_id, -1, self._cache_data_to_queue)

        parameters = dataSet.get_parameters()
        sql_param_list = ",".join([f"NEW.{p.name}" for p in parameters])
        sql_create_trigger_for_callback = f"""
        CREATE TRIGGER {self.trigger_id}
            AFTER INSERT ON '{self.table_name}'
        BEGIN
            SELECT {self.callback_id}({sql_param_list});
        END;"""
        atomic_transaction(conn, sql_create_trigger_for_callback)

        self.log = logging.getLogger(f"_Subscriber {self._id}")

    def _cache_data_to_queue(self, *args: Any) -> None:
        self.log.debug(f"Args:{args} put into queue for {self.callback_id}")
        self.data_queue.put(args)
        self._data_set_len += 1
        self._queue_length += 1

    def run(self) -> None:
        self.log.debug("Starting subscriber")
        self._loop()

    @staticmethod
    def _exhaust_queue(queue: Queue) -> List:
        result_list = []
        while True:
            try:
                result_list.append(queue.get(block=False))
            except Empty:
                break
        return result_list

    def _call_callback_on_queue_data(self) -> None:
        result_list = self._exhaust_queue(self.data_queue)
        self.callback(result_list, self._data_set_len, self.state)
        self.log.debug(f"{self.callback} called with "
                       f"result_list: {result_list}.")

    def _loop(self) -> None:
        while True:
            if self._stop_signal:
                self._clean_up()
                break

            if self._queue_length >= self.min_queue_length:
                self._call_callback_on_queue_data()
                self._queue_length = 0

            time.sleep(self._loop_sleep_time)

            if self.dataSet.completed:
                self._call_callback_on_queue_data()
                break

    def done_callback(self) -> None:
        self.log.debug("Done callback")
        self._call_callback_on_queue_data()

    def schedule_stop(self) -> None:
        if not self._stop_signal:
            self.log.debug("Scheduling stop")
            self._stop_signal = True

    def _clean_up(self) -> None:
        self.log.debug("Stopped subscriber")


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

    def __init__(self, path_to_db: str = None,
                 run_id: Optional[int] = None,
                 conn: Optional[ConnectionPlus] = None,
                 exp_id: Optional[int] = None,
                 name: str = None,
                 specs: Optional[SpecsOrInterDeps] = None,
                 values: Optional[VALUES] = None,
                 metadata: Optional[Mapping[str, Any]] = None) -> None:
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
            specs: paramspecs belonging to the dataset. Ignored if ``run_id`` is
              provided.
            values: values to insert into the dataset. Ignored if ``run_id`` is
              provided.
            metadata: metadata to insert into the dataset. Ignored if ``run_id``
              is provided.
        """
        self.conn = conn_from_dbpath_or_conn(conn, path_to_db)

        self._debug = False
        self.subscribers: Dict[str, _Subscriber] = {}
        self._interdeps: InterDependencies_
        self._parent_dataset_links: List[Link]

        if run_id is not None:
            if not run_exists(self.conn, run_id):
                raise ValueError(f"Run with run_id {run_id} does not exist in "
                                 f"the database")
            self._run_id = run_id
            self._completed = completed(self.conn, self.run_id)
            run_desc = self._get_run_description_from_db()
            self._interdeps = run_desc.interdeps
            self._metadata = get_metadata_from_run_id(self.conn, self.run_id)
            self._started = self.run_timestamp_raw is not None
            self._parent_dataset_links = str_to_links(
                get_parent_dataset_links(self.conn, self.run_id))
        else:
            # Actually perform all the side effects needed for the creation
            # of a new dataset. Note that a dataset is created (in the DB)
            # with no parameters; they are written to disk when the dataset
            # is marked as started
            if exp_id is None:
                exp_id = get_last_experiment(self.conn)
                if exp_id is None:  # if it's still None, then...
                    raise ValueError("No experiments found."
                                     "You can start a new one with:"
                                     " new_experiment(name, sample_name)")
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
                self._interdeps = specs
            elif specs is not None:
                self._interdeps = old_to_new(InterDependencies(*specs))
            else:
                self._interdeps = InterDependencies_()
            self._metadata = get_metadata_from_run_id(self.conn, self.run_id)
            self._parent_dataset_links = []

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
    def snapshot(self) -> Optional[dict]:
        """Snapshot of the run as dictionary (or None)"""
        snapshot_json = self.snapshot_raw
        if snapshot_json is not None:
            return json.loads(snapshot_json)
        else:
            return None

    @property
    def snapshot_raw(self) -> Optional[str]:
        """Snapshot of the run as a JSON-formatted string (or None)"""
        return select_one_where(self.conn, "runs", "snapshot",
                                "run_id", self.run_id)

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
        return tuple(self._interdeps.dependencies.keys())

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
        return RunDescriber(interdeps=self._interdeps)

    @property
    def metadata(self) -> Dict:
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

        if not all((isinstance(link, Link) for link in links)):
            raise ValueError('Invalid input. Did not receive a list of Links')

        for link in links:
            if link.head != self.guid:
                raise ValueError(
                    'Invalid input. All links must point to this dataset. '
                    'Got link(s) with head(s) pointing to another dataset.')

        self._parent_dataset_links = links

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

    def add_parameter(self, spec: ParamSpec) -> None:
        """
        Old method; don't use it.
        """
        raise NotImplementedError('This method has been removed. '
                                  'Please use DataSet.set_interdependencies '
                                  'instead.')

    def set_interdependencies(self, interdeps: InterDependencies_) -> None:
        """
        Overwrite the interdependencies object (which holds all added
        parameters and their relationships) of this dataset
        """
        if not isinstance(interdeps, InterDependencies_):
            raise TypeError('Wrong input type. Expected InterDepencies_, '
                            f'got {type(interdeps)}')

        if not self.pristine:
            mssg = ('Can not set interdependencies on a DataSet that has '
                    'been started.')
            raise RuntimeError(mssg)

        self._interdeps = interdeps

    def get_parameters(self) -> SPECS:
        rd_v0 = v1_to_v0(self.description)
        old_interdeps = rd_v0.interdeps
        return list(old_interdeps.paramspecs)

    def add_metadata(self, tag: str, metadata: Any) -> None:
        """
        Adds metadata to the :class:`.DataSet`. The metadata is stored under the
        provided tag. Note that None is not allowed as a metadata value.

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
        return self._started and not(self._completed)

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

    def mark_started(self) -> None:
        """
        Mark this :class:`.DataSet` as started. A :class:`.DataSet` that has been started can not
        have its parameters modified.

        Calling this on an already started :class:`.DataSet` is a NOOP.
        """
        if not self._started:
            self._perform_start_actions()
            self._started = True

    def _perform_start_actions(self) -> None:
        """
        Perform the actions that must take place once the run has been started
        """
        paramspecs = new_to_old(self._interdeps).paramspecs

        for spec in paramspecs:
            add_parameter(self.conn, self.table_name, spec)

        desc_str = serial.to_json_for_storage(self.description)

        update_run_description(self.conn, self.run_id, desc_str)

        set_run_timestamp(self.conn, self.run_id)

        pdl_str = links_to_str(self._parent_dataset_links)
        update_parent_datasets(self.conn, self.run_id, pdl_str)

    def mark_completed(self) -> None:
        """
        Mark :class:`.DataSet` as complete and thus read only and notify the subscribers
        """
        if self.pristine:
            raise RuntimeError('Can not mark DataSet as complete before it '
                               'has been marked as started.')
        self.completed = True
        for sub in self.subscribers.values():
            sub.done_callback()

    @deprecate(alternative='mark_completed')
    def mark_complete(self) -> None:
        self.mark_completed()

    def add_result(self, results: Mapping[str, VALUE]) -> int:
        """
        Add a logically single result to existing parameters

        Args:
            results: dictionary with name of a parameter as the key and the
                value to associate as the value.

        Returns:
            index in the DataSet that the result was stored at

        If a parameter exist in the :class:`.DataSet` and it's not in the results
        dictionary, "Null" values are inserted.

        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this :class:`.DataSet`.

        It is an error to add results to a completed :class:`.DataSet`.
        """

        if self.pristine:
            raise RuntimeError('This DataSet has not been marked as started. '
                               'Please mark the DataSet as started before '
                               'adding results to it.')

        if self.completed:
            raise CompletedError('This DataSet is complete, no further '
                                 'results can be added to it.')
        try:
            parameters = [self._interdeps._id_to_paramspec[name]
                          for name in results]
            self._interdeps.validate_subset(parameters)
        except DependencyError as de:
            raise ValueError(
                'Can not add result, missing setpoint values') from de

        index = insert_values(self.conn, self.table_name,
                              list(results.keys()),
                              list(results.values())
                              )
        return index

    def add_results(self, results: Sequence[Mapping[str, VALUE]]) -> int:
        """
        Adds a sequence of results to the :class:`.DataSet`.

        Args:
            results: list of name-value dictionaries where each dictionary
                provides the values for the parameters in that result. If some
                parameters are missing the corresponding values are assumed
                to be None

        Returns:
            the index in the :class:`.DataSet` that the **first** result was stored at

        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this :class:`.DataSet`.

        It is an error to add results to a completed :class:`.DataSet`.
        """

        if self.pristine:
            raise RuntimeError('This DataSet has not been marked as started. '
                               'Please mark the DataSet as started before '
                               'adding results to it.')

        if self.completed:
            raise CompletedError('This DataSet is complete, no further '
                                 'results can be added to it.')

        expected_keys = frozenset.union(*[frozenset(d) for d in results])
        values = [[d.get(k, None) for k in expected_keys] for d in results]

        len_before_add = length(self.conn, self.table_name)

        insert_many_values(self.conn, self.table_name, list(expected_keys),
                           values)
        return len_before_add

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

    def get_data(self,
                 *params: Union[str, ParamSpec, _BaseParameter],
                 start: Optional[int] = None,
                 end: Optional[int] = None) -> List[List[Any]]:
        """
        Returns the values stored in the :class:`.DataSet` for the specified parameters.
        The values are returned as a list of lists, SQL rows by SQL columns,
        e.g. datapoints by parameters. The data type of each element is based
        on the datatype provided when the :class:`.DataSet` was created. The parameter
        list may contain a mix of string parameter names, QCoDeS Parameter
        objects, and ParamSpec objects (as long as they have a ``name`` field).

        If provided, the start and end arguments select a range of results
        by result count (index). If the range is empty - that is, if the end is
        less than or equal to the start, or if start is after the current end
        of the :class:`.DataSet` – then a list of empty arrays is returned.

        For a more type independent and easier to work with view of the data
        you may want to consider using
        :py:meth:`.get_parameter_data`

        Args:
            *params: string parameter names, QCoDeS Parameter objects, and
                ParamSpec objects
            start: start value of selection range (by result count); ignored
                if None
            end: end value of selection range (by results count); ignored if
                None

        Returns:
            list of lists SQL rows of data by SQL columns. Each SQL row is a
            datapoint and each SQL column is a parameter. Each element will
            be of the datatypes stored in the database (numeric, array or
            string)
        """
        valid_param_names = self._validate_parameters(*params)
        return get_data(self.conn, self.table_name, valid_param_names,
                        start, end)

    def get_parameter_data(
            self,
            *params: Union[str, ParamSpec, _BaseParameter],
            start: Optional[int] = None,
            end: Optional[int] = None) -> Dict[str, Dict[str, numpy.ndarray]]:
        """
        Returns the values stored in the :class:`.DataSet` for the specified parameters
        and their dependencies. If no paramerers are supplied the values will
        be returned for all parameters that are not them self dependencies.

        The values are returned as a dictionary with names of the requested
        parameters as keys and values consisting of dictionaries with the
        names of the parameters and its dependencies as keys and numpy arrays
        of the data as values. If some of the parameters are stored as arrays
        the remaining parameters are expanded to the same shape as these.
        Apart from this expansion the data returned by this method
        is the transpose of the date returned by ``get_data``.

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
                                 for ps in self._interdeps.non_dependencies]
        else:
            valid_param_names = self._validate_parameters(*params)
        return get_parameter_data(self.conn, self.table_name,
                                  valid_param_names, start, end)

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
        import pandas as pd
        dfs = {}
        datadict = self.get_parameter_data(*params,
                                           start=start,
                                           end=end)
        for name, subdict in datadict.items():
            keys = list(subdict.keys())
            if len(keys) == 0:
                dfs[name] = pd.DataFrame()
                continue
            if len(keys) == 1:
                index = None
            elif len(keys) == 2:
                index = pd.Index(subdict[keys[1]].ravel(), name=keys[1])
            else:
                indexdata = tuple(numpy.concatenate(subdict[key])
                                  if subdict[key].dtype == numpy.dtype('O')
                                  else subdict[key].ravel()
                                  for key in keys[1:])
                index = pd.MultiIndex.from_arrays(
                    indexdata,
                    names=keys[1:])

            if subdict[keys[0]].dtype == numpy.dtype('O'):
                # ravel will not fully unpack a numpy array of arrays
                # which are of "object" dtype. This can happen if a variable
                # length array is stored in the db. We use concatenate to
                # flatten these
                mydata = numpy.concatenate(subdict[keys[0]])
            else:
                mydata = subdict[keys[0]].ravel()
            df = pd.DataFrame(mydata, index=index,
                              columns=[keys[0]])
            dfs[name] = df
        return dfs

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

        Raises:
            DataLengthException: If the data of multiple parameters have not same
                                 length and wanted to be merged in a single file.
            DataPathException: If the data of multiple parameters are wanted to be merged
                               in a single file but no filename provided.
        """
        import pandas as pd
        dfdict = self.get_data_as_pandas_dataframe()
        dfs_to_save = list()
        for parametername, df in dfdict.items():
            if not single_file:
                dst = os.path.join(path, f'{parametername}.dat')
                df.to_csv(path_or_buf=dst, header=False, sep='\t')
            else:
                dfs_to_save.append(df)
        if single_file:
            df_length = len(dfs_to_save[0])
            if any(len(df) != df_length for df in dfs_to_save):
                raise DataLengthException("You cannot concatenate data " +
                                          "with different length to a " +
                                          "single file.")
            if single_file_name == None:
                raise DataPathException("Please provide the desired file name " +
                                        "for the concatenated data.")
            else:
                dst = os.path.join(path, f'{single_file_name}.dat')
                df_to_save = pd.concat(dfs_to_save, axis=1)
                df_to_save.to_csv(path_or_buf=dst, header=False, sep='\t')

    def get_values(self, param_name: str) -> List[List[Any]]:
        """
        Get the values (i.e. not NULLs) of the specified parameter
        """
        if param_name not in self.parameters:
            raise ValueError('Unknown parameter, not in this DataSet')

        values = get_values(self.conn, self.table_name, param_name)

        return values

    def get_setpoints(self, param_name: str) -> Dict[str, List[List[Any]]]:
        """
        Get the setpoints for the specified parameter

        Args:
            param_name: The name of the parameter for which to get the
                setpoints
        """

        paramspec: ParamSpecBase = self._interdeps._id_to_paramspec[param_name]

        if param_name not in self.parameters:
            raise ValueError('Unknown parameter, not in this DataSet')

        if paramspec not in self._interdeps.dependencies.keys():
            raise ValueError(f'Parameter {param_name} has no setpoints.')

        setpoints = get_setpoints(self.conn, self.table_name, param_name)

        return setpoints

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
                 conn: Optional[ConnectionPlus] = None) -> DataSet:
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

    Return:
        the newly created :class:`.DataSet`
    """
    # note that passing `conn` is a secret feature that is unfortunately used
    # in `Runner` to pass a connection from an existing `Experiment`.
    d = DataSet(path_to_db=None, run_id=None, conn=conn,
                name=name, specs=specs, values=values,
                metadata=metadata, exp_id=exp_id)

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
