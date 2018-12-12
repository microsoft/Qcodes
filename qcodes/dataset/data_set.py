import functools
from collections import defaultdict
from itertools import chain
import json
from typing import Any, Dict, List, Optional, Union, Sized, Callable
from threading import Thread
import time
import importlib
import logging
import uuid
from queue import Queue, Empty
from time import sleep

from qcodes.dataset.param_spec import ParamSpec
from qcodes.instrument.parameter import _BaseParameter
from qcodes.dataset.sqlite_base import (atomic, atomic_transaction,
                                        transaction, add_parameter,
                                        connect, create_run, completed,
                                        is_column_in_table,
                                        get_experiments,
                                        get_last_experiment, select_one_where,
                                        length, modify_values,
                                        add_meta_data, mark_run_complete,
                                        modify_many_values, insert_values,
                                        insert_many_values,
                                        VALUE, VALUES, get_data,
                                        get_values,
                                        get_setpoints,
                                        get_metadata,
                                        get_metadata_from_run_id,
                                        one,
                                        get_experiment_name_from_experiment_id,
                                        get_sample_name_from_experiment_id,
                                        get_guid_from_run_id,
                                        get_runid_from_guid,
                                        get_run_timestamp_from_run_id,
                                        get_completed_timestamp_from_run_id,
                                        update_run_description,
                                        run_exists, remove_trigger,
                                        make_connection_plus_from,
                                        ConnectionPlus)
from qcodes.dataset.rabbitmq_storage_interface import RabbitMQWriterInterface
from qcodes.dataset.sqlite_storage_interface import (SqliteReaderInterface,
                                                     SqliteWriterInterface)
from qcodes.dataset.data_storage_interface import (DataReaderInterface,
                                                   DataWriterInterface,
                                                   DataStorageInterface)

from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.dependencies import InterDependencies
from qcodes.dataset.database import get_DB_location, path_to_dbfile
from qcodes.dataset.guids import generate_guid
from qcodes.utils.deprecate import deprecate
import qcodes.config
from qcodes.utils.helpers import NumpyJSONEncoder

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

log = logging.getLogger(__name__)
SPECS = List[ParamSpec]


class CompletedError(RuntimeError):
    pass


class _Subscriber(Thread):
    """
    Class to add a subscriber to the Sqlite storage backend of a DataSet. If a
    DataSet uses a different storage backend, subscribers will not work.

    The subscriber gets called every time an insert is made to the
    results_table.

    The _Subscriber is not meant to be instantiated directly, but rather used
    via the 'subscribe' method of the DataSet.

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
                 callback_kwargs: Optional[Dict[str, Any]]=None
                 ) -> None:
        super().__init__()

        self._id = id_

        self.dataSet = dataSet
        if not isinstance(self.dataSet.dsi.writer, SqliteWriterInterface):
            raise ValueError('Received a dataset with a writer interface of '
                             'type {type(dataSet.dsi.writer)}. Writer '
                             'interface must be of type '
                             'SqliteWriterInterface.')

        self.table_name = dataSet.dsi.writer.table_name
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

        conn = dataSet.dsi.writer.conn

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

    def _cache_data_to_queue(self, *args) -> None:
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
                         'description', 'completed_timestamp_raw', 'metadata')

    def __init__(self,
                 guid: Optional[str] = None,
                 run_id: Optional[int] = None,
                 readerinterface: type = SqliteReaderInterface,
                 writerinterface: type = SqliteWriterInterface,
                 **si_kwargs) -> None:
        """
        Create a new DataSet object. The object can either hold a new run or
        an already existing run.

        Args:
            guid: GUID of dataset. If not provided, a new dataset is created.
              If provided, the corresponding dataset is loaded (if it exists).
            run_id: an alternative to GUID that can be used IFF the
              SqliteStorageInterface is being used. It is an error to provide
              both a run_id and a GUID.
            storageinterface: The class of the storage interface to use for
              storing/loading the dataset
            si_kwargs: the kwargs to pass to the constructor of the
              storage interface
        """

        if not issubclass(readerinterface, DataReaderInterface):
            raise ValueError("The provided data reader interface is not "
                             "valid. Must be a subclass of "
                             "qcodes.dataset.data_storage_interface."
                             "DataReaderInterface")

        reader_is_sqlite = readerinterface == SqliteReaderInterface
        writer_is_sqlite = writerinterface == SqliteWriterInterface

        if run_id is not None and guid is not None:
            raise ValueError('Got values for both GUID and run_id. Please '
                             'provide at most a value for one of them.')

        # First handle the annoying situation of no GUID but a run_id

        if guid is None and run_id is not None:
            if not(reader_is_sqlite):
                raise ValueError('Got a run_id but is not using '
                                 'SQLiteReaderInterface.')

            conn = si_kwargs.get('conn', None)
            if conn is None:
                path_to_db = si_kwargs.get('path_to_db', None)
                if path_to_db is None:
                    conn = connect(get_DB_location())
                else:
                    conn = connect(path_to_db)
            try:
                guid = get_guid_from_run_id(conn, run_id)
            except RuntimeError:
                raise ValueError(f"Run with run_id "
                                 f"{run_id} does not "
                                 f"exist in the database")

        # Now (guid is None) is the switch for creating / loading a dataset


        if guid is not None:  # Case: Loading run
            log.info(f'Attempting to load existing run with GUID: {guid}')

            # Handle kwargs
            reader_kwargs = DataSet._kwargs_for_reader_when_loading(
                                readerinterface, **si_kwargs)
            writer_kwargs = DataSet._kwargs_for_writer_when_loading(
                                writerinterface, **si_kwargs)

            self._guid = guid
            self.dsi = DataStorageInterface(self._guid,
                                            reader=readerinterface,
                                            writer=writerinterface,
                                            reader_kwargs=reader_kwargs,
                                            writer_kwargs=writer_kwargs)
            if not self.dsi.run_exists():
                raise ValueError(f'No run with GUID {guid} found.')
            self.dsi.retrieve_meta_data()
            r = self.dsi.reader
            self.dsi.writer.resume_run(r.exp_id, r.run_id, r.name,
                                       r.table_name, r.counter)

        else:  # Case: Creating run
            self._guid = generate_guid()
            log.info(f'Creating new run with GUID: {self._guid}')

            # Handle kwargs
            reader_kwargs = DataSet._kwargs_for_reader_when_creating(
                                readerinterface, **si_kwargs)
            writer_kwargs = DataSet._kwargs_for_writer_when_creating(
                                writerinterface, **si_kwargs)
            creation_kwargs = DataSet._kwargs_for_create_run(
                                writerinterface, **si_kwargs)

            self.dsi = DataStorageInterface(self._guid,
                                            reader=readerinterface,
                                            writer=writerinterface,
                                            writer_kwargs=writer_kwargs,
                                            reader_kwargs=reader_kwargs)
            self.dsi.create_run(**creation_kwargs)

        # Assign all attributes
        run_meta_data = self.dsi.retrieve_meta_data()

        self._completed: bool = run_meta_data.run_completed is not None
        self._description = run_meta_data.run_description
        self._snapshot = run_meta_data.snapshot
        self._metadata = run_meta_data.tags

        self._started: bool = self._completed or self.number_of_results > 0

    @staticmethod
    def _kwargs_for_create_run(writer: type, **si_kwargs):
        """
        Helper function to dig out appropriate kwargs for create_run
        """
        if writer in (SqliteWriterInterface, RabbitMQWriterInterface):
            return {'exp_id': si_kwargs.get('exp_id', None),
                    'name': si_kwargs.get('name', None)}
        else:
            raise NotImplementedError('Only SQLiteWriterInterface and '
                                      'RabbitMQWriterInterface are '
                                      'currently supported.')

    @staticmethod
    def _kwargs_for_reader_when_loading(reader: type, **si_kwargs):
        """
        Helper function to dig out the appropriate kwargs for the reader
        """
        if reader == SqliteReaderInterface:
            conn = si_kwargs.get('conn', None)
            if conn is None:
                path_to_db = si_kwargs.get('path_to_db', None)
                if path_to_db is None:
                    conn = connect(get_DB_location())
                else:
                    conn = connect(path_to_db)
            elif si_kwargs.get('path_to_db', None) is not None:
                raise ValueError("Both `path_to_db` and `conn` "
                                 "arguments have been passed together "
                                 "with non-None values. This is not "
                                 "allowed.")
            return {'conn': conn}
        else:
            raise NotImplementedError('Only SQLiteReaderInterface is '
                                      'currently supported.')

    @staticmethod
    def _kwargs_for_writer_when_loading(writer: type, **si_kwargs):
        """
        Helper function to dig out the appropriate kwargs for the writer
        """

        if writer == SqliteWriterInterface:
            # When we are loading, the writer gets a NEW connection to
            # the same DB
            conn = si_kwargs.get('conn', None)
            if conn is None:
                path_to_db = si_kwargs.get('path_to_db', None)
                if path_to_db is None:
                    conn = connect(get_DB_location())
                else:
                    conn = connect(path_to_db)
            elif si_kwargs.get('path_to_db', None) is not None:
                raise ValueError("Both `path_to_db` and `conn` "
                                 "arguments have been passed together "
                                 "with non-None values. This is not "
                                 "allowed.")
            else:
                conn = connect(path_to_dbfile(conn))
            return {'conn': conn}
        elif writer == RabbitMQWriterInterface:
            return {}
        else:
            raise NotImplementedError('Only SQLiteWriterInterface and '
                                      'RabbitMQWriterInterface are '
                                      'currently supported.')

    @staticmethod
    def _kwargs_for_writer_when_creating(writer: type, **si_kwargs):
        """
        Helper function to dig out the appropriate kwargs for the writer
        """
        if writer == SqliteWriterInterface:
            conn = si_kwargs.get('conn', None)
            if conn is None:
                path_to_db = si_kwargs.get('path_to_db', None)
                if path_to_db is None:
                    conn = connect(get_DB_location())
                else:
                    conn = connect(path_to_db)
            elif si_kwargs.get('path_to_db', None) is not None:
                raise ValueError("Both `path_to_db` and `conn` "
                                 "arguments have been passed together "
                                 "with non-None values. This is not "
                                 "allowed.")
            return {'conn': conn}
        elif writer == RabbitMQWriterInterface:
            return {}
        else:
            raise NotImplementedError('Only SQLiteWriterInterface and '
                                      'RabbitMQWriterInterface are '
                                      'currently supported.')

    @staticmethod
    def _kwargs_for_reader_when_creating(reader: type, **si_kwargs):
        """
        Helper function to dig out the appropriate kwargs for the reader
        """
        if reader == SqliteReaderInterface:
            # when we are creating a new run, the reader gets a NEW connection
            # to the same DB. TODO: make this connection read-only
            conn = si_kwargs.get('conn', None)
            if conn is None:
                path_to_db = si_kwargs.get('path_to_db', None)
                if path_to_db is None:
                    conn = connect(get_DB_location())
                else:
                    conn = connect(path_to_db)
            elif si_kwargs.get('path_to_db', None) is not None:
                raise ValueError("Both `path_to_db` and `conn` "
                                 "arguments have been passed together "
                                 "with non-None values. This is not "
                                 "allowed.")
            else:
                conn = connect(path_to_dbfile(conn))
            return {'conn': conn}
        else:
            raise NotImplementedError('Only SQLiteReaderInterface is '
                                      'currently supported.')

    @property
    def name(self):
        md = self.dsi.retrieve_meta_data()
        return md.name

    @property
    def guid(self):
        return self._guid

    @property
    def snapshot(self) -> Optional[dict]:
        """Snapshot of the run as dictionary (or None)"""
        md = self.dsi.retrieve_meta_data()
        return md.snapshot

    @property
    def snapshot_raw(self) -> Optional[str]:
        """Snapshot of the run as a JSON-formatted string (or None)"""
        snapshot_raw = None

        if hasattr(self.dsi.reader, '_encode_snapshot'):
            current_snapshot = self.snapshot
            if current_snapshot is not None:
                snapshot_raw = self.dsi.reader._encode_snapshot(current_snapshot)

        return snapshot_raw

    @property
    def number_of_results(self) -> int:
        return self.dsi.retrieve_number_of_results()

    @property
    def parameters(self) -> str:
        idps = self._description.interdeps
        return ','.join([p.name for p in idps.paramspecs])

    @property
    def paramspecs(self) -> Dict[str, ParamSpec]:
        params = self.get_parameters()
        param_names = [p.name for p in params]
        return dict(zip(param_names, params))

    @property
    def exp_id(self) -> int:
        return getattr(self.dsi.reader, 'exp_id', None)

    @property
    def exp_name(self) -> str:
        md = self.dsi.retrieve_meta_data()
        return md.exp_name

    @property
    def sample_name(self) -> str:
        md = self.dsi.retrieve_meta_data()
        return md.sample_name

    @property
    def run_timestamp_raw(self) -> float:
        """
        Returns run timestamp as number of seconds since the Epoch

        The run timestamp is the moment when the measurement for this run
        started.
        """
        md = self.dsi.retrieve_meta_data()
        return md.run_started

    @property
    def description(self) -> RunDescriber:
        md = self.dsi.retrieve_meta_data()
        return md.run_description

    @property
    def metadata(self) -> Dict:
        return self._metadata

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

        for attr in DataSet.persistent_traits:
            if getattr(self, attr) != getattr(other, attr):
                if guids_match:
                    raise RuntimeError('Critical inconsistency detected! '
                                       'The two datasets have the same GUID,'
                                       f' but their "{attr}" differ.')
                else:
                    return False

        return True

    def run_timestamp(self, fmt: str="%Y-%m-%d %H:%M:%S") -> str:
        """
        Returns run timestamp in a human-readable format

        The run timestamp is the moment when the measurement for this run
        started.

        Consult with `time.strftime` for information about the format.
        """
        return time.strftime(fmt, time.localtime(self.run_timestamp_raw))

    @property
    def completed_timestamp_raw(self) -> Union[float, None]:
        """
        Returns timestamp when measurement run was completed
        as number of seconds since the Epoch

        If the run (or the dataset) is not completed, then returns None.
        """
        md = self.dsi.retrieve_meta_data()
        return md.run_completed

    def completed_timestamp(self,
                            fmt: str="%Y-%m-%d %H:%M:%S") -> Union[str, None]:
        """
        Returns timestamp when measurement run was completed
        in a human-readable format

        If the run (or the dataset) is not completed, then returns None.

        Consult with `time.strftime` for information about the format.
        """
        completed_timestamp_raw = self.completed_timestamp_raw

        if completed_timestamp_raw:
            completed_timestamp = time.strftime(
                fmt, time.localtime(completed_timestamp_raw))
        else:
            completed_timestamp = None

        return completed_timestamp

    @property
    def run_id(self) -> Optional[int]:
        w_run_id = getattr(self.dsi.writer, 'run_id', None)
        r_run_id = getattr(self.dsi.reader, 'run_id', None)
        if w_run_id is not None and r_run_id is not None:
            if w_run_id != r_run_id:
                raise RuntimeError('Reader and writer run_id inconsistent! '
                                   f'Reader run_id: {r_run_id}, '
                                   f'writer run_id: {w_run_id}')
        return w_run_id or r_run_id

    def _perform_start_actions(self) -> None:
        """
        Perform the actions that must take place once the run has been started
        """
        # write down the run_description
        self.dsi.store_meta_data(run_description=self.description)

        # let data storage interface prepare for storing actual data
        self.dsi.prepare_for_storing_results()

    def add_parameter(self, spec: ParamSpec):
        """
        Add a parameter to the DataSet. To ensure sanity, parameters must be
        added to the DataSet in a sequence matching their internal
        dependencies, i.e. first independent parameters, next other
        independent parameters inferred from the first ones, and finally
        the dependent parameters
        """
        if self._started:
            raise RuntimeError('It is not allowed to add parameters to a '
                               'started run')

        if self.parameters:
            old_params = self.parameters.split(',')
        else:
            old_params = []

        if spec.name in old_params:
            raise ValueError(f'Duplicate parameter name: {spec.name}')

        inf_from = spec.inferred_from.split(', ')
        if inf_from == ['']:
            inf_from = []
        for ifrm in inf_from:
            if ifrm not in old_params:
                raise ValueError('Can not infer parameter '
                                 f'{spec.name} from {ifrm}, '
                                 'no such parameter in this DataSet')

        dep_on = spec.depends_on.split(', ')
        if dep_on == ['']:
            dep_on = []
        for dp in dep_on:
            if dp not in old_params:
                raise ValueError('Can not have parameter '
                                 f'{spec.name} depend on {dp}, '
                                 'no such parameter in this DataSet')

        desc = self.description
        desc.interdeps = InterDependencies(*desc.interdeps.paramspecs, spec)
        self._description = desc

        self.dsi.store_meta_data(run_description=desc)

    def get_parameters(self) -> SPECS:
        return list(self.description.interdeps.paramspecs)

    def add_metadata(self, tag: str, metadata: Any):
        """
        Adds metadata to the DataSet. The metadata is stored under the
        provided tag. Note that None is not allowed as a metadata value.

        Args:
            tag: represents the key in the metadata dictionary
            metadata: actual metadata
        """
        self._metadata[tag] = metadata
        self.dsi.store_meta_data(tags=self._metadata)

    def add_snapshot(self, snapshot: dict, overwrite: bool=False) -> None:
        """
        Adds a snapshot to this run

        Args:
            snapshot: the snapshot dictionary
            overwrite: force overwrite an existing snapshot
        """
        current_snapshot = self.snapshot

        if current_snapshot is None or overwrite:
            self.dsi.store_meta_data(snapshot=snapshot)

        elif current_snapshot is not None and not overwrite:
            log.warning('This dataset already has a snapshot. Use overwrite'
                        '=True to overwrite that')

    @property
    def pristine(self) -> bool:
        """
        DataSet is in pristine state if it hasn't started and hasn't completed
        """
        return not self._started and not self.completed

    @property
    def running(self) -> bool:
        """
        DataSet is in running state if it has started but hasn't completed
        """
        return self._started and not self.completed

    @property
    def completed(self) -> bool:
        return self._completed

    @completed.setter
    def completed(self, value: bool):
        self._completed = value
        if value:
            self._started = True
            self.dsi.store_meta_data(run_completed=time.time())

    def mark_completed(self) -> None:
        """
        Mark dataset as complete and thus read only and notify the subscribers
        """
        self.completed = True
        for sub in self.dsi.writer.subscribers.values():
            sub.done_callback()

    @deprecate(alternative='mark_completed')
    def mark_complete(self):
        self.mark_completed()

    def add_result(self, results: Dict[str, VALUE]) -> None:
        """
        Add a logically single result to existing parameters

        Args:
            results: dictionary with name of a parameter as the key and the
                value to associate as the value.

        Returns:
            index in the DataSet that the result was stored at

        If a parameter exist in the dataset and it's not in the results
        dictionary, "Null" values are inserted.

        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.

        It is an error to add results to a completed DataSet.
        """

        if not self._started:
            self._perform_start_actions()
            self._started = True

        # TODO: Make this check less fugly
        for param in results.keys():
            if self.paramspecs[param].depends_on != '':
                deps = self.paramspecs[param].depends_on.split(', ')
                for dep in deps:
                    if dep not in results.keys():
                        raise ValueError(f'Can not add result for {param}, '
                                         f'since this depends on {dep}, '
                                         'which is not being added.')

        if self.completed:
            raise CompletedError

        self.dsi.store_results(
            {k: [v] for k, v in results.items()})

    def add_results(self, results: List[Dict[str, VALUE]]) -> int:
        """
        Adds a sequence of results to the DataSet.

        Args:
            results: list of name-value dictionaries where each dictionary
                provides the values for the parameters in that result. If some
                parameters are missing the corresponding values are assumed
                to be None

        Returns:
            the index in the DataSet that the **first** result was stored at

        It is an error to provide a value for a key or keyword that is not
        the name of a parameter in this DataSet.

        It is an error to add results to a completed DataSet.
        """

        if not self._started:
            self._perform_start_actions()
            self._started = True

        if self.completed:
            raise CompletedError

        expected_keys = tuple(frozenset.union(*[frozenset(d) for d in results]))
        values = [[d.get(k, None) for k in expected_keys] for d in results]
        values_transposed = list(map(list, zip(*values)))

        len_before_add = self.dsi.retrieve_number_of_results()

        self.dsi.store_results(
            {k: v for k, v in zip(expected_keys, values_transposed)})

        return len_before_add

    def get_data(self,
                 *params: Union[str, ParamSpec, _BaseParameter],
                 start: Optional[int] = None,
                 end: Optional[int] = None) -> List[List[Any]]:
        """
        Returns the values stored in the DataSet for the specified parameters.
        The values are returned as a list of lists, SQL rows by SQL columns,
        e.g. datapoints by parameters. The data type of each element is based
        on the datatype provided when the DataSet was created. The parameter
        list may contain a mix of string parameter names, QCoDeS Parameter
        objects, and ParamSpec objects (as long as they have a `name` field).

        If provided, the start and end arguments select a range of results
        by result count (index). If the range is empty - that is, if the end is
        less than or equal to the start, or if start is after the current end
        of the DataSet – then a list of empty arrays is returned.

        For a more type independent and easier to work with view of the data
        you may want to consider using
        :py:meth:`qcodes.dataset.data_export.get_data_by_id`

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

        results_iterator = self.dsi.replay_results(start=start, stop=end)

        data = [list(chain.from_iterable([result[p]
                                          for p in valid_param_names]))
                for result in results_iterator]

        return data

    def get_values(self, param_name: str) -> List[List[Any]]:
        """
        Get the values (i.e. not NULLs) of the specified parameter
        """
        if param_name not in self.parameters:
            raise ValueError('Unknown parameter, not in this DataSet')

        # This is a naive implementation, and should probably substituted by
        # a call to dsi.retrieve_results once that is implemented
        values = self.get_data(param_name)

        # Skipping the "None" values, for example, "NULL"s from SQLite
        values = [val for val in values
                  for subval in val
                  if subval is not None]

        return values

    def get_setpoints(self, param_name: str) -> Dict[str, List[List[Any]]]:
        """
        Get the setpoints for the specified parameter

        Args:
            param_name: The name of the parameter for which to get the
                setpoints
        """
        if param_name not in self.parameters:
            raise ValueError('Unknown parameter, not in this DataSet')

        sp_names_str = self.paramspecs[param_name].depends_on
        if sp_names_str == '':
            raise ValueError(f'Parameter {param_name} has no setpoints.')

        sp_names = sp_names_str.split(', ')

        # This is a naive implementation, and should probably be substituted by
        # a call to dsi.retrieve_results once that is implemented

        results_iterator = self.dsi.replay_results()

        setpoints: Dict[str, List[List[Any]]]
        setpoints = defaultdict(list)  # we are going to accumulate values

        for result in results_iterator:
            # Skipping the setpoint values for "None" values of the parameter
            # (for example, "NULL"s from SQLite)
            param_result_subitem_is_value = \
                [subitem is not None for subitem in result[param_name]]
            if all(param_result_subitem_is_value):
                for sp_name in sp_names:
                    setpoints[sp_name].append(result[sp_name])  # type:ignore

        return setpoints

    def subscribe(self,
                  callback: Callable[[Any, int, Optional[Any]], None],
                  min_wait: int = 0,
                  min_count: int = 1,
                  state: Optional[Any] = None,
                  callback_kwargs: Optional[Dict[str, Any]] = None
                  ) -> str:
        subscriber_id = uuid.uuid4().hex
        subscriber = _Subscriber(self, subscriber_id, callback, state,
                                 min_wait, min_count, callback_kwargs)
        self.dsi.writer.subscribers[subscriber_id] = subscriber
        subscriber.start()
        return subscriber_id

    def subscribe_from_config(self, name: str) -> str:
        """
        Subscribe a subscriber defined in the `qcodesrc.json` config file to
        the data of this `DataSet`. The definition can be found at
        `subscription.subscribers`.

        Args:
            name: identifier of the subscriber. Equal to the key of the entry
                in 'qcodesrc.json::subscription.subscribers'.
        """
        subscribers = qcodes.config.subscription.subscribers
        try:
            subscriber_info = getattr(subscribers, name)
        # the dot dict behind the config does not convert the error and
        # actually raises a `KeyError`
        except (AttributeError, KeyError):
            keys = ','.join(subscribers.keys())
            raise RuntimeError(
                f'subscribe_from_config: failed to subscribe "{name}" to DataSet '
                f'from list of subscribers in `qcodesrc.json` (subscriptions.'
                f'subscribers). Chose one of: {keys}')
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
        with atomic(self.dsi.writer.conn) as conn:
            sub = self.dsi.writer.subscribers[uuid]
            remove_trigger(conn, sub.trigger_id)
            sub.schedule_stop()
            sub.join()
            del self.dsi.writer.subscribers[uuid]

    def unsubscribe_all(self):
        """
        Remove all subscribers
        """
        sql = "select * from sqlite_master where type = 'trigger';"
        triggers = atomic_transaction(self.dsi.writer.conn, sql).fetchall()
        with atomic(self.dsi.writer.conn) as conn:
            for trigger in triggers:
                remove_trigger(conn, trigger['name'])
            for sub in self.dsi.writer.subscribers.values():
                sub.schedule_stop()
                sub.join()
            self.dsi.writer.subscribers.clear()

    def get_metadata(self, tag: str) -> Any:
        md = self.dsi.retrieve_meta_data()
        return md.tags[tag]

    def __len__(self) -> int:
        return self.number_of_results

    def __repr__(self) -> str:
        out = []
        heading = f"{self.name} #{self.run_id}"

        if hasattr(self.dsi.reader, 'path_to_db'):
            heading += f'@{self.dsi.reader.path_to_db}'

        out.append(heading)
        out.append("-" * len(heading))
        ps = self.get_parameters()
        if len(ps) > 0:
            for p in ps:
                out.append(f"{p.name} - {p.type}")

        return "\n".join(out)


# public api
def load_by_id(run_id: int, conn: Optional[ConnectionPlus]=None) -> DataSet:
    """
    Load dataset by run id

    If no connection is provided, lookup is performed in the database file that
    is specified in the config.

    Args:
        run_id: run id of the dataset
        conn: connection to the database to load from

    Returns:
        dataset with the given run id
    """
    if run_id is None:
        raise ValueError('run_id has to be a positive integer, not None.')

    conn = conn or connect(get_DB_location())

    d = DataSet(conn=conn, run_id=run_id)
    return d


def load_by_guid(guid: str, conn: Optional[ConnectionPlus]=None) -> DataSet:
    """
    Load a dataset by its GUID

    If no connection is provided, lookup is performed in the database file that
    is specified in the config.

    Args:
        guid: guid of the dataset
        conn: connection to the database to load from

    Returns:
        dataset with the given guid

    Raises:
        NameError if no run with the given GUID exists in the database
        RuntimeError if several runs with the given GUID are found
    """
    conn = conn or connect(get_DB_location())

    # this function raises a RuntimeError if more than one run matches the GUID
    run_id = get_runid_from_guid(conn, guid)

    if run_id == -1:
        raise NameError(f'No run with GUID: {guid} found in database.')

    return DataSet(run_id=run_id, conn=conn)


def load_by_counter(counter: int, exp_id: int,
                    conn: Optional[ConnectionPlus]=None) -> DataSet:
    """
    Load a dataset given its counter in a given experiment

    Lookup is performed in the database file that is specified in the config.

    Args:
        counter: counter of the dataset within the given experiment
        exp_id: id of the experiment where to look for the dataset
        conn: connection to the database to load from. If not provided, a
          connection to the DB file specified in the config is made

    Returns:
        dataset of the given counter in the given experiment
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


def new_data_set(name, exp_id: Optional[int] = None,
                 specs: SPECS = None, values=None,
                 metadata=None, conn=None) -> DataSet:
    """
    Create a new dataset in the currently active/selected database.

    If exp_id is not specified, the last experiment will be loaded by default.

    Args:
        name: the name of the new dataset
        exp_id: the id of the experiments this dataset belongs to, defaults
            to the last experiment
        specs: list of parameters to create this dataset with
        values: the values to associate with the parameters
        metadata: the metadata to associate with the dataset

    Return:
        the newly created dataset
    """
    # note that passing `conn` is a secret feature that is unfortunately used
    # in `Runner` to pass a connection from an existing `Experiment`.
    ds = DataSet(path_to_db=None, run_id=None, conn=conn,
                 name=name, exp_id=exp_id)

    specs = specs or []
    for paramspec in specs:
        ds.add_parameter(paramspec)

    return ds
