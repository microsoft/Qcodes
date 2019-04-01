import math
from typing import Union, Dict, Sequence, Optional, Any, Iterator, Callable, \
    TYPE_CHECKING
from numbers import Number
import json
import time

import wrapt
from numpy import ndarray

from qcodes.dataset.descriptions import RunDescriber
from qcodes.utils.helpers import NumpyJSONEncoder
from qcodes.dataset.data_storage_interface import (
    DataReaderInterface, DataWriterInterface, VALUES,
    MetaData, _Optional, NOT_GIVEN, SizedIterable)
from .sqlite_base import (
    connect, select_one_where, insert_values, insert_many_values,
    is_pristine_run, update_run_description, add_meta_data,
    atomic_transaction, _build_data_query)
from qcodes.dataset.database import get_DB_location
from qcodes.dataset.sqlite_base import (add_parameter,
                                        atomic,
                                        ConnectionPlus,
                                        create_run,
                                        get_experiments,
                                        get_guid_from_run_id,
                                        get_last_experiment,
                                        get_number_of_results,
                                        get_metadata_from_run_id,
                                        get_matching_exp_ids,
                                        get_parameters,
                                        get_result_counter_from_runid,
                                        get_runid_from_guid,
                                        is_guid_in_database,
                                        mark_run_complete,
                                        new_experiment,
                                        set_run_timestamp)
from qcodes.dataset.descriptions import RunDescriber

if TYPE_CHECKING:
    from qcodes.dataset.data_set import _Subscriber


class IteratorWithLength(wrapt.ObjectProxy, SizedIterable):
    def __init__(self, iterator: Iterator, length: Optional[int]):
        super(IteratorWithLength, self).__init__(iterator)
        self._self_length = length

    def __len__(self):
        return self._self_length


class SqliteReaderInterface(DataReaderInterface):
    """
    """
    def __init__(self, guid: str, *,
                 conn: Optional[ConnectionPlus]=None):

        if not isinstance(conn, ConnectionPlus):
            raise ValueError("conn must be a QCoDeS ConnectionPlus "
                             f"object. Received {type(conn)}")

        self.path_to_db = get_DB_location()
        self.conn = conn

        super().__init__(guid)

        # The following attributes are assigned in retrieve meta_data
        self.run_id: Optional[int] = None
        self.table_name: Optional[str] = None
        self.counter: Optional[int] = None
        self.name: Optional[str] = None
        self.exp_id: Optional[str] = None

    def run_exists(self) -> bool:
        """
        Return the truth value of the statement "a run with the guid of this
        instance exists in the database"
        """
        if self.guid == '' and self.run_id is None:
            return False

        answers = is_guid_in_database(self.conn, self.guid)
        return answers[self.guid]

    @staticmethod
    def _encode_snapshot(snapshot: dict) -> str:
        return json.dumps(snapshot, cls=NumpyJSONEncoder)

    @staticmethod
    def _decode_snapshot(snapshot: str) -> dict:
        return json.loads(snapshot)

    def retrieve_number_of_results(self) -> int:
        return get_number_of_results(self.conn, self.guid)

    def retrieve_results(self, params: Sequence[str]
                         ) -> Dict[str, Dict[str, ndarray]]:
        raise NotImplementedError

    def replay_results(self,
                       start: Optional[int] = None,
                       stop: Optional[int] = None
                       ) -> SizedIterable[Dict[str, VALUES]]:
        if not self.run_exists():
            raise ValueError(f"No run with guid {self.guid} exists.")

        query = _build_data_query(table_name=self.table_name,
                                  columns=['*'],
                                  start=start,
                                  end=stop)

        cursor = atomic_transaction(self.conn, query)

        # other elements of `description` are None, see docs for info.
        column_names = tuple((t[0] for t in cursor.description))
        # the first column is `id` which is not needed, hence is skipped.
        column_names = column_names[1:]

        # note that first element of `row` is skipped here as well.
        results_iterator = ({key: [value]
                             for key, value in zip(column_names,
                                                   tuple(row)[1:])}
                            for row in cursor)

        # calculate the length of iterator
        start_specified = start is not None
        stop_specified = stop is not None
        first = max((start if start_specified else -math.inf), 1) - 1
        last = min((stop if stop_specified else math.inf) - 1,
                   self.retrieve_number_of_results() - 1)
        iterator_length = int(max(last - first + 1, 0))

        results_iterator = IteratorWithLength(results_iterator,
                                              iterator_length)

        return results_iterator

    def _get_run_table_row_full(self) -> Dict:
        """
        Retrieve the full run table row

        Returns:
            A dict with the column names as keys and the raw values as values
        """
        sql = "SELECT * FROM runs WHERE run_id = ?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (self.run_id,))
        return dict(cursor.fetchall()[0])

    def _get_run_table_row_non_standard(self) -> Dict:
        """
        Retrieve all the non-standard (i.e. metadata) columns
        """
        return get_metadata_from_run_id(self.conn, self.run_id)

    def _get_experiment_table_info(self) -> Dict:
        """
        Get the relevant info from the experiments table
        """
        sql = """
              SELECT sample_name, experiments.name, start_time, end_time,
                  experiments.exp_id
              FROM experiments
              JOIN runs ON runs.exp_id = experiments.exp_id
              WHERE runs.run_id = ?
              """

        cursor = self.conn.cursor()
        cursor.execute(sql, (self.run_id,))
        rows = cursor.fetchall()
        result = dict(rows[0])
        # consistent naming of things is hard...
        result['exp_name'] = result.pop('name')

        return result

    def retrieve_meta_data(self) -> MetaData:
        if not self.run_exists():
            raise ValueError(f"No run with guid {self.guid} exists. Perhaps "
                             "you forgot to call create_run?")

        if self.run_id is None:
            self.run_id = get_runid_from_guid(self.conn, self.guid)

        if self.counter is None:
            self.counter = get_result_counter_from_runid(self.conn,
                                                         self.run_id)

        run_info = self._get_run_table_row_full()
        run_extra_info = self._get_run_table_row_non_standard()
        run_exp_info = self._get_experiment_table_info()
        run_info.update(run_exp_info)

        desc = RunDescriber.from_json(run_info['run_description'])
        run_started = run_info['run_timestamp']
        run_completed = run_info['completed_timestamp']
        # snapshot column may not exist, this will be changed in future db
        # versions so that this can be substituted by `run_info['snapshot']`
        snapshot_raw: Optional[str] = run_info.get('snapshot', None)
        snapshot = self._decode_snapshot(snapshot_raw) if snapshot_raw else None
        tags = run_extra_info
        name = run_info['name']
        exp_name = run_info['exp_name']
        sample_name = run_info['sample_name']

        if self.name is None:
            self.name = name

        if self.table_name is None:
            self.table_name = run_info['result_table_name']

        if self.exp_id is None:
            self.exp_id = run_exp_info['exp_id']

        md = MetaData(run_description=desc.to_json(),
                      run_started=run_started,
                      run_completed=run_completed,
                      tags=tags,
                      snapshot=snapshot,
                      name=name,
                      exp_name=exp_name,
                      sample_name=sample_name)

        return md


class SqliteWriterInterface(DataWriterInterface):
    """
    """
    def __init__(self, guid: str, *,
                 conn: Optional[ConnectionPlus] = None):

        if not isinstance(conn, ConnectionPlus):
            raise ValueError("conn must be a QCoDeS ConnectionPlus "
                             f"object. Received {type(conn)}")

        self.path_to_db = get_DB_location()
        self.conn = conn

        super().__init__(guid)

        # The following attributes are assigned by create_run or resume_run
        self.exp_id: Optional[int] = None
        self.name: Optional[str] = None
        self.run_id: Optional[int] = None
        self.table_name: Optional[str] = None
        self.counter: Optional[int] = None

        # Used by the DataSet
        self.subscribers: Dict[str, '_Subscriber'] = {}

    def create_run(self, **kwargs) -> None:
        """
        Create an entry for this run is the database file. The kwargs may be
        exp_id, exp_name, sample_name, name. The logic for creating new runs
        in existing experiments is as follows: first exp_id is used to look up
        an experiment. If an experiment is not found, nothing happens. Next the
        (exp_name, sample_name) tuple is used to look up an experiment. If an
        experiment is not found, one is CREATED with those two attributes.
        """

        self.name = kwargs.get('name', None) or "dataset"
        self.exp_id = kwargs.get('exp_id', None)
        exp_name = kwargs.get('exp_name', None)
        sample_name = kwargs.get('sample_name', None)

        if sum(1 if v is None else 0 for v in (exp_name, sample_name)) == 1:
            raise ValueError(f'Got values for exp_name: {exp_name} and '
                             f'sample_name: {sample_name}. They must both '
                             'be None or both be not-None.')

        if self.exp_id is None:
            if len(get_experiments(self.conn)) > 0:
                self.exp_id = get_last_experiment(self.conn)
            elif sample_name is None:
                raise ValueError("No experiments found. "
                                 "You can start a new one with:"
                                 " new_experiment(name, sample_name)")
            elif sample_name is not None:
                experiments = get_matching_exp_ids(self.conn,
                                                   name=exp_name,
                                                   sample_name=sample_name)
                if len(experiments) == 0:
                    self.exp_id = new_experiment(self.conn, name=exp_name,
                                                 sample_name=sample_name,
                                                 start_time=time.time())
                else:
                    self.exp_id = experiments[-1]

        with atomic(self.conn) as aconn:

            _, self.run_id, self.table_name = create_run(
                aconn, self.exp_id, self.name, self.guid)

        with atomic(self.conn) as aconn:
            self.counter = get_result_counter_from_runid(aconn, self.run_id)

    def resume_run(self, *args) -> None:
        """
        Args:
            exp_id: experiment id
            run_id: run_id
            name: run name
            table_name: name of results table
            counter: run count for this run in its experiment
        """
        self.exp_id = args[0]
        self.run_id = args[1]
        self.name = args[2]
        self.table_name = args[3]
        self.counter = args[4]

    def prepare_for_storing_results(self) -> None:
        pass

    def store_results(self, results: Dict[str, VALUES]):
        self._validate_results_dict(results)

        if len(next(iter(results.values()))) == 1:
            # in this case, the given dictionary contains single value per key
            insert_values(self.conn, self.table_name,
                          list(results.keys()),
                          [v[0] for v in results.values()])
        else:
            # here, the given dictionary contains multiple values per key
            values_transposed = list(map(list, zip(*results.values())))
            insert_many_values(self.conn, self.table_name,
                               list(results.keys()),
                               list(values_transposed))  # type: ignore

    def store_meta_data(self, metadata: MetaData) -> None:
        """
        Performs one atomic transaction for all the fields. Each field is
        set by a separate function that should check for inconsistencies and
        raise if it finds an inconsistency
        """

        queries: Dict[Callable[[ConnectionPlus, Any], None], _Optional[Any]]
        queries = {self._set_run_completed: metadata.run_completed,
                   self._set_run_started: metadata.run_started,
                   self._set_run_description: metadata.run_description,
                   self._set_tags: metadata.tags,
                   self._set_snapshot: metadata.snapshot}

        with atomic(self.conn) as conn:
            for func, kwarg in queries.items():
                if kwarg != NOT_GIVEN:
                    func(conn, kwarg)

    def _set_run_completed(self, conn: ConnectionPlus, time: float) -> None:
        """
        Set the complete_timestamp and is_complete. Will raise if the former
        is not-NULL or if the latter is 1
        """
        if not is_pristine_run(conn, self.run_id):
            raise ValueError(f'Can not write run_completed to GUID {self.guid}'
                              ', that run has already been completed.')

        mark_run_complete(conn, completion_time=time, run_id=self.run_id)

    def _set_run_started(self, conn: ConnectionPlus, time: float) -> None:
        """
        Set the run_timestamp. Will raise if it has already been set before.
        """
        set_run_timestamp(conn, self.run_id, timestamp=time)

    def _set_run_description(self, conn: ConnectionPlus, desc_str: str) \
            -> None:
        desc = RunDescriber.from_json(desc_str)
        # update the result_table columns and write to layouts and dependencies
        existing_params = get_parameters(conn, self.run_id)
        for param in desc.interdeps.paramspecs:
            if param in existing_params:
                pass
            else:
                add_parameter(conn, self.table_name, param)
        # update the run_description itself
        update_run_description(conn, self.run_id, desc.to_json())

    def _set_tags(self, conn: ConnectionPlus, tags: Dict[str, Any]) -> None:
        for tag, value in tags.items():
            add_meta_data(conn, self.run_id, {tag: value})

    def _set_snapshot(self, conn: ConnectionPlus, snapshot: dict) -> None:
        snapshot_json = self._encode_snapshot(snapshot)
        add_meta_data(conn, self.run_id, {'snapshot': snapshot_json})

    @staticmethod
    def _encode_snapshot(snapshot: dict) -> str:
        return json.dumps(snapshot, cls=NumpyJSONEncoder)

    @staticmethod
    def _decode_snapshot(snapshot: str) -> dict:
        return json.loads(snapshot)
