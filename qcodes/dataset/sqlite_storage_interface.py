import math
from typing import Union, Dict, Sequence, Optional, Any, Iterator
from numbers import Number
import json

import wrapt
from numpy import ndarray

from qcodes.dataset.data_set import _Subscriber
from qcodes.dataset.descriptions import RunDescriber
from qcodes.utils.helpers import NumpyJSONEncoder
from .data_storage_interface import (
    DataStorageInterface, VALUES, MetaData, _Optional, NOT_GIVEN, SizedIterable)
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
                                        get_parameters,
                                        get_result_counter_from_runid,
                                        get_runid_from_guid,
                                        is_guid_in_database,
                                        make_connection_plus_from,
                                        mark_run_complete)


class IteratorWithLength(wrapt.ObjectProxy, SizedIterable):
    def __init__(self, iterator: Iterator, length: Optional[int]):
        super(IteratorWithLength, self).__init__(iterator)
        self._self_length = length

    def __len__(self):
        return self._self_length


class SqliteStorageInterface(DataStorageInterface):
    """
    """
    def __init__(self, guid: str, *,
                 run_id: Optional[int]=None,
                 conn: Optional[ConnectionPlus]=None,
                 path_to_db: Optional[str]=None,
                 exp_id: Optional[int]=None,
                 name: Optional[str]=None):

        if path_to_db is not None and conn is not None:
            raise ValueError("Both `path_to_db` and `conn` arguments have "
                             "been passed together with non-None values. "
                             "This is not allowed.")

        self.path_to_db = path_to_db or get_DB_location()
        self.conn = make_connection_plus_from(conn) if conn is not None else \
            connect(self.path_to_db)

        # then GUID is ''
        if run_id is not None:
            try:
                guid = get_guid_from_run_id(self.conn, run_id)
            except RuntimeError:
                raise ValueError(f"Run with run_id {run_id} does not "
                                 "exist in the database")

        super().__init__(guid)

        # The following attributes are assigned by create_run OR
        # retrieve_meta_data, depending on what the DataSet constructor wants
        # (i.e. to load or create)
        self.run_id: Optional[int] = None
        self.table_name: Optional[str] = None
        self.counter: Optional[int] = None

        # the following values are only used in create_run. If this instance is
        # constructed to load a run, the following values are ignored,
        # and get overwritten by retrieve_meta_data call
        self.exp_id: Optional[int] = exp_id
        self.name: Optional[str] = name

        # to be implemented later
        self.subscribers: Dict[str, _Subscriber] = {}

    def run_exists(self) -> bool:
        """
        Return the truth value of the statement "a run with the guid of this
        instance exists in the database"
        """
        answers = is_guid_in_database(self.conn, self.guid)
        return answers[self.guid]

    def create_run(self) -> None:
        """
        Create an entry for this run is the database file
        """
        if self.run_exists():
            raise ValueError('Run already exists, can not create it.')

        if self.exp_id is None:
            if len(get_experiments(self.conn)) > 0:
                self.exp_id = get_last_experiment(self.conn)
            else:
                raise ValueError("No experiments found. "
                                 "You can start a new one with:"
                                 " new_experiment(name, sample_name)")

        self.name = self.name or "dataset"

        _, self.run_id, self.table_name = create_run(
            self.conn, self.exp_id, self.name, self.guid)

        self.counter = get_result_counter_from_runid(self.conn, self.run_id)

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
                               list(values_transposed))

    def store_meta_data(self, *,
                        run_started: _Optional[Optional[float]]=NOT_GIVEN,
                        run_completed: _Optional[Optional[float]]=NOT_GIVEN,
                        run_description: _Optional[RunDescriber]=NOT_GIVEN,
                        snapshot: _Optional[Optional[dict]]=NOT_GIVEN,
                        tags: _Optional[Dict[str, Any]]=NOT_GIVEN,
                        tier: _Optional[int]=NOT_GIVEN) -> None:
        """
        Performs one atomic transaction for all the fields. Each field is
        set by a separate function that should check for inconsistencies and
        raise if it finds an inconsistency
        """
        queries = {self._set_run_completed: run_completed,
                   self._set_run_description: run_description,
                   self._set_tags: tags,
                   self._set_snapshot: snapshot}

        with atomic(self.conn) as conn:
            for func, kwarg in queries.items():
                if kwarg != NOT_GIVEN:
                    func(conn, kwarg)

    def _set_run_completed(self, conn: ConnectionPlus, time: float):
        """
        Set the complete_timestamp and is_complete. Will raise if the former
        is not-NULL or if the latter is 1
        """
        if not is_pristine_run(conn, self.run_id):
            raise ValueError(f'Can not write run_completed to GUID {self.guid}'
                              ', that run has already been completed.')

        mark_run_complete(conn, completion_time=time, run_id=self.run_id)

    def _set_run_description(self, conn: ConnectionPlus, desc: RunDescriber):
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
        iterator_length = max(last - first + 1, 0)

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

        md = MetaData(run_description=desc,
                      run_started=run_started,
                      run_completed=run_completed,
                      tags=tags,
                      snapshot=snapshot,
                      name=name,
                      exp_name=exp_name,
                      sample_name=sample_name)

        return md
