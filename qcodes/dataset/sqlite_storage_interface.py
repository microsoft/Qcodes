from typing import Union, Dict, Sequence, Optional, Any
from numbers import Number
import json

from numpy import ndarray

from qcodes.dataset.descriptions import RunDescriber
from .data_storage_interface import (
    DataStorageInterface, VALUES, MetaData, _Optional, NOT_GIVEN)
from .sqlite_base import (
    connect, select_one_where, insert_values, insert_many_values)
from qcodes.dataset.database import get_DB_location
from qcodes.dataset.sqlite_base import (ConnectionPlus,
                                        create_run,
                                        get_experiments,
                                        get_last_experiment,
                                        get_number_of_results,
                                        get_metadata_from_run_id,
                                        get_runid_from_guid,
                                        is_guid_in_database,
                                        make_connection_plus_from)


class SqliteStorageInterface(DataStorageInterface):
    """
    """
    def __init__(self, guid: str, *,
                 conn: Optional[ConnectionPlus]=None,
                 path_to_db: Optional[str]=None,
                 exp_id: Optional[int]=None,
                 name: Optional[str]=None):

        super().__init__(guid)

        if path_to_db is not None and conn is not None:
            raise ValueError("Both `path_to_db` and `conn` arguments have "
                             "been passed together with non-None values. "
                             "This is not allowed.")

        self.path_to_db = path_to_db or get_DB_location()
        self.conn = make_connection_plus_from(conn) if conn is not None else \
            connect(self.path_to_db)

        # The run_id is assigned by create_run OR
        # retrieve_meta_data, depending on what the DataSet constructor wants
        # (i.e. to load or create)
        self.run_id: Optional[int] = None

        # the following values are only used in create_run. If this instance is
        # constructed to load a run, the following values are ignored
        self.exp_id = exp_id
        self.name = name

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
                exp_id = get_last_experiment(self.conn)
            else:
                raise ValueError("No experiments found. "
                                 "You can start a new one with:"
                                 " new_experiment(name, sample_name)")
        name = self.name or "dataset"
        _, run_id, __ = create_run(self.conn, exp_id, name,
                                   self.guid)

        self.run_id = run_id

    def store_results(self, results: Dict[str, VALUES]):
        self._validate_results_dict(results)
        if len(next(iter(results.values()))) == 1:
            insert_values(self.ds.conn, self.table_name,
                          list(results.keys()),
                          [v[0] for v in results.values()])
        else:
            values_transposed = list(map(list, zip(*results.values())))
            insert_many_values(self.ds.conn, self.table_name,
                               list(results.keys()),
                               list(values_transposed))

    def retrieve_number_of_results(self) -> int:
        return get_number_of_results(self.conn, self.guid)

    def retrieve_results(self, params,
                         start=None,
                         stop=None) -> Dict[str, ndarray]:
        raise NotImplementedError

    def store_meta_data(self, *,
                        run_started: _Optional[Optional[float]]=NOT_GIVEN,
                        run_completed: _Optional[Optional[float]]=NOT_GIVEN,
                        run_descriptor: _Optional[RunDescriber]=NOT_GIVEN,
                        snapshot: _Optional[Optional[dict]]=NOT_GIVEN,
                        tags: _Optional[Dict[str, Any]]=NOT_GIVEN,
                        tier: _Optional[int]=NOT_GIVEN) -> None:
        raise NotImplementedError

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
              SELECT sample_name, experiments.name, start_time, end_time
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
        snapshot = json.loads(snapshot_raw) if snapshot_raw else None
        tags = run_extra_info
        name = run_info['name']
        exp_name = run_info['exp_name']
        sample_name = run_info['sample_name']

        md = MetaData(run_description=desc,
                      run_started=run_started,
                      run_completed=run_completed,
                      tags=tags,
                      snapshot=snapshot,
                      name=name,
                      exp_name=exp_name,
                      sample_name=sample_name)

        return md
