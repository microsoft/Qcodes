from typing import Union, Dict, Sequence, Optional, Any
from numbers import Number
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
                                        get_metadata_from_run_id,
                                        make_connection_plus_from,
                                        run_exists)


class SqliteStorageInterface(DataStorageInterface):
    """
    """
    def __init__(self, guid: str, *,
                 conn: Optional[ConnectionPlus]=None,
                 path_to_db: Optional[str]=None,
                 exp_id: Optional[int]=None,
                 run_id: Optional[int]=None,
                 name: Optional[str]=None):

        super().__init__(guid)

        if path_to_db is not None and conn is not None:
            raise ValueError("Both `path_to_db` and `conn` arguments have "
                             "been passed together with non-None values. "
                             "This is not allowed.")

        self._path_to_db = path_to_db or get_DB_location()
        self.conn = make_connection_plus_from(conn) if conn is not None else \
            connect(self._path_to_db)

        self.run_id = run_id

        if run_id is not None:
            if not run_exists(self.conn, run_id):
                raise ValueError(f"Run with run_id {run_id} does not exist in "
                                 f"the database")

        else:
            # Actually perform all the side effects needed for the creation
            # of a new dataset
            if exp_id is None:
                if len(get_experiments(self.conn)) > 0:
                    exp_id = get_last_experiment(self.conn)
                else:
                    raise ValueError("No experiments found."
                                     "You can start a new one with:"
                                     " new_experiment(name, sample_name)")
            name = name or "dataset"
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
        raise NotImplementedError

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

    def get_run_table_row_full(self) -> Dict:
        """
        Retrieve the full run table row

        Returns:
            A dict with the column names as keys and the raw values as values
        """
        sql = "SELECT * FROM runs WHERE run_id = ?"
        cursor = self.conn.cursor()
        cursor.execute(sql, (self.run_id,))
        return dict(cursor.fetchall()[0])

    def get_run_table_row_non_standard(self) -> Dict:
        """
        Retrieve all the non-standard (i.e. metadata) columns
        """
        return get_metadata_from_run_id(self.conn, self.run_id)

    def retrieve_meta_data(self) -> MetaData:
        run_info = self.get_run_table_row_full()
        run_extra_info = self.get_run_table_row_non_standard()

        desc = RunDescriber.from_json(run_info['run_description'])
        run_started = run_info['run_timestamp']
        run_completed = run_info['completed_timestamp']
        # snapshot column may not exist, this will be changed in further db
        # versions so that this can be substituted by `run_info['snapshot']`
        snapshot = run_info.get('snapshot', None)
        tags = run_extra_info

        md = MetaData(run_description=desc,
                      run_started=run_started,
                      run_completed=run_completed,
                      tags=tags,
                      snapshot=snapshot)

        return md
