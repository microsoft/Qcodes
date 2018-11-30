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
                                        generate_guid,
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
            connect(self.path_to_db)

        self._run_id = run_id

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
                                       generate_guid())

            self._run_id = run_id

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

    def retrieve_meta_data(self) -> MetaData:
        raise NotImplementedError
