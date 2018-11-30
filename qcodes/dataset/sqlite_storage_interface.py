from typing import Union, Dict, Sequence, Optional, Any
from numbers import Number
from numpy import ndarray

from qcodes.dataset.descriptions import RunDescriber
from .data_storage_interface import (
    DataStorageInterface, VALUES, MetaData, _Optional, NOT_GIVEN)
from .sqlite_base import (
    connect, select_one_where, insert_values, insert_many_values)
from qcodes.dataset.database import get_DB_location


class SqliteStorageInterface(DataStorageInterface):
    """
    """
    def __init__(self, guid: str, ds):
        super().__init__(guid, ds)
        self.ds = ds
        self.table_name = select_one_where(self.ds.conn, "runs",
                                           "result_table_name",
                                           "guid", self.guid)

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
