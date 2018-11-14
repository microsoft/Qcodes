from typing import Union, Dict, Sequence
from numbers import Number
from numpy import ndarray

from .data_storage_interface import (
    DataStorageInterface, VALUES)
from .sqlite_base import (
    connect, select_one_where, insert_values, insert_many_values)
from qcodes.dataset.database import get_DB_location


class SqliteStorageInterface(DataStorageInterface):
    """
    """
    def __init__(self, guid: str):
        super().__init__(guid)
        self.conn = connect(get_DB_location())

        self.table_name = select_one_where(
            self.conn, "runs", "name", "guid", self.guid)

    def store_results(self, results: Dict[str, VALUES]):
        self._validate_results_dict(results)
        if len(results.values()[0]) == 1:
            insert_values(self.conn, self.table_name,
                          list(results.keys()),
                          list(results.values()))
        else:
            insert_many_values(self.conn, self.table_name,
                               results.keys(),
                               results.values())
