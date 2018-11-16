from typing import Union, Dict, Sequence
from numbers import Number
from numpy import ndarray
from abc import ABC, abstractmethod


VALUE = Union[str, Number, ndarray, bool, None]
VALUES = Union[Sequence[VALUE], ndarray]


class DataStorageInterface(ABC):
    """
    """
    def __init__(self, guid: str):
        self.guid = guid

    @abstractmethod
    def store_results(self, results: Dict[str, VALUES]) -> None:
        pass


def rows_from_results(results: Dict[str, VALUES]):
    """
    Helper function returning an iterator yielding the rows as tuples.
    Useful for file writing backends that are "row-centric", such as SQLite
    and GNUPlot
    """
    for values in zip(*results.values()):
        yield values
