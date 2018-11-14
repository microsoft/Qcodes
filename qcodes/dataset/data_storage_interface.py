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
    def store_results(self, results: Dict[str, VALUES]):
        pass
