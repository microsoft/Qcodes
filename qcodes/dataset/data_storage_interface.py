from typing import Union, Dict, Sequence, Any, Optional, TypeVar
from typing_extensions import Final
from numbers import Number
from numpy import ndarray
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from qcodes.dataset.descriptions import RunDescriber

VALUE = Union[str, Number, ndarray, bool, None]
VALUES = Union[Sequence[VALUE], ndarray]
# Introducing the `NOT_GIVEN` constant, intended to use as a default
# value of method arguments that may be left out but also may take `None`
# as a value.
NOT_GIVEN: Final = ('This is a placeholder for arguments that have not '
                    'been supplied.')
# define an `_Optional` type that allows to provide optional arguments to
# a method that may also be `NOT_GIVEN`
T = TypeVar('T')
_Optional = Union[T, str]


@dataclass
class MetaData():
    run_description: RunDescriber
    run_started: Optional[float] = None
    run_completed: Optional[float] = None
    snapshot: Optional[dict] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    tier: int = 1


class DataStorageInterface(ABC):
    """
    """
    VERSION: Final = 0

    def __init__(self, guid: str):
        self.guid = guid

    @abstractmethod
    def store_results(self, results: Dict[str, VALUES]) -> None:
        pass

    @abstractmethod
    def retrieve_number_of_results(self) -> int:
        pass

    @abstractmethod
    def retrieve_results(self, params,
                         start=None,
                         stop=None) -> Dict[str, ndarray]:
        pass

    @abstractmethod
    def store_meta_data(self, *,
                        run_started: _Optional[Optional[float]]=NOT_GIVEN,
                        run_completed: _Optional[Optional[float]]=NOT_GIVEN,
                        run_descriptor: _Optional[RunDescriber]=NOT_GIVEN,
                        snapshot: _Optional[Optional[dict]]=NOT_GIVEN,
                        tags: _Optional[Dict[str, Any]]=NOT_GIVEN,
                        tier: _Optional[int]=NOT_GIVEN) -> None:
        pass

    @abstractmethod
    def retrieve_meta_data(self) -> MetaData:
        pass

    @staticmethod
    def _validate_results_dict(results: Dict[str, VALUES]):
        assert len(results) != 0
        assert len(set(len(v) for k, v in results.items())) == 1
        assert len(next(iter(results.values()))) != 0


def rows_from_results(results: Dict[str, VALUES]):
    """
    Helper function returning an iterator yielding the rows as tuples.
    Useful for file writing backends that are "row-centric", such as SQLite
    and GNUPlot
    """
    for values in zip(*results.values()):
        yield values
