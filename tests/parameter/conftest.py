from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar

import pytest

import qcodes.validators as vals
from qcodes.instrument_drivers.mock_instruments import DummyChannelInstrument
from qcodes.parameters import ParamDataType, Parameter, ParamRawDataType

if TYPE_CHECKING:
    from collections.abc import Generator

    from qcodes.instrument import InstrumentBase

T = TypeVar("T")

NOT_PASSED: Literal["NOT_PASSED"] = "NOT_PASSED"


@pytest.fixture(params=(True, False, NOT_PASSED))
def snapshot_get(request: pytest.FixtureRequest) -> bool | Literal["NOT_PASSED"]:
    return request.param


@pytest.fixture(params=(True, False, NOT_PASSED))
def snapshot_value(request: pytest.FixtureRequest) -> bool | Literal["NOT_PASSED"]:
    return request.param


@pytest.fixture(params=(None, False, NOT_PASSED))
def get_cmd(
    request: pytest.FixtureRequest,
) -> None | Literal[False] | Literal["NOT_PASSED"]:
    return request.param


@pytest.fixture(params=(True, False, NOT_PASSED))
def get_if_invalid(request: pytest.FixtureRequest) -> bool | Literal["NOT_PASSED"]:
    return request.param


@pytest.fixture(params=(True, False, None, NOT_PASSED))
def update(request: pytest.FixtureRequest) -> bool | None | Literal["NOT_PASSED"]:
    return request.param


@pytest.fixture(params=(True, False))
def cache_is_valid(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(name="dummy_instrument")
def _make_dummy_instrument() -> Generator[DummyChannelInstrument, None, None]:
    instr = DummyChannelInstrument("dummy")
    yield instr
    instr.close()


class GettableParam(Parameter):
    """ Parameter that keeps track of number of get operations"""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._get_count = 0

    def get_raw(self) -> int:
        self._get_count += 1
        return 42


class BetterGettableParam(Parameter):
    """ Parameter that keeps track of number of get operations,
        But can actually store values"""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._get_count = 0

    def get_raw(self) -> Any:
        self._get_count += 1
        return self.cache.raw_value


class SettableParam(Parameter):
    """ Parameter that keeps track of number of set operations"""
    def __init__(self, *args: Any, **kwargs: Any):
        self._set_count = 0
        super().__init__(*args, **kwargs)

    def set_raw(self, value: Any) -> None:
        self._set_count += 1


class OverwriteGetParam(Parameter):
    """ Parameter that overwrites get."""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._value = 42
        self.set_count = 0
        self.get_count = 0

    def get(self) -> int:
        self.get_count += 1
        return self._value


class OverwriteSetParam(Parameter):
    """ Parameter that overwrites set."""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._value = 42
        self.set_count = 0
        self.get_count = 0

    def set(self, value: Any) -> None:
        self.set_count += 1
        self._value = value


class GetSetRawParameter(Parameter):
    """ Parameter that implements get and set raw"""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def get_raw(self) -> ParamRawDataType:
        return self.cache.raw_value

    def set_raw(self, value: ParamRawDataType) -> None:
        pass


class BookkeepingValidator(vals.Validator[T]):
    """
    Validator that keeps track of what it validates
    """

    def __init__(
        self, min_value: float = -float("inf"), max_value: float = float("inf")
    ):
        self.values_validated: list[T] = []

    def validate(self, value: T, context: str = "") -> None:
        self.values_validated.append(value)

    is_numeric = True

class MemoryParameter(Parameter):
    def __init__(self, get_cmd: None | Callable[[], Any] = None, **kwargs: Any):
        self.set_values: list[Any] = []
        self.get_values: list[Any] = []
        super().__init__(set_cmd=self.add_set_value,
                         get_cmd=self.create_get_func(get_cmd), **kwargs)

    def add_set_value(self, value: ParamDataType) -> None:
        self.set_values.append(value)

    def create_get_func(
        self, func: None | Callable[[], ParamDataType]
    ) -> Callable[[], ParamDataType]:
        def get_func() -> ParamDataType:
            if func is not None:
                val = func()
            else:
                val = self.cache.raw_value
            self.get_values.append(val)
            return val
        return get_func


class VirtualParameter(Parameter):
    def __init__(self, name: str, param: Parameter, **kwargs: Any):
        self._param = param
        super().__init__(name=name, **kwargs)

    @property
    def underlying_instrument(self) -> InstrumentBase | None:
        return self._param.instrument

    def get_raw(self) -> ParamRawDataType:
        return self._param.get()


blank_instruments = (
    None,  # no instrument at all
    namedtuple('noname', '')(),  # no .name
    namedtuple('blank', 'name')('')  # blank .name
)
named_instrument = namedtuple('yesname', 'name')('astro')


class ParameterMemory:

    def __init__(self) -> None:
        self._value: Any | None = None

    def get(self) -> ParamDataType:
        return self._value

    def set(self, value: ParamDataType) -> None:
        self._value = value

    def set_p_prefixed(self, val: int) -> None:
        self._value = f'PVAL: {val:d}'

    @staticmethod
    def parse_set_p(val: int) -> str:
        return f'{val:d}'

    @staticmethod
    def strip_prefix(val: str) -> int:
        return int(val[6:])
