from collections import namedtuple

import pytest

from qcodes.instrument.parameter import Parameter
import qcodes.utils.validators as vals

NOT_PASSED = 'NOT_PASSED'


@pytest.fixture(params=(True, False, NOT_PASSED))
def snapshot_get(request):
    return request.param


@pytest.fixture(params=(True, False, NOT_PASSED))
def snapshot_value(request):
    return request.param


@pytest.fixture(params=(None, False, NOT_PASSED))
def get_cmd(request):
    return request.param


@pytest.fixture(params=(True, False, NOT_PASSED))
def get_if_invalid(request):
    return request.param


@pytest.fixture(params=(True, False, None, NOT_PASSED))
def update(request):
    return request.param


@pytest.fixture(params=(True, False))
def cache_is_valid(request):
    return request.param


class GettableParam(Parameter):
    """ Parameter that keeps track of number of get operations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get_count = 0

    def get_raw(self):
        self._get_count += 1
        return 42


class BetterGettableParam(Parameter):
    """ Parameter that keeps track of number of get operations,
        But can actually store values"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get_count = 0

    def get_raw(self):
        self._get_count += 1
        return self.cache._raw_value


class SettableParam(Parameter):
    """ Parameter that keeps track of number of set operations"""
    def __init__(self, *args, **kwargs):
        self._set_count = 0
        super().__init__(*args, **kwargs)

    def set_raw(self, value):
        self._set_count += 1


class OverwriteGetParam(Parameter):
    """ Parameter that overwrites get."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._value = 42
        self.set_count = 0
        self.get_count = 0

    def get(self):
        self.get_count += 1
        return self._value


class OverwriteSetParam(Parameter):
    """ Parameter that overwrites set."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._value = 42
        self.set_count = 0
        self.get_count = 0

    def set(self, value):
        self.set_count += 1
        self._value = value


class GetSetRawParameter(Parameter):
    """ Parameter that implements get and set raw"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):
        return self.cache.raw_value

    def set_raw(self, value):
        pass


class BookkeepingValidator(vals.Validator):
    """
    Validator that keeps track of what it validates
    """
    def __init__(self, min_value=-float("inf"), max_value=float("inf")):
        self.values_validated = []

    def validate(self, value, context=''):
        self.values_validated.append(value)

    is_numeric = True


class MemoryParameter(Parameter):
    def __init__(self, get_cmd=None, **kwargs):
        self.set_values = []
        self.get_values = []
        super().__init__(set_cmd=self.add_set_value,
                         get_cmd=self.create_get_func(get_cmd), **kwargs)

    def add_set_value(self, value):
        self.set_values.append(value)

    def create_get_func(self, func):
        def get_func():
            if func is not None:
                val = func()
            else:
                val = self.cache.raw_value
            self.get_values.append(val)
            return val
        return get_func


blank_instruments = (
    None,  # no instrument at all
    namedtuple('noname', '')(),  # no .name
    namedtuple('blank', 'name')('')  # blank .name
)
named_instrument = namedtuple('yesname', 'name')('astro')


class ParameterMemory:

    def __init__(self):
        self._value = None

    def get(self):
        return self._value

    def set(self, value):
        self._value = value

    def set_p_prefixed(self, val):
        self._value = f'PVAL: {val:d}'

    @staticmethod
    def parse_set_p(val):
        return f'{val:d}'

    @staticmethod
    def strip_prefix(val):
        return int(val[6:])
