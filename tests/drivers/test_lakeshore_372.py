from __future__ import annotations

import logging
import time
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pytest
from typing_extensions import ParamSpec

from qcodes.instrument import InstrumentBase
from qcodes.instrument_drivers.Lakeshore import LakeshoreModel372
from qcodes.instrument_drivers.Lakeshore.lakeshore_base import (
    LakeshoreBaseSensorChannel,
)
from qcodes.logger import get_instrument_logger
from qcodes.utils import QCoDeSDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

VISA_LOGGER = ".".join((InstrumentBase.__module__, "com", "visa"))

P = ParamSpec("P")
T = TypeVar("T")


class MockVisaInstrument:
    """
    Mixin class that overrides write_raw and ask_raw to simulate an
    instrument.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.visa_log = get_instrument_logger(self, VISA_LOGGER)  # type: ignore[arg-type]

        # This base class mixin holds two dictionaries associated with the
        # pyvisa_instrument.write()
        self.cmds: dict[str, Callable[..., Any]] = {}
        # and pyvisa_instrument.query() functions
        self.queries: dict[str, Callable[..., Any]] = {}
        # the keys are the issued VISA commands like '*IDN?' or '*OPC'
        # the values are the corresponding methods to be called on the mock
        # instrument.

        # To facilitate the definition there are the decorators `@query' and
        # `@command`. These attach an attribute to the method, so that the
        # dictionaries can be filled here in the constructor. (This is
        # borderline abusive, but makes a it easy to define mocks)
        func_names = dir(self)
        # cycle through all methods
        for func_name in func_names:
            with warnings.catch_warnings():
                if func_name == "_name":
                    # silence warning when getting deprecated attribute
                    warnings.simplefilter("ignore", category=QCoDeSDeprecationWarning)

                f = getattr(self, func_name)
                # only add for methods that have such an attribute
                with suppress(AttributeError):
                    self.queries[getattr(f, "query_name")] = f
                with suppress(AttributeError):
                    self.cmds[getattr(f, "command_name")] = f

    def write_raw(self, cmd) -> None:
        cmd_parts = cmd.split(" ")
        cmd_str = cmd_parts[0].upper()
        if cmd_str in self.cmds:
            args = "".join(cmd_parts[1:])
            self.visa_log.debug(f"Query: {cmd} for command {cmd_str} with args {args}")
            self.cmds[cmd_str](args)
        else:
            super().write_raw(cmd)  # type: ignore[misc]

    def ask_raw(self, cmd) -> Any:
        query_parts = cmd.split(" ")
        query_str = query_parts[0].upper()
        if query_str in self.queries:
            args = "".join(query_parts[1:])
            self.visa_log.debug(
                f"Query: {cmd} for command {query_str} with args {args}"
            )
            response = self.queries[query_str](args)
            self.visa_log.debug(f"Response: {response}")
            return response
        else:
            return super().ask_raw(cmd)  # type: ignore[misc]


def query(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        func.query_name = name.upper()  # type: ignore[attr-defined]
        return func

    return wrapper


def command(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        func.command_name = name.upper()  # type: ignore[attr-defined]
        return func

    return wrapper


class DictClass:
    def __init__(self, **kwargs):
        # https://stackoverflow.com/questions/16237659/python-how-to-implement-getattr
        super().__setattr__("_attrs", kwargs)

        for kwarg, value in kwargs.items():
            self._attrs[kwarg] = value

    def __getattr__(self, attr):
        try:
            return self._attrs[attr]
        except KeyError as e:
            raise AttributeError from e

    def __setattr__(self, name: str, value: Any) -> None:
        self._attrs[name] = value


class LakeshoreModel372Mock(MockVisaInstrument, LakeshoreModel372):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # initial values
        self.heaters: dict[str, DictClass] = {}
        self.heaters["0"] = DictClass(
            P=1,
            I=2,
            D=3,
            mode=5,
            input_channel=2,
            powerup_enable=0,
            polarity=0,
            use_filter=0,
            delay=1,
            output_range=0,
            setpoint=4,
        )
        self.heaters["1"] = DictClass(
            P=1,
            I=2,
            D=3,
            mode=5,
            input_channel=2,
            powerup_enable=0,
            polarity=0,
            use_filter=0,
            delay=1,
            output_range=0,
            setpoint=4,
        )
        self.heaters["2"] = DictClass(
            P=1,
            I=2,
            D=3,
            mode=5,
            input_channel=2,
            powerup_enable=0,
            polarity=0,
            use_filter=0,
            delay=1,
            output_range=0,
            setpoint=4,
        )

        self.channel_mock = {
            str(i): DictClass(
                tlimit=i,
                T=4,
                enabled=1,  # True
                dwell=100,
                pause=3,
                curve_number=0,
                temperature_coefficient=1,  # 'negative',
                excitation_mode=0,  #'voltage',
                excitation_range_number=1,
                auto_range=0,  #'off',
                range=5,  #'200 mOhm',
                current_source_shunted=0,  # False,
                units=1,
            )  #'kelvin')
            for i in range(1, 17)
        }

        # simulate delayed heating
        self.simulate_heating = False
        self.start_heating_time = time.perf_counter()

    def start_heating(self):
        self.start_heating_time = time.perf_counter()
        self.simulate_heating = True

    def get_t_when_heating(self):
        """
        Simply define a fixed setpoint of 4 k for now
        """
        delta = abs(time.perf_counter() - self.start_heating_time)
        # make it simple to start with: linear ramp 1K per second
        # start at 7K.
        return max(4, 7 - delta)

    @query("KRDG?")
    def temperature(self, output):
        chan = self.channel_mock[output]
        if self.simulate_heating:
            return self.get_t_when_heating()
        return f"{chan.T}"


def instrument_fixture(
    scope: Literal["session", "package", "module", "class", "function"] = "function",
    name=None,
):
    def wrapper(func):
        @pytest.fixture(scope=scope, name=name)
        def wrapped_fixture():
            inst = func()
            try:
                yield inst
            finally:
                inst.close()

        return wrapped_fixture

    return wrapper


@instrument_fixture(scope="function")
def lakeshore_372():
    return LakeshoreModel372Mock(
        "lakeshore_372_fixture",
        "GPIB::3::INSTR",
        pyvisa_sim_file="lakeshore_model372.yaml",
        device_clear=False,
    )


def test_pid_set(lakeshore_372) -> None:
    ls = lakeshore_372
    P, I, D = 1, 2, 3  # noqa  E741
    for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
        h.P(P)
        h.I(I)
        h.D(D)
        assert (h.P(), h.I(), h.D()) == (P, I, D)


def test_output_mode(lakeshore_372) -> None:
    ls = lakeshore_372
    mode = "off"
    input_channel = 1
    powerup_enable = True
    polarity = "unipolar"
    use_filter = True
    delay = 1
    for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
        h.mode(mode)
        h.input_channel(input_channel)
        h.powerup_enable(powerup_enable)
        h.polarity(polarity)
        h.use_filter(use_filter)
        h.delay(delay)
        assert h.mode() == mode
        assert h.input_channel() == input_channel
        assert h.powerup_enable() == powerup_enable
        assert h.polarity() == polarity
        assert h.use_filter() == use_filter
        assert h.delay() == delay


def test_range(lakeshore_372) -> None:
    ls = lakeshore_372
    output_range = "10mA"
    for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
        h.output_range(output_range)
        assert h.output_range() == output_range


def test_tlimit(lakeshore_372) -> None:
    ls = lakeshore_372
    tlimit = 5.1
    for ch in ls.channels:
        ch.t_limit(tlimit)
        assert ch.t_limit() == tlimit


def test_setpoint(lakeshore_372) -> None:
    ls = lakeshore_372
    setpoint = 5.1
    for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
        h.setpoint(setpoint)
        assert h.setpoint() == setpoint


def test_select_range_limits(lakeshore_372) -> None:
    h = lakeshore_372.sample_heater
    ranges = list(range(1, 9))
    h.range_limits(ranges)

    for i in ranges:
        h.set_range_from_temperature(i - 0.5)
        assert h.output_range() == h.INVERSE_RANGES[i]

    h.set_range_from_temperature(ranges[-1] + 0.5)
    assert h.output_range() == h.INVERSE_RANGES[len(ranges)]


def test_set_and_wait_unit_setpoint_reached(lakeshore_372) -> None:
    ls = lakeshore_372
    ls.sample_heater.setpoint(4)
    ls.start_heating()
    ls.sample_heater.wait_until_set_point_reached()


def test_blocking_t(lakeshore_372) -> None:
    ls = lakeshore_372
    h = lakeshore_372.sample_heater
    ranges = list(range(1, 9))
    h.range_limits(ranges)
    ls.start_heating()
    h.blocking_t(4)


def test_get_term_sum() -> None:
    available_terms = [0, 1, 2, 4, 8, 16, 32]

    assert [32, 8, 2, 1] == LakeshoreBaseSensorChannel._get_sum_terms(
        available_terms, 1 + 2 + 8 + 32
    )

    assert [32] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 32)

    assert [16, 4, 1] == LakeshoreBaseSensorChannel._get_sum_terms(
        available_terms, 1 + 4 + 16
    )

    assert [0] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 0)


def test_get_term_sum_with_some_powers_of_2_omitted() -> None:
    available_terms = [0, 16, 32]

    assert [32, 16] == LakeshoreBaseSensorChannel._get_sum_terms(
        available_terms, 16 + 32
    )

    assert [32] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 32)

    assert [0] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 0)


def test_get_term_sum_returns_empty_list() -> None:
    available_terms = [0, 16, 32]

    assert [] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 15)


def test_get_term_sum_when_zero_is_not_in_available_terms() -> None:
    available_terms = [16, 32]

    assert [] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 3)

    # Note that `_get_sum_terms` expects '0' to be in the available_terms,
    # hence for this particular case it will still return a list with '0' in
    # it although that '0' is not part of the available_terms
    assert [0] == LakeshoreBaseSensorChannel._get_sum_terms(available_terms, 0)
