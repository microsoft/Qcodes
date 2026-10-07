"""
Tests for `qcodes.utils.logger`.
"""

import logging
import os
from copy import copy
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
from pytest import LogCaptureFixture

import qcodes as qc
from qcodes import logger
from qcodes.instrument import Instrument, InstrumentModule
from qcodes.instrument.ip_to_visa import IPToVisa
from qcodes.instrument.visa import VISA_LOGGER
from qcodes.instrument_drivers.american_magnetics import AMIModel430, AMIModel4303D
from qcodes.instrument_drivers.mock_instruments import (
    DummyChannel,
    DummyChannelInstrument,
    DummyInstrument,
)
from qcodes.instrument_drivers.tektronix import TektronixAWG5208
from qcodes.logger.log_analysis import capture_dataframe
from tests.drivers.test_lakeshore_372 import LakeshoreModel372Mock

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from qcodes.instrument.instrument_base import LoggerScope

TEST_LOG_MESSAGE = "test log message"

NUM_PYTEST_LOGGERS = 4

SHARED_INSTRUMENT_LOGGER_NAME = "qcodes.instrument.instrument_base"
SHARED_VISA_LOGGER_NAME = VISA_LOGGER

_LOGGER = logging.getLogger(__name__)


def scoped_class_logger_name(cls: type[object]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def logger_is_descendant(logger_: logging.Logger, ancestor: logging.Logger) -> bool:
    parent = logger_.parent
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parent
    return False


@pytest.fixture(autouse=True)
def cleanup_started_logger() -> "Generator[None, None, None]":
    # cleanup state left by a test calling start_logger
    root_logger = logging.getLogger()
    existing_handlers = copy(root_logger.handlers)
    yield
    post_test_handlers = copy(root_logger.handlers)
    for handler in post_test_handlers:
        if handler not in existing_handlers:
            handler.close()
            root_logger.removeHandler(handler)
    logger.logger.file_handler = None
    logger.logger.console_handler = None


@pytest.fixture
def awg5208(caplog: LogCaptureFixture) -> "Generator[TektronixAWG5208, None, None]":
    with caplog.at_level(logging.INFO):
        inst = TektronixAWG5208(
            "awg_sim",
            address="GPIB0::1::INSTR",
            pyvisa_sim_file="Tektronix_AWG5208.yaml",
        )

    try:
        yield inst
    finally:
        inst.close()


@pytest.fixture
def model372() -> "Generator[LakeshoreModel372Mock, None, None]":
    inst = LakeshoreModel372Mock(
        "lakeshore_372",
        "GPIB::3::INSTR",
        pyvisa_sim_file="lakeshore_model372.yaml",
        device_clear=False,
    )
    inst.sample_heater.range_limits([0, 0.25, 0.5, 1, 2, 3, 4, 7])
    inst.warmup_heater.range_limits([0, 0.25, 0.5, 1, 2, 3, 4, 7])
    try:
        yield inst
    finally:
        inst.close()


@pytest.fixture()
def AMI430_3D() -> (
    "Generator[tuple[AMIModel4303D, AMIModel430, AMIModel430, AMIModel430], None, None]"
):
    mag_x = AMIModel430(
        "x",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )
    mag_y = AMIModel430(
        "y",
        address="GPIB::2::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )
    mag_z = AMIModel430(
        "z",
        address="GPIB::3::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )
    field_limit: list[Callable[[float, float, float], bool]] = [
        lambda x, y, z: x == 0 and y == 0 and z < 3,
        lambda x, y, z: bool(np.linalg.norm([x, y, z]) < 2),
    ]
    driver = AMIModel4303D("AMI430_3D", mag_x, mag_y, mag_z, field_limit)
    try:
        yield driver, mag_x, mag_y, mag_z
    finally:
        driver.close()
        mag_x.close()
        mag_y.close()
        mag_z.close()


def test_get_log_file_name() -> None:
    fp = logger.logger.get_log_file_name().split(os.sep)
    assert str(os.getpid()) in fp[-1]
    assert logger.logger.PYTHON_LOG_NAME in fp[-1]
    assert fp[-2] == logger.logger.LOGGING_DIR
    assert fp[-3] == ".qcodes"


def test_start_logger() -> None:
    # remove all Handlers
    logger.start_logger()
    assert isinstance(logger.get_console_handler(), logging.Handler)
    assert isinstance(logger.get_file_handler(), logging.Handler)

    console_level = logger.get_level_code(qc.config.logger.console_level)
    file_level = logger.get_level_code(qc.config.logger.file_level)
    console_handler = logger.get_console_handler()
    assert console_handler is not None
    assert console_handler.level == console_level
    file_handler = logger.get_file_handler()
    assert file_handler is not None
    assert file_handler.level == file_level

    assert logging.getLogger().level == logger.get_level_code("NOTSET")


def test_start_logger_twice() -> None:
    root_logger = logging.getLogger()

    assert len(root_logger.handlers) == NUM_PYTEST_LOGGERS

    logger.start_logger()
    logger.start_logger()
    handlers = root_logger.handlers
    # we expect there to be two log handlers file+console
    # plus the existing ones from pytest
    assert len(handlers) == 2 + NUM_PYTEST_LOGGERS


def test_set_level_without_starting_raises() -> None:
    with pytest.raises(RuntimeError), logger.console_level("DEBUG"):
        pass
    assert len(logging.getLogger().handlers) == NUM_PYTEST_LOGGERS


def test_handler_level() -> None:
    logger.start_logger()
    with logger.LogCapture(level=logging.INFO) as logs:
        _LOGGER.debug(TEST_LOG_MESSAGE)
    assert logs.value == ""

    with (
        logger.LogCapture(level=logging.INFO) as logs,
        logger.handler_level(level=logging.DEBUG, handler=logs.string_handler),
    ):
        print(logs.string_handler)
        _LOGGER.debug(TEST_LOG_MESSAGE)
    assert logs.value.strip() == TEST_LOG_MESSAGE


def test_filter_instrument(
    AMI430_3D: tuple[AMIModel4303D, AMIModel430, AMIModel430, AMIModel430],
) -> None:
    driver, mag_x, mag_y, _mag_z = AMI430_3D

    logger.start_logger()

    # filter one instrument
    driver.cartesian((0, 0, 0))
    with (
        logger.LogCapture(level=logging.DEBUG) as logs,
        logger.filter_instrument(mag_x, handler=logs.string_handler),
    ):
        driver.cartesian((0, 0, 1))
    for line in logs.value.splitlines():
        assert "[x(AMIModel430)]" in line
        assert "[y(AMIModel430)]" not in line
        assert "[z(AMIModel430)]" not in line

    # filter multiple instruments
    driver.cartesian((0, 0, 0))
    with (
        logger.LogCapture(level=logging.DEBUG) as logs,
        logger.filter_instrument((mag_x, mag_y), handler=logs.string_handler),
    ):
        driver.cartesian((0, 0, 1))

    any_x = False
    any_y = False
    for line in logs.value.splitlines():
        has_x = "[x(AMIModel430)]" in line
        has_y = "[y(AMIModel430)]" in line
        has_z = "[z(AMIModel430)]" in line

        assert has_x or has_y
        assert not has_z

        any_x |= has_x
        any_y |= has_y
    assert any_x
    assert any_y


def test_filter_without_started_logger_raises(
    AMI430_3D: tuple[AMIModel4303D, AMIModel430, AMIModel430, AMIModel430],
) -> None:
    driver, mag_x, _mag_y, _mag_z = AMI430_3D

    # filter one instrument
    driver.cartesian((0, 0, 0))
    with pytest.raises(RuntimeError), logger.filter_instrument(mag_x):
        pass


def test_capture_dataframe() -> None:
    root_logger = logging.getLogger()
    # the logger must be started to set level
    # debug on the rootlogger
    logger.start_logger()
    with capture_dataframe() as (_, cb):
        root_logger.debug(TEST_LOG_MESSAGE)
        df = cb()
    assert len(df) == 1
    assert df.message[0] == TEST_LOG_MESSAGE


def test_channels(model372: LakeshoreModel372Mock) -> None:
    """
    Test that messages logged in a channel are propagated to the
    main instrument.
    """
    inst = model372

    # set range to some other value so that it will actually be set in
    # the next call.
    inst.sample_heater.range_limits([0, 0.25, 0.5, 1, 2, 3, 4, 7])
    inst.sample_heater.set_range_from_temperature(1)
    with logger.LogCapture(level=logging.DEBUG) as logs_unfiltered:
        inst.sample_heater.set_range_from_temperature(0.1)

    # reset without capturing
    inst.sample_heater.set_range_from_temperature(1)
    # rerun with instrument filter
    with (
        logger.LogCapture(level=logging.DEBUG) as logs_filtered,
        logger.filter_instrument(inst, handler=logs_filtered.string_handler),
    ):
        inst.sample_heater.set_range_from_temperature(0.1)

    logs_filtered_2 = [
        log for log in logs_filtered.value.splitlines() if "[lakeshore" in log
    ]
    logs_unfiltered_2 = [
        log for log in logs_unfiltered.value.splitlines() if "[lakeshore" in log
    ]

    for f, u in zip(logs_filtered_2, logs_unfiltered_2):
        assert f == u


def test_channels_nomessages(model372: LakeshoreModel372Mock) -> None:
    """
    Test that messages logged in a channel are not propagated to
    any instrument.
    """
    inst = model372
    # test with wrong instrument
    mock = Instrument("mock")
    inst.sample_heater.set_range_from_temperature(1)
    with (
        logger.LogCapture(level=logging.DEBUG) as logs,
        logger.filter_instrument(mock, handler=logs.string_handler),
    ):
        inst.sample_heater.set_range_from_temperature(0.1)
    logs_2 = [log for log in logs.value.splitlines() if "[lakeshore" in log]
    assert len(logs_2) == 0
    mock.close()


@pytest.mark.usefixtures("awg5208")
def test_instrument_connect_message(caplog: LogCaptureFixture) -> None:
    """
    Test that the connect_message method logs as expected

    This test kind of belongs both here and in the tests for the instrument
    code, but it is more conveniently written here
    """

    setup_records = caplog.get_records("setup")

    idn = {"vendor": "QCoDeS", "model": "AWG5208", "serial": "1000", "firmware": "0.1"}
    expected_con_mssg = f"[awg_sim(TektronixAWG5208)] Connected to instrument: {idn}"

    assert any(rec.msg == expected_con_mssg for rec in setup_records)


def test_instrument_logger_extra_info(awg5208, caplog: LogCaptureFixture) -> None:
    """
    Test that the connect_message method logs as expected

    This test kind of belongs both here and in the tests for the instrument
    code, but it is more conveniently written here
    """
    extra_key = "some_extra"
    extra_val = "extra_value"

    with caplog.at_level(logging.INFO):
        awg5208.visa_log.info("test message")
    assert not hasattr(caplog.records[0], extra_key)
    caplog.clear()

    with caplog.at_level(logging.INFO):
        awg5208.visa_log.info("test message", extra={extra_key: extra_val})
    assert getattr(caplog.records[0], extra_key) == "extra_value"


def test_installation_info_logging() -> None:
    """
    Test that installation information is logged upon starting the logging
    """
    logger.start_logger()

    with open(logger.get_log_file_name()) as f:
        lines = f.readlines()

    assert "QCoDeS version:" in lines[-3]
    assert "QCoDeS installed in editable mode:" in lines[-2]
    assert "All installed package versions:" in lines[-1]


def test_get_level_name_with_invalid_type() -> None:
    """Only str and int can be converted to a logging level name."""
    with pytest.raises(RuntimeError, match="get_level_name: Cannot to convert level"):
        logger.get_level_name(1.5)  # type: ignore[arg-type]


def test_get_level_code_with_invalid_type() -> None:
    """Only str and int can be converted to a logging level code."""
    with pytest.raises(RuntimeError, match="get_level_code: Cannot to convert level"):
        logger.get_level_code(1.5)  # type: ignore[arg-type]


class ScopedDummyInstrument(DummyInstrument):
    """Dummy instrument that opts in to a per instrument logger."""

    default_logger_scope: ClassVar["LoggerScope"] = "instrument"


class ScopedDummyChannelInstrument(DummyChannelInstrument):
    """Instrument with channels that opts in to a per instrument logger."""

    default_logger_scope: ClassVar["LoggerScope"] = "instrument"


class ScopedAMIModel430(AMIModel430):
    """VISA instrument that opts in to a per instrument logger."""

    default_logger_scope: ClassVar["LoggerScope"] = "instrument"


class ScopedIPToVisa(IPToVisa):
    """IP-to-VISA instrument that opts in to a per instrument logger."""

    default_logger_scope: ClassVar["LoggerScope"] = "instrument"


@pytest.fixture(name="restore_shared_logger_levels", autouse=True)
def _restore_shared_logger_levels() -> "Generator[None, None, None]":
    """
    Restore levels of the shared and test-scoped loggers.

    Scoped loggers stay in the logging registry for the lifetime of the
    process, so a level set by one test would otherwise leak into the next.
    """
    roots = (
        SHARED_INSTRUMENT_LOGGER_NAME,
        SHARED_VISA_LOGGER_NAME,
        __name__,
        "test_logger_module_a",
        "test_logger_module_b",
    )

    def scoped_loggers() -> list[logging.Logger]:
        return [
            logging.getLogger(name)
            for name in (*roots, *logging.Logger.manager.loggerDict)
            if any(name == root or name.startswith(f"{root}.") for root in roots)
        ]

    levels = {logger_.name: logger_.level for logger_ in scoped_loggers()}
    yield
    for logger_ in scoped_loggers():
        logger_.setLevel(levels.get(logger_.name, logging.NOTSET))


def test_default_logger_scope_is_shared() -> None:
    """By default all instruments keep sharing a single logger."""
    inst_a = DummyInstrument("shared_scope_a")
    inst_b = DummyInstrument("shared_scope_b")

    assert inst_a.log.logger.name == SHARED_INSTRUMENT_LOGGER_NAME
    assert inst_a.log.logger is inst_b.log.logger


def test_instrument_scope_gives_one_logger_per_instrument() -> None:
    inst_a = ScopedDummyInstrument("instrument_scope_a")
    inst_b = ScopedDummyInstrument("instrument_scope_b")
    class_logger_name = scoped_class_logger_name(ScopedDummyInstrument)

    assert inst_a.log.logger.name == f"{class_logger_name}.instrument_scope_a"
    assert inst_b.log.logger.name == f"{class_logger_name}.instrument_scope_b"
    assert inst_a.log.logger is not inst_b.log.logger


def test_instrument_scope_isolates_logger_level() -> None:
    """Changing the level of one scoped logger must not affect any other."""
    class_logger = logging.getLogger(scoped_class_logger_name(ScopedDummyInstrument))
    class_logger.setLevel(logging.WARNING)

    inst_a = ScopedDummyInstrument("level_isolation_a")
    inst_b = ScopedDummyInstrument("level_isolation_b")
    inst_a.log.logger.setLevel(logging.DEBUG)

    assert inst_a.log.logger.level == logging.DEBUG
    assert inst_b.log.logger.level == logging.NOTSET
    assert inst_b.log.logger.getEffectiveLevel() == logging.WARNING


def test_scoped_logger_does_not_inherit_level_from_shared_logger() -> None:
    """Opted-in drivers leave the shared instrument logger hierarchy."""
    logging.getLogger(__name__).setLevel(logging.WARNING)
    logging.getLogger(SHARED_INSTRUMENT_LOGGER_NAME).setLevel(logging.ERROR)

    inst = ScopedDummyInstrument("level_inheritance")

    assert inst.log.logger.level == logging.NOTSET
    assert inst.log.logger.getEffectiveLevel() == logging.WARNING


def test_level_can_be_set_for_a_whole_driver_class() -> None:
    """
    A level configured on the driver class node applies to every instrument of
    that driver, including ones created afterwards, and not to other drivers.
    """
    logging.getLogger(__name__).setLevel(logging.WARNING)
    class_logger = logging.getLogger(scoped_class_logger_name(ScopedDummyInstrument))
    class_logger.setLevel(logging.DEBUG)

    # created only after the level was configured
    inst_a = ScopedDummyInstrument("class_level_a")
    inst_b = ScopedDummyInstrument("class_level_b")
    other = ScopedDummyChannelInstrument("class_level_other")

    assert inst_a.log.logger.getEffectiveLevel() == logging.DEBUG
    assert inst_b.log.logger.getEffectiveLevel() == logging.DEBUG
    assert other.log.logger.getEffectiveLevel() == logging.WARNING


def test_driver_class_level_applies_to_submodules() -> None:
    logging.getLogger(__name__).setLevel(logging.WARNING)
    class_logger = logging.getLogger(
        scoped_class_logger_name(ScopedDummyChannelInstrument)
    )
    class_logger.setLevel(logging.DEBUG)

    inst = ScopedDummyChannelInstrument("class_level_channels")
    channel = inst.submodules["A"]

    assert channel.log.logger.getEffectiveLevel() == logging.DEBUG  # type: ignore[union-attr]


def test_scope_is_fixed_when_root_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Changing the scope after the root was created must not split the hierarchy
    for submodules that are added later.
    """
    inst = ScopedDummyChannelInstrument("fixed_scope")
    monkeypatch.setattr(ScopedDummyChannelInstrument, "default_logger_scope", "shared")

    channel = DummyChannel(inst, "late_channel", "Z")
    inst.add_submodule("late_channel", channel)

    class_logger_name = scoped_class_logger_name(ScopedDummyChannelInstrument)
    assert channel.log.logger.name == f"{class_logger_name}.fixed_scope.late_channel"
    assert channel.log.logger.parent is inst.log.logger


def test_instrument_level_overrides_driver_class_level() -> None:
    """The per instrument node must stay more specific than the class node."""
    logging.getLogger(scoped_class_logger_name(ScopedDummyInstrument)).setLevel(
        logging.ERROR
    )

    inst_a = ScopedDummyInstrument("override_a")
    inst_b = ScopedDummyInstrument("override_b")
    inst_a.log.logger.setLevel(logging.DEBUG)

    assert inst_a.log.logger.getEffectiveLevel() == logging.DEBUG
    assert inst_b.log.logger.getEffectiveLevel() == logging.ERROR


def test_scoped_logger_keeps_instrument_extra_info(caplog: LogCaptureFixture) -> None:
    """Records must keep the info that ``filter_instrument`` relies on."""
    inst = ScopedDummyInstrument("extra_info")
    inst.log.logger.setLevel(logging.INFO)

    with caplog.at_level(logging.INFO):
        inst.log.info(TEST_LOG_MESSAGE)

    record = caplog.records[-1]
    assert record.name == inst.log.logger.name
    assert record.__dict__["instrument_name"] == "extra_info"
    assert record.__dict__["instrument_type"] == "ScopedDummyInstrument"


def test_scoped_logger_is_filterable_by_instrument() -> None:
    inst = ScopedDummyInstrument("filterable")
    other = ScopedDummyInstrument("not_filterable")
    inst.log.logger.setLevel(logging.INFO)
    other.log.logger.setLevel(logging.INFO)

    with (
        logger.LogCapture(level=logging.DEBUG) as logs,
        logger.filter_instrument(inst, handler=logs.string_handler),
    ):
        inst.log.info(TEST_LOG_MESSAGE)
        other.log.info(TEST_LOG_MESSAGE)

    assert "[filterable(ScopedDummyInstrument)]" in logs.value
    assert "[not_filterable(ScopedDummyInstrument)]" not in logs.value


def test_submodule_logger_is_child_of_instrument_logger() -> None:
    """A submodule is named after its own name and sits below its parent."""
    inst = ScopedDummyChannelInstrument("scoped_channels")
    channel_a = inst.submodules["A"]
    channel_b = inst.submodules["B"]
    class_logger_name = scoped_class_logger_name(ScopedDummyChannelInstrument)

    assert inst.log.logger.name == f"{class_logger_name}.scoped_channels"
    assert (
        channel_a.log.logger.name  # type: ignore[union-attr]
        == f"{class_logger_name}.scoped_channels.ChanA"
    )
    assert (
        channel_b.log.logger.name  # type: ignore[union-attr]
        == f"{class_logger_name}.scoped_channels.ChanB"
    )
    assert channel_a.log.logger.parent is inst.log.logger  # type: ignore[union-attr]
    assert channel_b.log.logger.parent is inst.log.logger  # type: ignore[union-attr]


def test_nested_submodule_logger_is_child_of_its_parent_module() -> None:
    inst = ScopedDummyChannelInstrument("nested_modules")
    channel = inst.submodules["A"]
    nested = InstrumentModule(channel, "nested")  # type: ignore[arg-type]
    channel.add_submodule("nested", nested)  # type: ignore[union-attr]

    assert nested.name_parts == ["nested_modules", "ChanA", "nested"]
    assert nested.log.logger.name == (
        f"{scoped_class_logger_name(ScopedDummyChannelInstrument)}"
        ".nested_modules.ChanA.nested"
    )
    assert nested.log.logger.parent is channel.log.logger  # type: ignore[union-attr]


def test_instrument_level_applies_to_its_submodules() -> None:
    """Setting the level on an instrument must also cover its channels."""
    logging.getLogger(__name__).setLevel(logging.WARNING)
    inst = ScopedDummyChannelInstrument("level_to_channels")
    other = ScopedDummyChannelInstrument("level_to_channels_other")
    channel = inst.submodules["A"]

    inst.log.logger.setLevel(logging.DEBUG)

    assert channel.log.logger.getEffectiveLevel() == logging.DEBUG  # type: ignore[union-attr]
    assert other.submodules["A"].log.logger.getEffectiveLevel() == logging.WARNING  # type: ignore[union-attr]


def test_submodule_level_can_be_set_for_one_module() -> None:
    """A level can be set on exactly one submodule of one instrument."""
    logging.getLogger(__name__).setLevel(logging.WARNING)
    inst_a = ScopedDummyChannelInstrument("submodule_level_a")
    inst_b = ScopedDummyChannelInstrument("submodule_level_b")

    inst_a.submodules["A"].log.logger.setLevel(logging.DEBUG)  # type: ignore[union-attr]

    assert inst_a.submodules["A"].log.logger.getEffectiveLevel() == logging.DEBUG  # type: ignore[union-attr]
    assert inst_a.submodules["B"].log.logger.getEffectiveLevel() == logging.WARNING  # type: ignore[union-attr]
    assert inst_b.submodules["A"].log.logger.getEffectiveLevel() == logging.WARNING  # type: ignore[union-attr]


def test_submodule_of_unscoped_instrument_uses_shared_logger() -> None:
    inst = DummyChannelInstrument("unscoped_channels")
    channel = inst.submodules["A"]

    assert channel.log.logger.name == SHARED_INSTRUMENT_LOGGER_NAME  # type: ignore[union-attr]


def test_visa_log_is_shared_by_default() -> None:
    inst = AMIModel430(
        "shared_visa_log",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )

    assert inst.visa_log.logger.name == SHARED_VISA_LOGGER_NAME
    assert inst.log.logger.name == SHARED_INSTRUMENT_LOGGER_NAME


def test_visa_log_follows_instrument_scope() -> None:
    class_logger_name = scoped_class_logger_name(ScopedAMIModel430)
    inst = ScopedAMIModel430(
        "scoped_visa_log",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )

    assert inst.visa_log.logger.name == f"{class_logger_name}.scoped_visa_log.com.visa"
    assert inst.log.logger.name == f"{class_logger_name}.scoped_visa_log"
    assert inst.visa_log.logger is not inst.log.logger


def test_driver_class_logger_is_ancestor_of_log_and_visa_loggers() -> None:
    class_logger = logging.getLogger(scoped_class_logger_name(ScopedAMIModel430))
    inst = ScopedAMIModel430(
        "scoped_logger_parents",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )

    assert inst.log.logger.parent is class_logger
    assert logger_is_descendant(inst.visa_log.logger, inst.log.logger)
    assert logger_is_descendant(inst.visa_log.logger, class_logger)
    assert logger_is_descendant(inst.switch_heater.log.logger, inst.log.logger)


def test_instrument_level_applies_to_visa_traffic_and_submodules() -> None:
    """One level on an instrument covers its driver, VISA and module records."""
    logging.getLogger(__name__).setLevel(logging.WARNING)
    inst = ScopedAMIModel430(
        "level_for_whole_instrument",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )
    other = ScopedAMIModel430(
        "level_for_other_instrument",
        address="GPIB::2::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )

    inst.log.logger.setLevel(logging.DEBUG)

    assert inst.visa_log.logger.getEffectiveLevel() == logging.DEBUG
    assert inst.switch_heater.log.logger.getEffectiveLevel() == logging.DEBUG
    assert other.log.logger.getEffectiveLevel() == logging.WARNING
    assert other.visa_log.logger.getEffectiveLevel() == logging.WARNING


def test_visa_level_can_differ_from_instrument_level() -> None:
    """The VISA traffic of an instrument can be configured separately."""
    inst = ScopedAMIModel430(
        "separate_visa_level",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )

    inst.log.logger.setLevel(logging.DEBUG)
    inst.visa_log.logger.setLevel(logging.WARNING)

    assert inst.log.logger.getEffectiveLevel() == logging.DEBUG
    assert inst.switch_heater.log.logger.getEffectiveLevel() == logging.DEBUG
    assert inst.visa_log.logger.getEffectiveLevel() == logging.WARNING


def test_driver_class_level_applies_to_visa_traffic() -> None:
    logging.getLogger(__name__).setLevel(logging.WARNING)
    logging.getLogger(scoped_class_logger_name(ScopedAMIModel430)).setLevel(
        logging.DEBUG
    )

    inst = ScopedAMIModel430(
        "class_level_for_visa",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )

    assert inst.visa_log.logger.getEffectiveLevel() == logging.DEBUG


def test_handler_on_instrument_logger_receives_all_its_records() -> None:
    """A handler on the instrument logger sees driver, module and VISA records."""
    inst = ScopedAMIModel430(
        "handler_for_whole_instrument",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )
    other = ScopedAMIModel430(
        "handler_for_other_instrument",
        address="GPIB::2::INSTR",
        pyvisa_sim_file="AMI430.yaml",
        terminator="\n",
    )
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.DEBUG)
    inst.log.logger.addHandler(handler)
    inst.log.logger.setLevel(logging.DEBUG)
    try:
        inst.log.info(TEST_LOG_MESSAGE)
        inst.switch_heater.log.info(TEST_LOG_MESSAGE)
        inst.visa_log.info(TEST_LOG_MESSAGE)
        other.log.info(TEST_LOG_MESSAGE)
        other.visa_log.info(TEST_LOG_MESSAGE)
    finally:
        inst.log.logger.removeHandler(handler)

    class_logger_name = scoped_class_logger_name(ScopedAMIModel430)
    assert [record.name for record in records] == [
        f"{class_logger_name}.handler_for_whole_instrument",
        f"{class_logger_name}.handler_for_whole_instrument.SwitchHeater",
        f"{class_logger_name}.handler_for_whole_instrument.com.visa",
    ]


def test_same_class_name_in_different_modules_has_distinct_loggers() -> None:
    driver_a: type[DummyInstrument] = type(
        "DuplicateDriver",
        (DummyInstrument,),
        {
            "__module__": "test_logger_module_a",
            "default_logger_scope": "instrument",
        },
    )
    driver_b: type[DummyInstrument] = type(
        "DuplicateDriver",
        (DummyInstrument,),
        {
            "__module__": "test_logger_module_b",
            "default_logger_scope": "instrument",
        },
    )

    inst_a = driver_a("duplicate_a")
    inst_b = driver_b("duplicate_b")

    assert inst_a.log.logger.name == "test_logger_module_a.DuplicateDriver.duplicate_a"
    assert inst_b.log.logger.name == "test_logger_module_b.DuplicateDriver.duplicate_b"
    assert inst_a.log.logger is not inst_b.log.logger


def test_ip_to_visa_log_follows_instrument_scope() -> None:
    class_logger_name = scoped_class_logger_name(ScopedIPToVisa)
    inst = ScopedIPToVisa(
        "scoped_ip_to_visa",
        address="GPIB::1::INSTR",
        port=None,
        pyvisa_sim_file="AMI430.yaml",
    )

    assert inst.log.logger.name == f"{class_logger_name}.scoped_ip_to_visa"
    assert (
        inst.visa_log.logger.name == f"{class_logger_name}.scoped_ip_to_visa.com.visa"
    )
