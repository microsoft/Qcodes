import logging

import numpy as np
import pytest
from pytest import LogCaptureFixture

from qcodes.instrument_drivers.Keithley import Keithley2450


@pytest.fixture(scope="function")
def k2450():
    """
    Create a Keithley 2450 instrument
    """
    driver = Keithley2450(
        "k2450", address="GPIB::2::INSTR", pyvisa_sim_file="Keithley_2450.yaml"
    )
    yield driver
    driver.close()


def test_wrong_mode(caplog: LogCaptureFixture) -> None:
    """
    Starting an instrument in the wrong mode should result in a warning. Additionally, no
    parameters should be available, other then the parameters "IDN" and "timeout" which
    are created by the Instrument and VisaInstrument parent classes
    """
    with caplog.at_level(logging.WARNING):
        instrument = Keithley2450(
            "wrong_mode", address="GPIB::1::INSTR", pyvisa_sim_file="Keithley_2450.yaml"
        )
        assert "The instrument is in an unsupported language mode." in caplog.text
        assert list(instrument.parameters.keys()) == ["IDN", "timeout"]
        instrument.close()


def test_change_source_function(k2450) -> None:
    """
    The parameters available on the source sub-module depend on the function.
    """
    k2450.source.function("current")
    assert k2450.source is k2450.submodules["_source_current"]
    assert hasattr(k2450.source, "current")
    assert not hasattr(k2450.source, "voltage")

    k2450.source.function("voltage")
    assert k2450.source is k2450.submodules["_source_voltage"]
    assert not hasattr(k2450.source, "current")
    assert hasattr(k2450.source, "voltage")

    # to cover all bases :-)
    assert (
        k2450.submodules["_source_current"] is not k2450.submodules["_source_voltage"]
    )


def test_source_change_error(k2450) -> None:
    """
    If the sense function is in resistance mode, test that an error is generated
    when we change the source function
    """
    k2450.source.function("voltage")
    k2450.sense.function("resistance")
    with pytest.raises(
        RuntimeError,
        match="Cannot change the source function while sense function is in 'resistance' mode",
    ):
        k2450.source.function("current")

    k2450.sense.function("voltage")
    # after changing the sense function we should be able to change the source function
    k2450.source.function("current")


def test_sense_current_mode(k2450) -> None:
    """
    Test that when we are in a sense function, for example, 'current', that
    the sense property is returning to the correct submodule. We also test
    that if we are in a sense function (e.g. 'current') the returned submodule
    does not have inappropriate parameters, (e.g. 'voltage', or 'current')
    """
    sense_functions = {"current", "voltage", "resistance"}
    for sense_function in sense_functions:
        k2450.sense.function(sense_function)
        assert k2450.sense is k2450.submodules[f"_sense_{sense_function}"]
        assert hasattr(k2450.sense, sense_function)

        for other_sense_function in sense_functions.difference({sense_function}):
            assert not hasattr(k2450.sense, other_sense_function)


def test_setpoint_always_follows_source_function(k2450) -> None:
    """
    Changing the source and/or sense functions should not confuse the setpoints. These
    should always follow the source module
    """
    n = 100
    sense_modes = np.random.choice(["current", "voltage", "resistance"], n)
    source_modes = np.random.choice(["current", "voltage"], n)

    for sense_mode, source_mode in zip(sense_modes, source_modes):
        k2450.sense.function("voltage")  # In 'resistance' sense mode, we cannot
        # change the source mode by design. Therefore temporarily switch to
        # 'voltage'
        k2450.source.function(source_mode)
        k2450.sense.function(sense_mode)
        assert k2450.sense.sweep.setpoints == (k2450.source.sweep_axis,)


def test_reset_sweep_on_source_change(k2450) -> None:
    """
    If we change the source function, we need to run the sweep setup again
    """
    # first set sense to a mode where we are allowed to change source
    k2450.sense.function("current")
    k2450.source.function("voltage")
    k2450.source.sweep_setup(0, 1, 10)
    assert np.all(k2450.source.get_sweep_axis() == np.linspace(0, 1, 10))

    k2450.source.function("current")
    with pytest.raises(ValueError):
        k2450.source.get_sweep_axis()


def test_sweep(k2450) -> None:
    """
    Verify that we can start sweeps
    """
    k2450.sense.function("current")
    k2450.source.function("voltage")
    k2450.source.sweep_setup(0, 1, 10)
    k2450.sense.sweep.get()
