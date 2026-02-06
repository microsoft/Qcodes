from collections import Counter
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest

from qcodes.instrument_drivers.Keithley import (
    Keithley2600Channel,
    Keithley2600MeasurementStatus,
    Keithley2614B,
)


@pytest.fixture(scope="function", name="driver")
def _make_driver():
    driver = Keithley2614B(
        "Keithley_2600", address="GPIB::1::INSTR", pyvisa_sim_file="Keithley_2600.yaml"
    )

    yield driver
    driver.close()


@pytest.fixture(scope="function", name="smus")
def _make_smus(driver):
    smu_names = {"smua", "smub"}
    assert smu_names == set(list(driver.submodules.keys()))

    yield tuple(getattr(driver, smu_name) for smu_name in smu_names)


def test_idn(driver) -> None:
    assert {
        "firmware": "3.0.0",
        "model": "2601B",
        "serial": "1398687",
        "vendor": "Keithley Instruments Inc.",
    } == driver.IDN()


def test_smu_channels_and_their_parameters(driver) -> None:
    assert {"smua", "smub"} == set(list(driver.submodules.keys()))

    for smu_name in ("smua", "smub"):
        smu = getattr(driver, smu_name)

        smu.volt(1.0)
        assert smu.volt.measurement_status is None

        assert 1.0 == smu.volt()
        assert smu.volt.measurement_status == Keithley2600MeasurementStatus.NORMAL

        smu.curr(1.0)
        assert smu.volt.measurement_status is None

        assert 1.0 == smu.curr()
        assert smu.curr.measurement_status == Keithley2600MeasurementStatus.NORMAL

        assert 0.0 == smu.res()

        assert "current" == smu.mode()
        smu.mode("voltage")
        assert smu.mode() == "voltage"

        assert smu.output() is False
        smu.output(True)
        assert smu.output() is True

        assert 0.0 == smu.nplc()
        smu.nplc(2.3)
        assert smu.nplc() == 2.3

        assert 0.0 == smu.sourcerange_v()
        some_valid_sourcerange_v = driver._vranges[smu.model][2]
        smu.sourcerange_v(some_valid_sourcerange_v)
        assert smu.sourcerange_v() == some_valid_sourcerange_v

        assert smu.source_autorange_v_enabled() is False
        smu.source_autorange_v_enabled(True)
        assert smu.source_autorange_v_enabled() is True

        assert 0.0 == smu.measurerange_v()
        some_valid_measurerange_v = driver._vranges[smu.model][2]
        smu.measurerange_v(some_valid_measurerange_v)
        assert smu.measurerange_v() == some_valid_measurerange_v

        assert smu.measure_autorange_v_enabled() is False
        smu.measure_autorange_v_enabled(True)
        assert smu.measure_autorange_v_enabled() is True

        assert 0.0 == smu.sourcerange_i()
        some_valid_sourcerange_i = driver._iranges[smu.model][2]
        smu.sourcerange_i(some_valid_sourcerange_i)
        assert smu.sourcerange_i() == some_valid_sourcerange_i

        assert smu.source_autorange_i_enabled() is False
        smu.source_autorange_i_enabled(True)
        assert smu.source_autorange_i_enabled() is True

        assert 0.0 == smu.measurerange_i()
        some_valid_measurerange_i = driver._iranges[smu.model][2]
        smu.measurerange_i(some_valid_measurerange_i)
        assert smu.measurerange_i() == some_valid_measurerange_i

        assert smu.measure_autorange_i_enabled() is False
        smu.measure_autorange_i_enabled(True)
        assert smu.measure_autorange_i_enabled() is True

        assert 0.0 == smu.limitv()
        smu.limitv(2.3)
        assert smu.limitv() == 2.3

        assert 0.0 == smu.limiti()
        smu.limiti(2.3)
        assert smu.limiti() == 2.3

        assert "current" == smu.timetrace_mode()
        smu.timetrace_mode("voltage")
        assert smu.timetrace_mode() == "voltage"

        assert 500 == smu.timetrace_npts()
        smu.timetrace_npts(600)
        assert smu.timetrace_npts() == 600

        assert 0.001 == smu.timetrace_dt()
        smu.timetrace_dt(0.002)
        assert smu.timetrace_dt() == 0.002

        dt = smu.timetrace_dt()
        npts = smu.timetrace_npts()
        expected_time_axis = np.linspace(0, dt * npts, npts, endpoint=False)
        assert len(expected_time_axis) == len(smu.time_axis())
        assert Counter(expected_time_axis) == Counter(smu.time_axis())
        assert set(expected_time_axis) == set(smu.time_axis())

        smu.timetrace_mode("current")
        assert "A" == smu.timetrace.unit
        assert "Current" == smu.timetrace.label
        assert smu.time_axis == smu.timetrace.setpoints[0]

        smu.timetrace_mode("voltage")
        assert "V" == smu.timetrace.unit
        assert "Voltage" == smu.timetrace.label
        assert smu.time_axis == smu.timetrace.setpoints[0]


def test_setting_source_voltage_range_disables_autorange(smus) -> None:
    for smu in smus:
        smu.source_autorange_v_enabled(True)
        assert smu.source_autorange_v_enabled() is True
        some_valid_sourcerange_v = smu.root_instrument._vranges[smu.model][2]
        smu.sourcerange_v(some_valid_sourcerange_v)
        assert smu.source_autorange_v_enabled() is False


def test_setting_measure_voltage_range_disables_autorange(smus) -> None:
    for smu in smus:
        smu.measure_autorange_v_enabled(True)
        assert smu.measure_autorange_v_enabled() is True
        some_valid_measurerange_v = smu.root_instrument._vranges[smu.model][2]
        smu.measurerange_v(some_valid_measurerange_v)
        assert smu.measure_autorange_v_enabled() is False


def test_setting_source_current_range_disables_autorange(smus) -> None:
    for smu in smus:
        smu.source_autorange_i_enabled(True)
        assert smu.source_autorange_i_enabled() is True
        some_valid_sourcerange_i = smu.root_instrument._iranges[smu.model][2]
        smu.sourcerange_i(some_valid_sourcerange_i)
        assert smu.source_autorange_i_enabled() is False


def test_setting_measure_current_range_disables_autorange(smus) -> None:
    for smu in smus:
        smu.measure_autorange_i_enabled(True)
        assert smu.measure_autorange_i_enabled() is True
        some_valid_measurerange_i = smu.root_instrument._iranges[smu.model][2]
        smu.measurerange_i(some_valid_measurerange_i)
        assert smu.measure_autorange_i_enabled() is False


def test_fast_sweep_parameters(smus) -> None:
    for smu in smus:
        smu = cast("Keithley2600Channel", smu)

        start_val = 0.0001
        stop_val = 0.001
        npts = 50

        # Test IV mode
        smu.fastsweep_mode("IV")
        assert smu.fastsweep.unit == "A"
        assert smu.fastsweep.label == "current"

        # Test VI mode
        smu.fastsweep_mode("VI")
        assert smu.fastsweep.unit == "V"
        assert smu.fastsweep.label == "voltage"

        # Test VIfourprobe mode
        smu.fastsweep_mode("VIfourprobe")
        assert smu.fastsweep.unit == "V"
        assert smu.fastsweep.label == "voltage"

        smu.fastsweep_start(start_val)
        assert smu.fastsweep_start() == start_val

        smu.fastsweep_stop(stop_val)
        assert smu.fastsweep_stop() == stop_val

        smu.fastsweep_npts(npts)
        assert smu.fastsweep_npts() == npts

        # Test the setpoints (fastsweep_setpoints)
        setpoints = smu.fastsweep_setpoints()
        expected_setpoints = np.linspace(start_val, stop_val, npts)
        np.testing.assert_array_almost_equal(setpoints, expected_setpoints)


def test_fastsweep(driver) -> None:
    smu = driver.smua

    # Configure the fastsweep
    smu.fastsweep_mode("IV")
    smu.fastsweep_start(0.0)
    smu.fastsweep_stop(1.0)
    smu.fastsweep_npts(10)

    # Mock _execute_lua to return fake measurement data
    fake_data = np.linspace(0, 0.001, 10)  # Fake current readings

    with patch.object(smu, "_execute_lua", return_value=fake_data):
        result = smu.fastsweep()

    assert len(result) == 10
    np.testing.assert_array_equal(result, fake_data)
