from typing import TYPE_CHECKING

import pytest

from qcodes.extensions import DriverTestCase
from qcodes.instrument_drivers.agilent import AgilentE8257D

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytest_mock import MockerFixture


class TestAgilentE8257D(DriverTestCase[AgilentE8257D]):
    """
    This is a test suite for testing the QuTech_ControlBox Instrument.
    It is designed to provide a test function for each function as well as for
    general things such as testing if the com s are working.
    """

    driver = AgilentE8257D

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.instrument.off()  # Not a test but a safety measure

    def test_firmware_version(self) -> None:
        v = self.instrument.IDN.get()
        self.assertEqual(v["vendor"], "Agilent Technologies")
        self.assertEqual(v["model"], "E8257D")

    def test_on_off(self) -> None:
        self.instrument.off()
        self.assertEqual(self.instrument.status.get(), "Off")

        self.instrument.on()
        self.assertEqual(self.instrument.status.get(), "On")

        self.instrument.status.set("off")
        self.assertEqual(self.instrument.status.get(), "Off")

        self.instrument.status.set("On")
        self.assertEqual(self.instrument.status.get(), "On")

        # Ensure test ends with instrument being off
        self.instrument.off()
        self.assertEqual(self.instrument.status.get(), "Off")

        # This should raise an error because it is not a valid input
        with self.assertRaises(ValueError):
            self.instrument.status.set("on24")

    def test_frequency(self) -> None:
        with self.assertRaises(ValueError):
            self.instrument.frequency.set(32e9)
        with self.assertRaises(ValueError):
            self.instrument.frequency.set(32)

        cur_f = self.instrument.frequency.get()
        test_f = 2e9
        self.instrument.frequency.set(test_f)
        self.assertEqual(self.instrument.frequency.get(), test_f)

        test_f = 2.2435e9
        self.instrument.frequency.set(test_f)
        self.assertEqual(self.instrument.frequency.get(), test_f)

        # leave the setup in the initial state
        self.instrument.frequency.set(cur_f)

    def test_power(self) -> None:
        with self.assertRaises(ValueError):
            self.instrument.power.set(-150)
        with self.assertRaises(ValueError):
            self.instrument.power.set(32)

        cur_val = self.instrument.power.get()
        test_val = -18
        self.instrument.power.set(test_val)
        self.assertEqual(self.instrument.power.get(), test_val)

        test_val = -5
        self.instrument.power.set(test_val)
        self.assertEqual(self.instrument.power.get(), test_val)

        # leave the setup in the initial state
        self.instrument.power.set(cur_val)

    def test_phase(self) -> None:
        with self.assertRaises(ValueError):
            self.instrument.phase.set(-250)
        with self.assertRaises(ValueError):
            self.instrument.phase.set(181)

        cur_val = self.instrument.phase.get()
        test_val = 12
        self.instrument.phase.set(test_val)
        self.assertAlmostEqual(self.instrument.phase.get(), test_val, places=4)

        test_val = -80
        self.instrument.phase.set(test_val)
        self.assertAlmostEqual(self.instrument.phase.get(), test_val, places=4)

        # leave the setup in the initial state
        self.instrument.phase.set(cur_val)


@pytest.fixture(name="e8257d")
def _make_e8257d() -> "Generator[AgilentE8257D, None, None]":
    # the simulated instrument is closed after each test, so it is never
    # picked up by the hardware tests in TestAgilentE8257D above
    driver = AgilentE8257D(
        "e8257d", address="GPIB::1::INSTR", pyvisa_sim_file="Agilent_E8257D.yaml"
    )
    try:
        yield driver
    finally:
        driver.close()


def test_on_off(e8257d: AgilentE8257D, mocker: "MockerFixture") -> None:
    write_raw = mocker.spy(e8257d, "write_raw")

    e8257d.on()
    assert e8257d.output_enabled() is True

    e8257d.off()
    assert e8257d.output_enabled() is False

    assert write_raw.call_args_list == [
        mocker.call("OUTP 1"),
        mocker.call("OUTP 0"),
    ]


def test_limits_from_options(e8257d: AgilentE8257D) -> None:
    # frequency option 520 with the 1E1 step attenuator
    e8257d.frequency.validate(20e9)
    with pytest.raises(ValueError):
        e8257d.frequency.validate(21e9)

    e8257d.power.validate(-135)
    with pytest.raises(ValueError):
        e8257d.power.validate(11)
