from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from qcodes.instrument_drivers.rohde_schwarz.SGS100A import RohdeSchwarz_SGS100A

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture(scope="function", name="sg")
def _make_sg():
    """
    Create a RohdeSchwarz SGS100A instrument
    """
    driver = RohdeSchwarz_SGS100A(
        "sgs100a", address="GPIB::1::INSTR", pyvisa_sim_file="RSSGS100A.yaml"
    )
    yield driver
    driver.close()


def verify_property(sg, param_name, vals: "Sequence[Any]"):
    param = getattr(sg, param_name)
    for val in vals:
        param(val)
        new_val = param()
        if isinstance(new_val, float):
            assert np.isclose(new_val, val)
        else:
            assert new_val == val


def test_frequency(sg) -> None:
    verify_property(sg, "frequency", [1e6, 2e6, 3e9, 20e9])


def test_phase(sg) -> None:
    verify_property(sg, "phase", [0, 45, 90, 180, 270, 359, 360])


def test_power(sg) -> None:
    verify_property(sg, "power", [-120, -50, 0, 10, 25])


def test_status(sg) -> None:
    verify_property(sg, "status", [True, False])


def test_IQ_state(sg) -> None:
    verify_property(sg, "IQ_state", [True, False])


def test_pulsemod_state(sg) -> None:
    verify_property(sg, "pulsemod_state", [True, False])


def test_pulsemod_source(sg) -> None:
    verify_property(sg, "pulsemod_source", ["INT", "EXT"])


def test_ref_osc_source(sg) -> None:
    verify_property(sg, "ref_osc_source", ["INT", "EXT"])


def test_LO_source(sg) -> None:
    verify_property(sg, "LO_source", ["INT", "EXT"])


def test_ref_LO_out(sg) -> None:
    verify_property(sg, "ref_LO_out", ["REF", "LO", "OFF"])


def test_ref_osc_output_freq(sg) -> None:
    verify_property(sg, "ref_osc_output_freq", ["10MHz", "100MHz", "1000MHz"])


def test_ref_osc_external_freq(sg) -> None:
    verify_property(sg, "ref_osc_external_freq", ["10MHz", "100MHz", "1000MHz"])


def test_IQ_impairments(sg) -> None:
    verify_property(sg, "IQ_impairments", [True, False])


def test_I_offset(sg) -> None:
    verify_property(sg, "I_offset", [-10, 0, 10])


def test_Q_offset(sg) -> None:
    verify_property(sg, "Q_offset", [-10, 0, 10])


def test_IQ_gain_imbalance(sg) -> None:
    verify_property(sg, "IQ_gain_imbalance", [-1, 0, 1])


def test_IQ_angle(sg) -> None:
    verify_property(sg, "IQ_angle", [-8, -4, 0, 4, 8])
