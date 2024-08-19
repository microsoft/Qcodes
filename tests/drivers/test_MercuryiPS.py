import logging

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from pytest import LogCaptureFixture

from qcodes.instrument_drivers.oxford.MercuryiPS_VISA import MercuryiPS
from qcodes.math_utils.field_vector import FieldVector


@pytest.fixture(scope="function")
def driver():
    mips = MercuryiPS(
        "mips", address="GPIB::1::INSTR", pyvisa_sim_file="MercuryiPS.yaml"
    )
    yield mips
    mips.close()


@pytest.fixture(scope="function")
def driver_spher_lim():
    def spherical_limits(x, y, z):
        """
        Checks that the field is inside a sphere of radius 2
        """
        return np.sqrt(x**2 + y**2 + z**2) <= 2

    mips_sl = MercuryiPS(
        "mips_sl",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="MercuryiPS.yaml",
        field_limits=spherical_limits,
    )

    yield mips_sl
    mips_sl.close()


@pytest.fixture(scope="function")
def driver_cyl_lim():
    def cylindrical_limits(x, y, z):
        """
        Checks that the field is inside a particular cylinder
        """
        rho_check = np.sqrt(x**2 + y**2) <= 2
        z_check = z < 3 and z > -1

        return rho_check and z_check

    mips_cl = MercuryiPS(
        "mips_cl",
        address="GPIB::1::INSTR",
        pyvisa_sim_file="MercuryiPS.yaml",
        field_limits=cylindrical_limits,
    )

    yield mips_cl
    mips_cl.close()


def test_idn(driver) -> None:
    assert driver.IDN()["model"] == "SIMULATED MERCURY iPS"


def test_simple_setting(driver) -> None:
    """
    Some very simple setting of parameters. Mainly just to
    sanity-check the pyvisa-sim setup.
    """
    assert driver.GRPX.field_target() == 0
    driver.GRPX.field_target(0.1)
    assert driver.GRPX.field_target() == 0.1


def test_vector_setting(driver) -> None:
    assert driver.field_target().distance(FieldVector(0, 0, 0)) <= 1e-8
    driver.field_target(FieldVector(r=0.1, theta=0, phi=0))
    assert driver.field_target().distance(FieldVector(r=0.1, theta=0, phi=0)) <= 1e-8


def test_vector_ramp_rate(driver) -> None:
    driver.field_ramp_rate(FieldVector(0.1, 0.1, 0.1))
    assert driver.field_ramp_rate().distance(FieldVector(0.1, 0.1, 0.1)) <= 1e-8


def test_wrong_field_limit_raises() -> None:
    # check that a non-callable input fails
    with pytest.raises(ValueError):
        MercuryiPS(
            "mips",
            address="GPIB::1::INSTR",
            pyvisa_sim_file="MercuryiPS.yaml",
            field_limits=0,  # type: ignore[arg-type]
        )


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(
    x=hst.floats(min_value=-3, max_value=3),
    y=hst.floats(min_value=-3, max_value=3),
    z=hst.floats(min_value=-3, max_value=3),
)
def test_field_limits(x, y, z, driver_spher_lim, driver_cyl_lim) -> None:
    """
    Try with a few different field_limits functions and see if we get no-go
    exceptions when we expect them
    """

    # TODO: there really isn't a reason to do this tuple-by-tuple
    #  and not point-by-point. Extend the test to do that.

    for mip in [driver_spher_lim, driver_cyl_lim]:
        # reset (PyVISA-sim unfortunately has memory)
        mip.x_target(0)
        mip.y_target(0)
        mip.z_target(0)

        if mip._field_limits(x, y, z):
            mip.x_target(x)
            mip.y_target(y)
            mip.z_target(z)

        else:
            with pytest.raises(ValueError):
                mip.x_target(x)
                mip.y_target(y)
                mip.z_target(z)


def get_ramp_order(caplog_records):
    """
    Helper function used in test_ramp_safely
    """
    order = []
    for record in caplog_records:
        mssg = record.message
        if "RTOS" in mssg:
            axis = mssg[mssg.find("GRP") + 3 : mssg.find("GRP") + 4]
            order.append(axis.lower())
    return order


@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(
    x=hst.floats(min_value=-3, max_value=3),
    y=hst.floats(min_value=-3, max_value=3),
    z=hst.floats(min_value=-3, max_value=3),
)
def test_ramp_safely(driver, x, y, z, caplog: LogCaptureFixture) -> None:
    """
    Test that we get the first-down-then-up order right
    """
    # reset the instrument to default
    driver.GRPX.ramp_status("HOLD")
    driver.GRPY.ramp_status("HOLD")
    driver.GRPZ.ramp_status("HOLD")

    # the current field values are always zero for the sim. instr.
    # Use the FieldVector interface here to increase coverage.
    driver.field_target(FieldVector(x=x, y=y, z=z))

    exp_order = np.array(["x", "y", "z"])[np.argsort(np.abs(np.array([x, y, z])))]

    with caplog.at_level(logging.DEBUG, logger="qcodes.instrument.visa"):
        caplog.clear()
        driver._ramp_safely()
        ramp_order = get_ramp_order(caplog.records)

    assert ramp_order == list(exp_order)
