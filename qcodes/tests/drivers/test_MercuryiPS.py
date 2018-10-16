import logging

import pytest
import numpy as np
import hypothesis as hst

from qcodes.instrument_drivers.oxford.MercuryiPS_VISA import MercuryiPS
import qcodes.instrument.sims as sims


visalib = sims.__file__.replace('__init__.py', 'MercuryiPS.yaml@sim')


@pytest.fixture(scope='function')
def driver():

    mips = MercuryiPS('mips', address='GPIB::1::INSTR',
                      visalib=visalib)
    yield mips
    mips.close()


@pytest.fixture(scope='function')
def driver_spher_lim():

    def spherical_limits(x, y, z):
        """
        Checks that the field is inside a sphere of radius 2
        """
        return np.sqrt(x**2 + y**2 + z**2) <= 2

    mips_sl = MercuryiPS('mips_sl', address='GPIB::1::INSTR',
                         visalib=visalib, field_limits=spherical_limits)

    yield mips_sl
    mips_sl.close()


@pytest.fixture(scope='function')
def driver_cyl_lim():
    def cylindrical_limits(x, y, z):
        """
        Checks that the field is inside a particular cylinder
        """
        rho_check = np.sqrt(x**2 + y**2) <= 2
        z_check = z < 3 and z > -1

        return rho_check and z_check

    mips_cl = MercuryiPS('mips_cl', address='GPIB::1::INSTR',
                         visalib=visalib, field_limits=cylindrical_limits)

    yield mips_cl
    mips_cl.close()


def test_idn(driver):
    assert driver.IDN()['model'] == 'SIMULATED MERCURY iPS'


def test_simple_setting(driver):
    """
    Some very simple setting of parameters. Mainly just to
    sanity-check the pyvisa-sim setup.
    """
    assert driver.GRPX.field_target() == 0
    driver.GRPX.field_target(0.1)
    assert driver.GRPX.field_target() == 0.1


def test_wrong_field_limit_raises():

    # check that a non-callable input fails
    with pytest.raises(ValueError):
        MercuryiPS('mips', address='GPIB::1::INSTR',
                   visalib=visalib,
                   field_limits=0)


@hst.given(x=hst.strategies.floats(min_value=-3, max_value=3),
           y=hst.strategies.floats(min_value=-3, max_value=3),
           z=hst.strategies.floats(min_value=-3, max_value=3))
def test_field_limits(x, y, z, driver_spher_lim, driver_cyl_lim):
    """
    Try with a few different field_limits functions and see if we get no-go
    exceptions when we expect them
    """

    # TODO: there really isn't a reason to do this tuple-by-tuple
    # and not point-by-point. Extend the test to do that.

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
        if 'RTOS' in mssg:
            axis = mssg[mssg.find('GRP')+3:mssg.find('GRP')+4]
            order.append(axis.lower())
    return order


@hst.given(x=hst.strategies.floats(min_value=-3, max_value=3),
           y=hst.strategies.floats(min_value=-3, max_value=3),
           z=hst.strategies.floats(min_value=-3, max_value=3))
def test_ramp_safely(driver, x, y, z, caplog):
    """
    Test that we get the first-down-then-up order right
    """
    # reset the instrument to default
    driver.GRPX.ramp_status('HOLD')
    driver.GRPY.ramp_status('HOLD')
    driver.GRPZ.ramp_status('HOLD')

    # the current field values are always zero for the sim. instr.

    driver.x_target(x)
    driver.y_target(y)
    driver.z_target(z)

    exp_order = np.array(['x', 'y', 'z'])[np.argsort(np.abs(np.array([x, y, z])))]

    with caplog.at_level(logging.DEBUG, logger='qcodes.instrument.visa'):
        caplog.clear()
        driver._ramp_safely()
        ramp_order = get_ramp_order(caplog.records)

    assert ramp_order == list(exp_order)
