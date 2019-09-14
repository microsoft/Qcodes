import os

import pytest

from qcodes.instrument_drivers.tektronix.AWG5208 import AWG5208
import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Tektronix_AWG5208.yaml@sim')


@pytest.fixture(scope='function')
def awg():
    awg_sim = AWG5208('awg_sim',
                      address='GPIB0::1::INSTR',
                      visalib=visalib)
    yield awg_sim

    awg_sim.close()


def test_init_awg(awg):

    idn_dict = awg.IDN()

    assert idn_dict['vendor'] == 'QCoDeS'


def test_channel_resolution_docstring(awg):

    expected_docstring = ("12 bit resolution allows for four "
                          "markers, 13 bit resolution "
                          "allows for three, etc. with 16 bit "
                          "allowing for ZERO markers")

    actual_docstring = awg.ch1.resolution.__doc__.split(os.linesep)[0]

    assert actual_docstring == expected_docstring
