import os

import pytest

from qcodes.instrument_drivers.tektronix.AWG5208 import AWG5208


@pytest.fixture(scope='function')
def awg():
    awg_sim = AWG5208(
        "awg_sim", address="GPIB0::1::INSTR", pyvisa_sim_file="Tektronix_AWG5208.yaml"
    )
    yield awg_sim

    awg_sim.close()


def test_init_awg(awg) -> None:

    idn_dict = awg.IDN()

    assert idn_dict['vendor'] == 'QCoDeS'


def test_channel_resolution_docstring(awg) -> None:

    expected_docstring = ("12 bit resolution allows for four "
                          "markers, 13 bit resolution "
                          "allows for three, etc. with 16 bit "
                          "allowing for ZERO markers")

    actual_docstring = awg.ch1.resolution.__doc__.split(os.linesep)[0]

    assert actual_docstring == expected_docstring
