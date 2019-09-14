import pytest
import numpy as np

from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Tektronix_AWG5014C.yaml@sim')


@pytest.fixture(scope='function')
def awg():
    awg_sim = Tektronix_AWG5014('awg_sim',
                                address='GPIB0::1::INSTR',
                                timeout=1,
                                terminator='\n',
                                visalib=visalib)
    yield awg_sim

    awg_sim.close()


def test_init_awg(awg):

    idn_dict = awg.IDN()

    assert idn_dict['vendor'] == 'QCoDeS'


def test_pack_waveform(awg):

    N = 25

    waveform = np.random.rand(N)
    m1 = np.random.randint(0, 2, N)
    m2 = np.random.randint(0, 2, N)

    package = awg._pack_waveform(waveform, m1, m2)

    assert package is not None


def test_make_awg_file(awg):

    N = 25

    waveforms = [[np.random.rand(N)]]
    m1s = [[np.random.randint(0, 2, N)]]
    m2s = [[np.random.randint(0, 2, N)]]
    nreps = [1]
    trig_waits = [0]
    goto_states = [0]
    jump_tos = [0]

    awgfile = awg.make_awg_file(waveforms,
                                m1s,
                                m2s,
                                nreps,
                                trig_waits,
                                goto_states,
                                jump_tos,
                                preservechannelsettings=False)

    assert len(awgfile) > 0
