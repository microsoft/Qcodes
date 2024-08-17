import numpy as np
import pytest

from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014


@pytest.fixture(scope="function")
def awg():
    awg_sim = Tektronix_AWG5014(
        "awg_sim",
        address="GPIB0::1::INSTR",
        timeout=1,
        terminator="\n",
        pyvisa_sim_file="Tektronix_AWG5014C.yaml",
    )
    yield awg_sim

    awg_sim.close()


def test_init_awg(awg) -> None:
    idn_dict = awg.IDN()

    assert idn_dict["vendor"] == "QCoDeS"


def test_pack_waveform(awg) -> None:
    N = 25

    waveform = np.random.rand(N)
    m1 = np.random.randint(0, 2, N)
    m2 = np.random.randint(0, 2, N)

    package = awg._pack_waveform(waveform, m1, m2)

    assert package is not None


def test_make_awg_file(awg) -> None:
    N = 25

    waveforms = [[np.random.rand(N)]]
    m1s = [[np.random.randint(0, 2, N)]]
    m2s = [[np.random.randint(0, 2, N)]]
    nreps = [1]
    trig_waits = [0]
    goto_states = [0]
    jump_tos = [0]

    awgfile = awg.make_awg_file(
        waveforms,
        m1s,
        m2s,
        nreps,
        trig_waits,
        goto_states,
        jump_tos,
        preservechannelsettings=False,
    )

    assert len(awgfile) > 0
