import time

import qcodes
from qcodes.instrument_drivers.stanford_research.SR830_channels import SR830
import qcodes.instrument.sims as sims


# path to the .yaml file containing the simulated instrument
visalib = sims.__file__.replace('__init__.py', 'SR830.yaml@sim')


def test_init():
    sr830 = SR830("sr830", "GPIB::1::INSTR", terminator="\n", visalib=visalib)

    assert sr830.channels[0].short_name == "channel1"
    assert sr830.channels[1].short_name == "channel2"

    sr830.channels[0].display('X')
    sr830.channels[0].ratio('none')
    sr830.buffer_SR(512)

    sr830.buffer_reset()
    sr830.buffer_start()
    time.sleep(1)
    sr830.buffer_pause()

    sr830.channels[0].databuffer.prepare_buffer_readout()
    meas = qcodes.Measure(sr830.channels[0].databuffer)