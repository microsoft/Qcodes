from qcodes.instrument_drivers.stanford_research.SR830_channels import SR830
import qcodes.instrument.sims as sims


# path to the .yaml file containing the simulated instrument
visalib = sims.__file__.replace('__init__.py', 'SR830.yaml@sim')


def test_init():
    sr830 = SR830("sr830", "GPIB::1::INSTR", terminator="\n", visalib=visalib)
    assert sr830.channels[0].short_name == "channel1"
    assert sr830.channels[1].short_name == "channel2"
