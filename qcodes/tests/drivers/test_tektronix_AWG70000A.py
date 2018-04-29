from io import StringIO

import pytest
from hypothesis import given, settings
import hypothesis.strategies as hst
from lxml import etree

from qcodes.instrument_drivers.tektronix.AWG70002A import AWG70002A
from qcodes.instrument_drivers.tektronix.AWG70000A import AWG70000A
import qcodes.instrument.sims as sims
visalib = sims.__file__.replace('__init__.py', 'Tektronix_AWG70000A.yaml@sim')


@pytest.fixture(scope='function')
def awg2():
    awg2_sim = AWG70002A('awg2_sim',
                         address='GPIB0::2::65535::INSTR',
                         visalib=visalib)
    yield awg2_sim

    awg2_sim.close()


def test_init_awg2(awg2):

    idn_dict = awg2.IDN()

    assert idn_dict['vendor'] == 'QCoDeS'


@settings(deadline=1500, max_examples=25)
@given(N=hst.integers(1, 1000))
def test_SML_successful_generation_vary_length(N):

    tw = [0]*N
    nreps = [1]*N
    ejs = [0]*N
    ejt = [0]*N
    goto = [0]*N
    wfm_names = [['ch{}pos{}'.format(ch, pos)
                  for pos in range(N)] for ch in range(1, 3)]

    seqname = 'seq'

    smlstring = AWG70000A._makeSMLFile(tw, nreps, ejs, ejt, goto,
                                       wfm_names, seqname)

    # This line will raise an exception if the XML is not valid
    etree.parse(StringIO(smlstring))


@given(num_samples=hst.integers(min_value=2400),
       markers_included=hst.booleans())
def test_WFMXHeader_succesful(num_samples, markers_included):

    xmlstr = AWG70000A._makeWFMXFileHeader(num_samples, markers_included)
    etree.parse(StringIO(xmlstr))


@given(num_samples=hst.integers(max_value=2399),
       markers_included=hst.booleans())
def test_WFMXHeader_failing(num_samples, markers_included):
    with pytest.raises(ValueError):
        AWG70000A._makeWFMXFileHeader(num_samples, markers_included)


# TODO: Add some failing tests for inproper input
