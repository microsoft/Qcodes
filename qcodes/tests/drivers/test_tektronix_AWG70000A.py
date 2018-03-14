from io import StringIO

import pytest
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as hst
from lxml import etree

from qcodes.instrument_drivers.tektronix.AWG70002A import AWG70002A
from qcodes.instrument_drivers.tektronix.AWG70000A import AWG70000A
import qcodes.instrument.sims as sims

from broadbean.broadbean import InvalidForgedSequenceError

visalib = sims.__file__.replace('__init__.py', 'Tektronix_AWG70000A.yaml@sim')


@pytest.fixture(scope='function')
def awg2():
    awg2_sim = AWG70002A('awg2_sim',
                         address='GPIB0::2::65535::INSTR',
                         visalib=visalib)
    yield awg2_sim

    awg2_sim.close()


@pytest.fixture(scope='module')
def forged_sequence():
    """
    Return an example forged sequence containing a
    subsequence
    """

    N = 5
    num_chans = 3
    types = ['element']*(N-1) + ['subsequence']

    def random_element(num_chans):
        """
        Return an element with random values

        200 points long
        """
        data = {n: {} for n in range(1, 1 + num_chans)}
        for key in data.keys():
            data[key] = {'wfm': np.random.randn(200),
                         'm1': np.random.randint(0, 2, 200),
                         'm2': np.random.randint(0, 2, 200)}

        return data

    seq = {i: {} for i in range(1, 1 + N)}
    for pos1, typ in zip(seq.keys(), types):
        seq[pos1] = {'type': typ,
                     'content': {},
                     'sequencing': {}}

    for pos1 in range(1, N):
        seq[pos1]['content'] = {1: {'data': random_element(num_chans)}}

    # and finally add the subsequence
    seq[N]['content'] = {1: {'data': random_element(num_chans),
                             'sequencing': {'nreps': 2}},
                         2: {'data': random_element(num_chans),
                             'sequencing': {'nreps': 2}}}

    return seq


def test_init_awg2(awg2):

    idn_dict = awg2.IDN()

    assert idn_dict['vendor'] == 'QCoDeS'


@settings(deadline=1500, max_examples=7)
@given(N=hst.integers(1, 100))
def test_SML_successful_generation_vary_length(N):

    tw = [0]*N
    nreps = [1]*N
    ejs = [0]*N
    ejt = [0]*N
    goto = [0]*N
    wfm_names = [['pos{}ch{}'.format(pos, ch)
                  for ch in range(1, 3)] for pos in range(N)]

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


def test_seqxfilefromfs_failing(forged_sequence):

    # typing convenience
    make_seqx = AWG70000A.makeSEQXFileFromForgedSequence

    # TODO: the number of channels is defined in the
    # forged_sequence fixture but used here
    chan_map = {n: n for n in range(1, 4)}

    # the input dict (first argument) is not a valid forged
    # sequence dict
    with pytest.raises(InvalidForgedSequenceError):
        make_seqx({}, [], {})

    # wrong number of channel amplitudes
    with pytest.raises(ValueError):
        make_seqx(forged_sequence, amplitudes=[1, 1], channel_mapping=chan_map)

    # wrong channel mapping keys
    with pytest.raises(ValueError):
        make_seqx(forged_sequence, [1, 1, 1],
                  channel_mapping={1: None, 3: None})

    # wrong channel mapping values
    with pytest.raises(ValueError):
        make_seqx(forged_sequence, [1, 1, 1],
                  channel_mapping={1: 10, 2: 8, 3: -1})

# TODO: Add some failing tests for inproper input
