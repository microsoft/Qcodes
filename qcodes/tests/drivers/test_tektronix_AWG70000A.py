from io import StringIO, BytesIO
import zipfile

import pytest
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as hst
from lxml import etree

from qcodes.instrument_drivers.tektronix.AWG70002A import AWG70002A
from qcodes.instrument_drivers.tektronix.AWG70000A import AWG70000A
import qcodes.instrument.sims as sims
import qcodes.tests.drivers.auxiliary_files as auxfiles

from broadbean.sequence import InvalidForgedSequenceError

visalib = sims.__file__.replace('__init__.py', 'Tektronix_AWG70000A.yaml@sim')


def strip_outer_tags(sml: str) -> str:
    """
    Helper function to strip the outer tags of an SML file so that it
    complies with the schema provided by tektronix
    """
    # make function idempotent
    if not sml[1:9] == 'DataFile':
        print('Incorrect file format or outer tags '
              'already stripped')
        return sml

    ind1 = sml.find('>\r\n')
    sml = sml[ind1+3:]  # strip off the opening DataFile tag
    sml = sml[:-24]  # remove the </Setup> and closing tag
    return sml


@pytest.fixture(scope='function')
def awg2():
    awg2_sim = AWG70002A('awg2_sim',
                         address='GPIB0::2::65535::INSTR',
                         visalib=visalib)
    yield awg2_sim

    awg2_sim.close()


@pytest.fixture(scope='module')
def random_wfm_m1_m2_package():
    """
    Make a random 2400 points np.array([wfm, m1, m2]).
    The waveform has values in [-0.1, 0.1)
    """
    wfm = 0.2*(np.random.rand(2400) - 0.5)
    m1 = np.random.randint(0, 2, 2400)
    m2 = np.random.randint(0, 2, 2400)
    return np.array([wfm, m1, m2])


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

        2400 points long, the minimum allowed by the instrument
        """
        data = {n: {} for n in range(1, 1 + num_chans)}
        for key in data.keys():
            data[key] = {'wfm': np.random.randn(2400),
                         'm1': np.random.randint(0, 2, 2400),
                         'm2': np.random.randint(0, 2, 2400)}

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


@settings(deadline=2500, max_examples=7)
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
                                       wfm_names, seqname, chans=3)

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
    make_seqx = AWG70000A.make_SEQX_from_forged_sequence

    # TODO: the number of channels is defined in the
    # forged_sequence fixture but used here
    chan_map = {n: n for n in range(1, 4)}

    # the input dict (first argument) is not a valid forged
    # sequence dict
    with pytest.raises(InvalidForgedSequenceError):
        make_seqx({}, [], 'yolo', {})

    # wrong number of channel amplitudes
    with pytest.raises(ValueError):
        make_seqx(forged_sequence, amplitudes=[1, 1],
                  seqname='dummyname', channel_mapping=chan_map)

    # wrong channel mapping keys
    with pytest.raises(ValueError):
        make_seqx(forged_sequence, [1, 1, 1],
                  seqname='dummyname',
                  channel_mapping={1: None, 3: None})

    # wrong channel mapping values
    with pytest.raises(ValueError):
        make_seqx(forged_sequence, [1, 1, 1],
                  seqname='dummyname',
                  channel_mapping={1: 10, 2: 8, 3: -1})


def test_seqxfile_from_fs(forged_sequence):

    # typing convenience
    make_seqx = AWG70000A.make_SEQX_from_forged_sequence

    path_to_schema = auxfiles.__file__.replace('__init__.py',
                                               'awgSeqDataSets.xsd')

    with open(path_to_schema, 'r') as fid:
        raw_schema = fid.read()

    schema = etree.XMLSchema(etree.XML(raw_schema.encode('utf-8')))
    parser = etree.XMLParser(schema=schema)

    seqx = make_seqx(forged_sequence, [10, 10, 10], 'myseq')

    zf = zipfile.ZipFile(BytesIO(seqx))

    # Check for double/missing file extensions
    for filename in zf.namelist():
        assert filename.count('.') == 1

    # validate the SML files (describing sequences)
    seq_names = [fn for fn in zf.namelist() if 'Sequences/' in fn]

    for seq_name in seq_names:
        with zf.open(seq_name) as fid:
            raw_seq_sml = fid.read()
            str_seq_sml = strip_outer_tags(raw_seq_sml.decode())
            # the next line parses using the schema and will raise
            # XMLSyntaxError if something is wrong
            etree.XML(str_seq_sml, parser=parser)

# TODO: Add some failing tests for inproper input


def test_makeSEQXFile(awg2, random_wfm_m1_m2_package):
    """
    Test that this function works (for some input)
    """

    seqlen = 25
    chans = 3

    wfmpkg = random_wfm_m1_m2_package

    trig_waits = [0]*seqlen
    nreps = [1]*seqlen
    event_jumps = [0]*seqlen
    event_jump_to = [0]*seqlen
    go_to = [0]*seqlen
    wfms = [[wfmpkg for i in range(seqlen)] for j in range(chans)]
    amplitudes = [0.5]*chans
    seqname = "testseq"

    seqxfile = awg2.makeSEQXFile(trig_waits, nreps, event_jumps,
                                 event_jump_to, go_to, wfms,
                                 amplitudes, seqname)