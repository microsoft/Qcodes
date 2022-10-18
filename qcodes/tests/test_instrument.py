"""
Test suite for Instrument and InstrumentBase
"""
import contextlib
import io
import re
import weakref

import pytest

from qcodes.instrument import Instrument, InstrumentBase, find_or_create_instrument
from qcodes.metadatable import Metadatable
from qcodes.parameters import Function, Parameter

from .instrument_mocks import (
    DummyChannelInstrument,
    DummyFailingInstrument,
    DummyInstrument,
    MockMetaParabola,
    MockParabola,
)


@pytest.fixture(name="testdummy", scope="function")
def _dummy_dac():
    instrument = DummyInstrument(name="testdummy", gates=["dac1", "dac2", "dac3"])
    try:
        yield instrument
    finally:
        instrument.close()


@pytest.fixture(name="testdummychannelinstr", scope="function")
def _dummy_channel_instr():
    instrument = DummyChannelInstrument(name="testdummy")
    try:
        yield instrument
    finally:
        instrument.close()


@pytest.fixture(name="parabola", scope="function")
def _dummy_parabola():
    instrument = MockParabola("parabola")
    try:
        yield instrument
    finally:
        instrument.close()


@pytest.fixture(name='close_before_and_after', scope='function')
def _close_before_and_after():
    Instrument.close_all()
    try:
        yield
    finally:
        Instrument.close_all()


def test_validate_function(testdummy):
    testdummy.validate_status()  # test the instrument has valid values

    testdummy.dac1.cache._value = 1000  # overrule the validator
    testdummy.dac1.cache._raw_value = 1000  # overrule the validator
    with pytest.raises(Exception):
        testdummy.validate_status()


def test_check_instances(testdummy):
    with pytest.raises(KeyError, match='Another instrument has the name: testdummy'):
        DummyInstrument(name='testdummy', gates=['dac1', 'dac2', 'dac3'])

    assert Instrument.instances() == []
    assert DummyInstrument.instances() == [testdummy]
    assert testdummy.instances() == [testdummy]


def test_instrument_fail(close_before_and_after):
    with pytest.raises(RuntimeError, match="Failed to create instrument"):
        DummyFailingInstrument(name="failinginstrument")

    assert Instrument.instances() == []
    assert DummyFailingInstrument.instances() == []
    assert Instrument._all_instruments == {}


def test_instrument_on_invalid_identifier(close_before_and_after):
    # Check if warning and error raised when invalid identifer name given
    with pytest.warns(
        UserWarning, match="Changed !-name to !_name for instrument identifier"
    ):
        with pytest.raises(ValueError, match="!_name invalid instrument identifier"):
            DummyInstrument(name="!-name")

    assert Instrument.instances() == []
    assert DummyInstrument.instances() == []
    assert Instrument._all_instruments == {}


def test_instrument_warns_on_hyphen_in_name(close_before_and_after):
    # Check if warning is raised and name is valid
    # identifier when dashes '-' are converted to underscores '_'
    with pytest.warns(
        UserWarning, match="Changed -name to _name for instrument identifier"
    ):
        instr = DummyInstrument(name="-name")

    assert instr.name == "_name"
    assert Instrument.instances() == []
    assert DummyInstrument.instances() == [instr]
    assert Instrument._all_instruments != {}


def test_instrument_allows_channel_name_starting_with_number(close_before_and_after):
    instr = DummyChannelInstrument(name="foo", channel_names=["1", "2", "3"])

    for chan in instr.channels:
        assert chan.short_name.isidentifier() is False
        assert chan.full_name.isidentifier() is True
    assert Instrument.instances() == []
    assert DummyChannelInstrument.instances() == [instr]
    assert Instrument._all_instruments != {}


def test_instrument_channel_name_raise_on_invalid(close_before_and_after):
    with pytest.raises(ValueError, match="foo_☃ invalid instrument identifier"):
        DummyChannelInstrument(name="foo", channel_names=["☃"])
    assert Instrument.instances() == []
    assert DummyChannelInstrument.instances() == []
    assert Instrument._all_instruments == {}


def test_instrument_retry_with_same_name(close_before_and_after):
    with pytest.raises(RuntimeError, match="Failed to create instrument"):
        DummyFailingInstrument(name="failinginstrument")
    instr = DummyFailingInstrument(name="failinginstrument", fail=False)

    # Check that the instrument is successfully registered after failing first
    assert Instrument.instances() == []
    assert DummyFailingInstrument.instances() == [instr]
    expected_dict = weakref.WeakValueDictionary()
    expected_dict["failinginstrument"] = instr
    assert Instrument._all_instruments == expected_dict


def test_attr_access(testdummy):

    # test the instrument works
    testdummy.dac1.set(10)
    val = testdummy.dac1.get()
    assert val == 10

    # close the instrument
    testdummy.close()

    # make sure the name property still exists
    assert hasattr(testdummy, 'name')
    assert testdummy.name == 'testdummy'

    # make sure we can still print the instrument
    assert "testdummy" in testdummy.__repr__()
    assert "testdummy" in str(testdummy)

    # make sure the gate is removed
    assert not hasattr(testdummy, "dac1")


def test_attr_access_channels(testdummychannelinstr):
    instr = testdummychannelinstr

    channel = instr.channels[0]
    # close the instrument
    instr.close()

    # make sure the name property still exists
    assert hasattr(instr, "name")
    assert instr.name == "testdummy"
    assert instr.full_name == "testdummy"
    assert instr.short_name == "testdummy"

    # make sure we can still print the instrument
    assert "testdummy" in instr.__repr__()
    assert "testdummy" in str(instr)

    # make sure the submodules, parameters, and functions are removed
    assert not hasattr(instr, "parameters")
    assert not hasattr(instr, "submodules")
    assert not hasattr(instr, "instrument_modules")
    assert not hasattr(instr, "functions")

    assert channel.name == "testdummy_ChanA"
    assert channel.full_name == "testdummy_ChanA"
    assert channel.short_name == "ChanA"
    assert not hasattr(channel, "parameters")
    assert not hasattr(channel, "submodules")
    assert not hasattr(channel, "instrument_modules")
    assert not hasattr(channel, "functions")


def test_get_idn(testdummy):
    idn = {
        "vendor": "QCoDeS",
        "model": str(testdummy.__class__),
        "seral": "NA",
        "firmware": "NA",
    }
    assert testdummy.get_idn() == idn


def test_repr(testdummy):
    assert repr(testdummy) == '<DummyInstrument: testdummy>'


def test_add_remove_f_p(testdummy):
    with pytest.raises(KeyError, match="Duplicate parameter name dac1"):
        testdummy.add_parameter("dac1", get_cmd="foo")

    testdummy.add_function('function', call_cmd='foo')

    with pytest.raises(KeyError, match='Duplicate function name function'):
        testdummy.add_function('function', call_cmd='foo')

    testdummy.add_function('dac1', call_cmd='foo')

    # test custom __get_attr__ for functions
    fcn = testdummy['function']
    assert isinstance(fcn, Function)
    # by design, one gets the parameter if a function exists
    # and has same name
    dac1 = testdummy['dac1']
    assert isinstance(dac1, Parameter)


def test_instances(testdummy, parabola):
    instruments = [testdummy, parabola]
    for instrument in instruments:
        for other_instrument in instruments:
            instances = instrument.instances()
            # check that each instrument is in only its own
            if other_instrument is instrument:
                assert instrument in instances
            else:
                assert other_instrument not in instances

            # check that we can find each instrument from any other
            assert instrument is other_instrument.find_instrument(instrument.name)

        # check that we can find this instrument from the base class
        assert instrument is Instrument.find_instrument(instrument.name)


def test_is_valid(testdummy):
    assert Instrument.is_valid(testdummy)
    testdummy.close()
    assert not Instrument.is_valid(testdummy)


def test_snapshot_value(testdummy):
    testdummy.add_parameter('has_snapshot_value',
                            parameter_class=Parameter,
                            initial_value=42,
                            snapshot_value=True,
                            get_cmd=None, set_cmd=None)
    testdummy.add_parameter('no_snapshot_value',
                            parameter_class=Parameter,
                            initial_value=42,
                            snapshot_value=False,
                            get_cmd=None, set_cmd=None)

    snapshot = testdummy.snapshot()

    assert 'name' in snapshot
    assert 'testdummy' in snapshot['name']

    assert 'value' in snapshot['parameters']['has_snapshot_value']
    assert 42 == snapshot['parameters']['has_snapshot_value']['value']
    assert 'value' not in snapshot['parameters']['no_snapshot_value']


def test_meta_instrument(parabola):
    mock_instrument = MockMetaParabola("mock_parabola", parabola)

    # Check that the mock instrument can return values
    assert mock_instrument.parabola() == parabola.parabola()
    mock_instrument.x(1)
    mock_instrument.y(2)
    assert mock_instrument.parabola() == parabola.parabola()
    assert mock_instrument.parabola() != 0

    # Add a scaling factor
    mock_instrument.gain(2)
    assert mock_instrument.parabola() == parabola.parabola()*2

    # Check snapshots
    snap = mock_instrument.snapshot(update=True)
    assert "parameters" in snap
    assert "gain" in snap["parameters"]
    assert snap["parameters"]["gain"]["value"] == 2

    # Check printable snapshot
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        mock_instrument.print_readable_snapshot()
    readable_snap = f.getvalue()

    # Line length satisfied
    assert all(len(line) <= 80 for line in readable_snap.splitlines())
    # Gain is included in output with correct value
    assert re.search(r"gain[ \t]+:[ \t]+2", readable_snap) is not None


def test_find(testdummy):
    """Test finding an existing instrument"""

    instr_2 = find_or_create_instrument(
        DummyInstrument, name='testdummy', gates=['dac1', 'dac2', 'dac3'])

    assert instr_2 is testdummy
    assert instr_2.name == testdummy.name


def test_find_same_name_but_different_class(close_before_and_after, request):
    """Test finding an existing instrument with different class"""
    instr = DummyInstrument(
        name='instr', gates=['dac1', 'dac2', 'dac3'])
    request.addfinalizer(instr.close)

    class GammyInstrument(Instrument):
        some_other_attr = 25

    # Find an instrument with the same name but of different class
    error_msg = ("Instrument instr is <class "
                 "'qcodes.tests.instrument_mocks.DummyInstrument'> but "
                 "<class 'qcodes.tests.test_instrument"
                 ".test_find_same_name_but_different_class.<locals>"
                 ".GammyInstrument'> was requested")

    with pytest.raises(TypeError, match=error_msg):
        _ = find_or_create_instrument(
            GammyInstrument, name='instr', gates=['dac1', 'dac2', 'dac3'])


def test_create(close_before_and_after, request):
    """Test creating an instrument that does not yet exist"""
    instr = find_or_create_instrument(
        DummyInstrument, name='instr', gates=['dac1', 'dac2', 'dac3'])
    request.addfinalizer(instr.close)
    assert 'instr' == instr.name


def test_other_exception(close_before_and_after):
    """Test an unexpected exception occurred during finding instrument"""
    with pytest.raises(TypeError, match="unhashable type: 'dict'"):
        # in order to raise an unexpected exception, and make sure it is
        # passed through the call stack, let's pass an empty dict instead
        # of a string with instrument name
        _ = find_or_create_instrument(DummyInstrument, {})


def test_recreate(close_before_and_after, request):
    """Test the case when instrument needs to be recreated"""
    instr = DummyInstrument(
        name='instr', gates=['dac1', 'dac2', 'dac3'])
    request.addfinalizer(instr.close)

    assert ['instr'] == list(Instrument._all_instruments.keys())

    instr_2 = find_or_create_instrument(
        DummyInstrument, name='instr', gates=['dac1', 'dac2'],
        recreate=True
    )
    request.addfinalizer(instr_2.close)

    assert ['instr'] == list(Instrument._all_instruments.keys())

    assert instr_2 in Instrument._all_instruments.values()
    assert instr not in Instrument._all_instruments.values()


def test_instrument_metadata(request):
    metadatadict = {1: "data", "some": "data"}
    instrument = DummyInstrument(name='testdummy', gates=['dac1', 'dac2', 'dac3'],
                                 metadata=metadatadict)
    request.addfinalizer(instrument.close)
    assert instrument.metadata == metadatadict


def test_instrumentbase_metadata():
    metadatadict = {1: "data", "some": "data"}
    instrument = InstrumentBase('instr', metadata=metadatadict)
    assert instrument.metadata == metadatadict


@pytest.mark.parametrize("cls", [(InstrumentBase), (Instrument)])
def test_instrument_label(cls):
    """Instrument uses nicely formatted label if available."""
    instrument = cls(name="name")
    assert instrument.label == "name"

    random_ascii = "~!@#$%^&*()_-+=`{}[];'\":,./<>?|\\ äöüß"
    instrument.label = random_ascii
    assert instrument.label == random_ascii

    label = "Nicely-formatted label"
    instrument = cls(name="name1", label=label)
    assert instrument.label == label


def test_snapshot_and_meta_attrs():
    """Test snapshot of InstrumentBase contains _meta_attrs attributes"""
    instr = InstrumentBase("instr", label="Label")

    assert instr.name == 'instr'

    assert hasattr(instr, "_meta_attrs")
    assert instr._meta_attrs == ["name", "label"]

    snapshot = instr.snapshot()

    assert 'name' in snapshot
    assert 'instr' == snapshot['name']

    assert "label" in snapshot
    assert "Label" == snapshot["label"]

    assert '__class__' in snapshot
    assert 'InstrumentBase' in snapshot['__class__']


class TestSnapshotType(Metadatable):

    __test__ = False

    def __init__(self, sample_value: int) -> None:
        super().__init__()
        self.sample_value = sample_value

    def snapshot_base(self, update=True, params_to_skip_update=None):
        return {"sample_key": self.sample_value}


class TestInstrument(InstrumentBase):

    __test__ = False

    def __init__(self, name, label) -> None:
        super().__init__(name, label=label)
        self._meta_attrs.extend(["test_attribute"])
        self._test_attribute = TestSnapshotType(12)

    @property
    def test_attribute(self) -> TestSnapshotType:
        return self._test_attribute


def test_snapshot_and_meta_attrs2():
    """Test snapshot of child of InstrumentBase which contains _meta_attrs attribute that is itself Metadatable"""
    instr = TestInstrument("instr", label="Label")

    assert instr.name == "instr"

    assert hasattr(instr, "_meta_attrs")
    assert instr._meta_attrs == ["name", "label", "test_attribute"]

    snapshot = instr.snapshot()

    assert "name" in snapshot
    assert "instr" == snapshot["name"]

    assert "label" in snapshot
    assert "Label" == snapshot["label"]

    assert "__class__" in snapshot
    assert "TestInstrument" in snapshot["__class__"]

    assert "test_attribute" in snapshot
    assert {"sample_key": 12} == snapshot["test_attribute"]
