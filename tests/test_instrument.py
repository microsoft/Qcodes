"""
Test suite for Instrument and InstrumentBase
"""

from __future__ import annotations

import contextlib
import gc
import io
import logging
import re
import socket
import sys
import warnings
import weakref
from typing import TYPE_CHECKING, Any, assert_type
from weakref import WeakValueDictionary

import pytest
from pytest import FixtureRequest

from qcodes.instrument import (
    Instrument,
    InstrumentBase,
    InstrumentChannel,
    InstrumentModule,
    IPInstrument,
    find_or_create_instrument,
)
from qcodes.instrument import instrument as instrument_module
from qcodes.instrument_drivers.mock_instruments import (
    DummyBase,
    DummyChannelInstrument,
    DummyFailingInstrument,
    DummyInstrument,
    MockMetaParabola,
    MockParabola,
)
from qcodes.metadatable import Metadatable
from qcodes.parameters import Function, Parameter
from qcodes.utils import strip_attrs

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from qcodes.metadatable.metadatable_base import Snapshot, SnapshotUpdate


@pytest.fixture(name="testdummy", scope="function")
def _dummy_dac() -> Iterator[DummyInstrument]:
    instrument = DummyInstrument(name="testdummy", gates=["dac1", "dac2", "dac3"])
    try:
        yield instrument
    finally:
        instrument.close()


@pytest.fixture(name="testdummychannelinstr", scope="function")
def _dummy_channel_instr() -> Iterator[DummyChannelInstrument]:
    instrument = DummyChannelInstrument(name="testdummy")
    try:
        yield instrument
    finally:
        instrument.close()


@pytest.fixture(name="parabola", scope="function")
def _dummy_parabola() -> Iterator[MockParabola]:
    instrument = MockParabola("parabola")
    try:
        yield instrument
    finally:
        instrument.close()


@pytest.fixture(name="empty_instrument_with_empty_submodule", scope="function")
def _empty_instrument_with_empty_submodule() -> Iterator[Instrument]:
    instrument = Instrument("EmptyInstrument")
    instrument.add_submodule("A", InstrumentModule(instrument, "B"))
    try:
        yield instrument
    finally:
        instrument.close()


@pytest.fixture(name="close_before_and_after", scope="function")
def _close_before_and_after() -> Iterator[None]:
    Instrument.close_all()
    try:
        yield
    finally:
        Instrument.close_all()


@pytest.fixture(name="automatic_gc_disabled", scope="function")
def _automatic_gc_disabled() -> Iterator[None]:
    """Disable automatic garbage collection so that only the test triggers it."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def test_instrument_type(request: pytest.FixtureRequest) -> None:
    # make sure that the type of the instrument is correct.
    # Due to our use of a metaclass for instrument this could be
    # incorrect. See comment in instrument.py
    testldummy = DummyInstrument("dummy")
    request.addfinalizer(testldummy.close)
    assert_type(testldummy, DummyInstrument)


def test_validate_function(testdummy: DummyInstrument) -> None:
    testdummy.validate_status()  # test the instrument has valid values

    testdummy.dac1.cache._value = 1000  # overrule the validator
    testdummy.dac1.cache._raw_value = 1000  # overrule the validator
    with pytest.raises(
        ValueError, match=r"1000 is invalid: must be between -800 and 400 inclusive"
    ):
        testdummy.validate_status()


def test_check_instances(testdummy: DummyInstrument) -> None:
    with pytest.raises(KeyError, match="Another instrument has the name: testdummy"):
        DummyInstrument(name="testdummy", gates=["dac1", "dac2", "dac3"])

    assert Instrument.instances() == []
    assert DummyInstrument.instances() == [testdummy]
    assert testdummy.instances() == [testdummy]


@pytest.mark.usefixtures("close_before_and_after")
def test_instrument_fail() -> None:
    with pytest.raises(RuntimeError, match="Failed to create instrument"):
        DummyFailingInstrument(name="failinginstrument")

    assert Instrument.instances() == []
    assert DummyFailingInstrument.instances() == []
    assert Instrument._all_instruments == WeakValueDictionary()


@pytest.mark.usefixtures("close_before_and_after")
def test_instrument_on_invalid_identifier() -> None:
    # Check if warning and error raised when invalid identifer name given
    with (
        pytest.warns(
            UserWarning, match="Changed !-name to !_name for instrument identifier"
        ),
        pytest.raises(ValueError, match="!_name invalid instrument identifier"),
    ):
        DummyInstrument(name="!-name")

    assert Instrument.instances() == []
    assert DummyInstrument.instances() == []
    assert Instrument._all_instruments == WeakValueDictionary()


@pytest.mark.usefixtures("close_before_and_after")
def test_instrument_warns_on_hyphen_in_name() -> None:
    # Check if warning is raised and name is valid
    # identifier when dashes '-' are converted to underscores '_'
    with pytest.warns(
        UserWarning, match="Changed -name to _name for instrument identifier"
    ):
        instr = DummyInstrument(name="-name")

    assert instr.name == "_name"
    assert Instrument.instances() == []
    assert DummyInstrument.instances() == [instr]
    assert Instrument._all_instruments != WeakValueDictionary()


@pytest.mark.usefixtures("close_before_and_after")
def test_instrument_allows_channel_name_starting_with_number() -> None:
    instr = DummyChannelInstrument(name="foo", channel_names=["1", "2", "3"])

    for chan in instr.channels:
        assert chan.short_name.isidentifier() is False
        assert chan.full_name.isidentifier() is True
    assert Instrument.instances() == []
    assert DummyChannelInstrument.instances() == [instr]
    assert Instrument._all_instruments != WeakValueDictionary()


@pytest.mark.usefixtures("close_before_and_after")
def test_instrument_channel_name_raise_on_invalid() -> None:
    with pytest.raises(ValueError, match="foo_☃ invalid instrument identifier"):
        DummyChannelInstrument(name="foo", channel_names=["☃"])
    assert Instrument.instances() == []
    assert DummyChannelInstrument.instances() == []
    assert Instrument._all_instruments == WeakValueDictionary()


@pytest.mark.usefixtures("close_before_and_after")
def test_instrument_retry_with_same_name() -> None:
    with pytest.raises(RuntimeError, match="Failed to create instrument"):
        DummyFailingInstrument(name="failinginstrument")
    instr = DummyFailingInstrument(name="failinginstrument", fail=False)

    # Check that the instrument is successfully registered after failing first
    assert Instrument.instances() == []
    assert DummyFailingInstrument.instances() == [instr]
    expected_dict: WeakValueDictionary[str, Instrument] = weakref.WeakValueDictionary()
    expected_dict["failinginstrument"] = instr
    assert Instrument._all_instruments == expected_dict


def test_attr_access(testdummy: DummyInstrument) -> None:
    # test the instrument works
    testdummy.dac1.set(10)
    val = testdummy.dac1.get()
    assert val == 10

    # close the instrument
    testdummy.close()

    # make sure the name property still exists
    assert hasattr(testdummy, "name")
    assert testdummy.name == "testdummy"

    # make sure we can still print the instrument
    assert "testdummy" in testdummy.__repr__()
    assert "testdummy" in str(testdummy)

    # make sure the gate is removed
    assert not hasattr(testdummy, "dac1")


def test_parameter_property(testdummy: DummyInstrument) -> None:
    # since this is added dynamically we cannot know the type statically
    assert_type(testdummy.dac1, Any)
    # this is an assigned attribute so we know it statically
    assert_type(testdummy.fixed_parameter, Parameter)

    assert testdummy.fixed_parameter.get() == 5
    testdummy.fixed_parameter.set(10)
    assert testdummy.fixed_parameter.get() == 10


def test_attr_access_channels(testdummychannelinstr: DummyChannelInstrument) -> None:
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


def test_get_idn(testdummy: DummyInstrument) -> None:
    idn = {
        "vendor": "QCoDeS",
        "model": str(testdummy.__class__),
        "serial": "NA",
        "firmware": "NA",
    }
    assert testdummy.get_idn() == idn


def test_repr(testdummy: DummyInstrument) -> None:
    assert repr(testdummy) == "<DummyInstrument: testdummy>"


def test_add_remove_f_p(testdummy: DummyInstrument) -> None:
    with pytest.raises(KeyError, match="Duplicate parameter name dac1"):
        testdummy.add_parameter("dac1", get_cmd="foo")

    testdummy.add_function("function", call_cmd="foo")

    with pytest.raises(KeyError, match="Duplicate function name function"):
        testdummy.add_function("function", call_cmd="foo")

    testdummy.add_function("dac1", call_cmd="foo")

    # test custom __get_attr__ for functions
    with pytest.warns(
        PendingDeprecationWarning,
        match="Use attributes directly on the instrument object instead",
    ):
        fcn = testdummy["function"]
    assert isinstance(fcn, Function)
    # by design, one gets the parameter if a function exists
    # and has same name
    with pytest.warns(
        PendingDeprecationWarning,
        match="Use attributes directly on the instrument object instead",
    ):
        dac1 = testdummy["dac1"]
    assert isinstance(dac1, Parameter)


def test_instances(testdummy: DummyInstrument, parabola: MockParabola) -> None:
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


def test_is_valid(testdummy: DummyInstrument) -> None:
    assert Instrument.is_valid(testdummy)
    testdummy.close()
    assert not Instrument.is_valid(testdummy)


def test_snapshot_value(testdummy: DummyInstrument) -> None:
    testdummy.add_parameter(
        "has_snapshot_value",
        parameter_class=Parameter,
        initial_value=42,
        snapshot_value=True,
        get_cmd=None,
        set_cmd=None,
    )
    testdummy.add_parameter(
        "no_snapshot_value",
        parameter_class=Parameter,
        initial_value=42,
        snapshot_value=False,
        get_cmd=None,
        set_cmd=None,
    )

    snapshot = testdummy.snapshot()

    assert "name" in snapshot
    assert "testdummy" in snapshot["name"]

    assert "value" in snapshot["parameters"]["has_snapshot_value"]
    assert 42 == snapshot["parameters"]["has_snapshot_value"]["value"]
    assert "value" not in snapshot["parameters"]["no_snapshot_value"]


def test_meta_instrument(parabola: MockParabola) -> None:
    mock_instrument = MockMetaParabola("mock_parabola", parabola)

    # Check that the mock instrument can return values
    assert mock_instrument.parabola() == parabola.parabola()
    mock_instrument.x(1)
    mock_instrument.y(2)
    assert mock_instrument.parabola() == parabola.parabola()
    assert mock_instrument.parabola() != 0

    # Add a scaling factor
    mock_instrument.gain(2)
    assert mock_instrument.parabola() == parabola.parabola() * 2

    # Check snapshots
    snap = mock_instrument.snapshot(update="All")
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


def test_empty_instrument(empty_instrument_with_empty_submodule: Instrument) -> None:
    empty_instrument_with_empty_submodule.print_readable_snapshot()


def test_find(testdummy: DummyInstrument) -> None:
    """Test finding an existing instrument"""

    instr_2 = find_or_create_instrument(
        DummyInstrument, name="testdummy", gates=["dac1", "dac2", "dac3"]
    )

    assert instr_2 is testdummy
    assert instr_2.name == testdummy.name


@pytest.mark.usefixtures("close_before_and_after")
def test_find_same_name_but_different_class(request: FixtureRequest) -> None:
    """Test finding an existing instrument with different class"""
    instr = DummyInstrument(name="instr", gates=["dac1", "dac2", "dac3"])
    request.addfinalizer(instr.close)

    class GammyInstrument(Instrument):
        some_other_attr = 25

    # Find an instrument with the same name but of different class
    error_msg = (
        "Instrument instr is <class "
        "'qcodes.instrument_drivers.mock_instruments.DummyInstrument'> but "
        "<class 'tests.test_instrument"
        ".test_find_same_name_but_different_class.<locals>"
        ".GammyInstrument'> was requested"
    )

    with pytest.raises(TypeError, match=error_msg):
        _ = find_or_create_instrument(
            GammyInstrument, name="instr", gates=["dac1", "dac2", "dac3"]
        )


@pytest.mark.usefixtures("close_before_and_after")
def test_create(request: FixtureRequest) -> None:
    """Test creating an instrument that does not yet exist"""
    instr = find_or_create_instrument(
        DummyInstrument, name="instr", gates=["dac1", "dac2", "dac3"]
    )
    request.addfinalizer(instr.close)
    assert "instr" == instr.name


@pytest.mark.usefixtures("close_before_and_after")
def test_other_exception() -> None:
    """Test an unexpected exception occurred during finding instrument"""
    with pytest.raises(TypeError, match="unhashable type: 'dict'"):
        # in order to raise an unexpected exception, and make sure it is
        # passed through the call stack, let's pass an empty dict instead
        # of a string with instrument name
        _ = find_or_create_instrument(DummyInstrument, {})  #  type: ignore[arg-type]


@pytest.mark.usefixtures("close_before_and_after")
def test_recreate(request: FixtureRequest) -> None:
    """Test the case when instrument needs to be recreated"""
    instr = DummyInstrument(name="instr", gates=["dac1", "dac2", "dac3"])
    request.addfinalizer(instr.close)

    assert ["instr"] == list(Instrument._all_instruments.keys())

    instr_2 = find_or_create_instrument(
        DummyInstrument, name="instr", gates=["dac1", "dac2"], recreate=True
    )
    request.addfinalizer(instr_2.close)

    assert ["instr"] == list(Instrument._all_instruments.keys())

    assert instr_2 in Instrument._all_instruments.values()
    assert instr not in Instrument._all_instruments.values()


@pytest.mark.usefixtures("close_before_and_after")
def test_close_all_closes_all_instruments() -> None:
    """``close_all`` closes every registered instrument by default."""
    dummy = DummyInstrument(name="dummy", gates=["dac1"])
    parabola = MockParabola("parabola")

    assert Instrument.is_valid(dummy)
    assert Instrument.is_valid(parabola)

    Instrument.close_all()

    assert not Instrument.is_valid(dummy)
    assert not Instrument.is_valid(parabola)
    assert Instrument._all_instruments == WeakValueDictionary()


@pytest.mark.usefixtures("close_before_and_after")
def test_close_all_only_subclasses_from_leaf_class() -> None:
    """``only_subclasses`` on a leaf class leaves sibling classes open."""
    dummy = DummyInstrument(name="dummy", gates=["dac1"])
    parabola = MockParabola("parabola")

    # DummyInstrument and MockParabola are siblings (both subclass DummyBase),
    # so closing only DummyInstrument subclasses must leave the parabola open.
    DummyInstrument.close_all(only_subclasses=True)

    assert not Instrument.is_valid(dummy)
    assert Instrument.is_valid(parabola)

    # The remaining instrument can still be closed with a plain close_all.
    Instrument.close_all()
    assert not Instrument.is_valid(parabola)
    assert Instrument._all_instruments == WeakValueDictionary()


@pytest.mark.usefixtures("close_before_and_after")
def test_close_all_only_subclasses_from_base_class() -> None:
    """``only_subclasses`` closes instances of the class and its subclasses."""
    dummy = DummyInstrument(name="dummy", gates=["dac1"])
    parabola = MockParabola("parabola")

    # Both DummyInstrument and MockParabola are subclasses of DummyBase, so both
    # are closed when calling close_all on the shared base class.
    DummyBase.close_all(only_subclasses=True)

    assert not Instrument.is_valid(dummy)
    assert not Instrument.is_valid(parabola)
    assert Instrument._all_instruments == WeakValueDictionary()


@pytest.mark.usefixtures("close_before_and_after")
def test_close_all_only_subclasses_false_closes_everything() -> None:
    """``only_subclasses=False`` closes all instruments regardless of class."""
    dummy = DummyInstrument(name="dummy", gates=["dac1"])
    parabola = MockParabola("parabola")

    DummyInstrument.close_all(only_subclasses=False)

    assert not Instrument.is_valid(dummy)
    assert not Instrument.is_valid(parabola)
    assert Instrument._all_instruments == WeakValueDictionary()


@pytest.mark.usefixtures("close_before_and_after")
def test_close_all_log_status(caplog: pytest.LogCaptureFixture) -> None:
    """``log_status=True`` logs the closing of each instrument."""
    dummy = DummyInstrument(name="dummy", gates=["dac1"])

    with caplog.at_level(logging.INFO, logger="qcodes.instrument.instrument"):
        Instrument.close_all(log_status=True)

    assert "Closing all registered instruments" in caplog.text
    assert "Closing dummy" in caplog.text
    assert not Instrument.is_valid(dummy)


@pytest.mark.usefixtures("close_before_and_after")
def test_close_all_no_log_by_default(caplog: pytest.LogCaptureFixture) -> None:
    """``close_all`` does not log anything when ``log_status`` is not set."""
    dummy = DummyInstrument(name="dummy", gates=["dac1"])

    with caplog.at_level(logging.INFO, logger="qcodes.instrument.instrument"):
        Instrument.close_all()

    assert "Closing all registered instruments" not in caplog.text
    assert "Closing dummy" not in caplog.text
    assert not Instrument.is_valid(dummy)


def test_close_all_only_accepts_keyword_arguments() -> None:
    """The ``close_all`` options are keyword-only."""
    with pytest.raises(TypeError):
        Instrument.close_all(True)  # type: ignore[misc]


class _RegistryCollectingGarbageWhenIterated(WeakValueDictionary[str, Instrument]):
    """
    Instrument registry that runs a garbage collection in the middle of the
    first iteration over it.

    A garbage collection can run whenever objects are allocated, so this makes
    a rare but possible situation deterministic.
    """

    def __init__(self) -> None:
        super().__init__()
        self.collected_garbage = False

    def _collect_garbage_once(self) -> None:
        if not self.collected_garbage:
            self.collected_garbage = True
            gc.collect()

    def __iter__(self) -> Iterator[str]:
        for name in super().__iter__():
            yield name
            self._collect_garbage_once()

    def items(self) -> Iterator[tuple[str, Instrument]]:  # type: ignore[override]
        for item in super().items():
            yield item
            self._collect_garbage_once()


def _leak_instrument(name: str) -> None:
    """
    Create an instrument and drop it without closing it.

    The instrument references itself through its parameters, so it stays
    registered until the garbage collector finalizes it.
    """
    Instrument(name)
    assert Instrument.exist(name)


@pytest.mark.usefixtures("close_before_and_after", "automatic_gc_disabled")
def test_close_when_gc_finalizes_instrument_during_registry_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Closing an instrument unregisters it, so that its name can be reused, even
    if a garbage collection that finalizes another instrument interrupts an
    iteration over the registry.
    """
    monkeypatch.setattr(
        Instrument, "_all_instruments", _RegistryCollectingGarbageWhenIterated()
    )
    dummy = DummyInstrument(name="dummy", gates=["dac1"])
    _leak_instrument("leaked")

    dummy.close()

    assert not Instrument.exist("dummy")
    DummyInstrument(name="dummy", gates=["dac1"]).close()

    with pytest.warns(
        ResourceWarning, match="unclosed instrument <Instrument: leaked>"
    ):
        gc.collect()
    assert len(Instrument._all_instruments) == 0


@pytest.mark.usefixtures("close_before_and_after", "automatic_gc_disabled")
def test_close_all_when_gc_finalizes_instrument_during_registry_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``close_all`` closes all instruments even if a garbage collection that
    finalizes another instrument runs while it iterates over the registry.
    """
    registry = _RegistryCollectingGarbageWhenIterated()
    monkeypatch.setattr(Instrument, "_all_instruments", registry)
    dummy = DummyInstrument(name="dummy", gates=["dac1"])
    _leak_instrument("leaked")
    parabola = MockParabola("parabola")

    with pytest.warns(
        ResourceWarning, match="unclosed instrument <Instrument: leaked>"
    ):
        Instrument.close_all()

    assert registry.collected_garbage
    assert not Instrument.is_valid(dummy)
    assert not Instrument.is_valid(parabola)
    assert len(Instrument._all_instruments) == 0


@pytest.mark.usefixtures("close_before_and_after")
def test_closing_instrument_again_keeps_new_instrument_with_same_name() -> None:
    """
    Closing an instrument again does not unregister another instrument that
    has been registered under the same name in the meantime.
    """
    old_dummy = DummyInstrument(name="dummy", gates=["dac1"])
    old_dummy.close()
    new_dummy = DummyInstrument(name="dummy", gates=["dac1"])

    old_dummy.close()

    assert Instrument.find_instrument("dummy") is new_dummy
    new_dummy.close()


def test_instrument_metadata(request: FixtureRequest) -> None:
    metadatadict = {1: "data", "some": "data"}
    instrument = DummyInstrument(
        name="testdummy", gates=["dac1", "dac2", "dac3"], metadata=metadatadict
    )
    request.addfinalizer(instrument.close)
    assert instrument.metadata == metadatadict


def test_instrumentbase_metadata() -> None:
    metadatadict = {1: "data", "some": "data"}
    instrument = InstrumentBase("instr", metadata=metadatadict)
    assert instrument.metadata == metadatadict


@pytest.mark.parametrize("cls", [(InstrumentBase), (Instrument)])
def test_instrument_label(cls: type[InstrumentBase], request: FixtureRequest) -> None:
    """Instrument uses nicely formatted label if available."""
    instrument = cls(name="name")
    if isinstance(cls, type(Instrument)):
        request.addfinalizer(instrument.close)
    assert instrument.label == "name"

    random_ascii = "~!@#$%^&*()_-+=`{}[];'\":,./<>?|\\ äöüß"
    instrument.label = random_ascii
    assert instrument.label == random_ascii

    label = "Nicely-formatted label"
    instrument = cls(name="name1", label=label)
    if isinstance(cls, type(Instrument)):
        request.addfinalizer(instrument.close)
    assert instrument.label == label


@pytest.mark.usefixtures("automatic_gc_disabled")
def test_instrument_without_ref_is_gced() -> None:
    # When there are no active references to an instrument and it is
    # gced there should be no live references to the instrument

    def use_some_instrument() -> None:
        _ = Instrument("SomeInstrument")
        assert list(Instrument._all_instruments.keys()) == ["SomeInstrument"]

    assert len(Instrument._all_instruments) == 0
    use_some_instrument()
    with pytest.warns(ResourceWarning, match="unclosed instrument"):
        gc.collect()
    assert len(Instrument._all_instruments) == 0


def _instrument_class_recording_close(
    closed: list[str],
    *,
    fail_to_initialize: bool = False,
    created: list[weakref.ref[Instrument]] | None = None,
) -> type[Instrument]:
    """
    Create an instrument class that records the name of each instrument that
    is closed in ``closed``, also when the instrument has been stripped by an
    earlier close. If ``fail_to_initialize`` is True, initializing an
    instrument of the class raises a ``RuntimeError``. If ``created`` is given,
    weak references to the created instruments are appended to it.
    """

    class InstrumentRecordingClose(Instrument):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            if created is not None:
                created.append(weakref.ref(self))
            if fail_to_initialize:
                raise RuntimeError("Failed to initialize")

        def close(self) -> None:
            closed.append(self.name)
            super().close()

    return InstrumentRecordingClose


@pytest.mark.usefixtures("close_before_and_after", "automatic_gc_disabled")
def test_garbage_collecting_unclosed_instrument_closes_it_and_warns() -> None:
    closed: list[str] = []
    _instrument_class_recording_close(closed)("unclosed")

    with pytest.warns(
        ResourceWarning,
        match=r"unclosed instrument <InstrumentRecordingClose: unclosed>\. "
        r"Close instruments explicitly",
    ):
        gc.collect()

    assert closed == ["unclosed"]
    assert not Instrument.exist("unclosed")


@pytest.mark.usefixtures("close_before_and_after", "automatic_gc_disabled")
def test_instrument_that_fails_to_initialize_is_closed_right_away() -> None:
    """
    An instrument whose initialization failed is closed right away, so that
    its resources are freed, rather than when it is garbage collected. It is
    neither closed again nor warned about when garbage collected, as it was
    never handed to the user who could therefore not have closed it.
    """
    closed: list[str] = []
    created: list[weakref.ref[Instrument]] = []
    failing_class = _instrument_class_recording_close(
        closed, fail_to_initialize=True, created=created
    )

    with pytest.raises(RuntimeError, match="Failed to initialize"):
        failing_class("failing")

    assert closed == ["failing"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gc.collect()

    (instrument,) = created
    assert instrument() is None
    assert closed == ["failing"]
    assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []


@pytest.mark.usefixtures("close_before_and_after", "automatic_gc_disabled")
def test_failure_to_close_instrument_after_failed_initialization_is_noted() -> None:
    """
    If closing an instrument after its initialization failed fails too, the
    error of the initialization is raised with a note about the failure to
    close it, and closing is not retried when it is garbage collected.
    """
    close_attempts: list[str] = []
    created: list[weakref.ref[Instrument]] = []

    class FailingInstrument(Instrument):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            created.append(weakref.ref(self))
            raise RuntimeError("Failed to initialize")

        def close(self) -> None:
            close_attempts.append(self.name)
            raise OSError("Failed to close")

    with pytest.raises(RuntimeError, match="Failed to initialize") as exc_info:
        FailingInstrument("failing")

    assert exc_info.value.__notes__ == [
        (
            "Closing the FailingInstrument instrument after its failed "
            "initialization also failed: OSError('Failed to close')"
        )
    ]
    # the traceback of the error refers to the instrument
    del exc_info
    gc.collect()

    (instrument,) = created
    assert instrument() is None
    assert close_attempts == ["failing"]


@pytest.mark.usefixtures("close_before_and_after")
def test_instrument_whose_init_returns_a_value_raises_and_is_closed() -> None:
    closed: list[str] = []

    class ReturningInstrument(Instrument):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            # deliberately returns a value, which __init__ must not
            return 1  # type: ignore[return-value]  # noqa: PLE0101

        def close(self) -> None:
            closed.append(self.name)
            super().close()

    with pytest.raises(TypeError, match=r"__init__\(\) should return None, not 'int'"):
        ReturningInstrument("returning")

    assert closed == ["returning"]
    assert not Instrument.exist("returning")


def test_instrument_whose_new_returns_another_object_returns_it() -> None:
    other = object()

    class OtherInstrument(Instrument):
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            return other

    assert OtherInstrument("other") is other


@pytest.mark.usefixtures("close_before_and_after")
def test_instrument_with_name_in_use_is_closed_right_away() -> None:
    """
    An instrument that cannot be registered because its name is in use is
    closed right away, so that its resources are freed, without unregistering
    the instrument that uses the name.
    """
    closed: list[str] = []
    instrument_class = _instrument_class_recording_close(closed)
    existing_instrument = instrument_class("instr")

    with pytest.raises(KeyError, match="Another instrument has the name: instr"):
        instrument_class("instr")

    assert closed == ["instr"]
    assert Instrument.find_instrument("instr") is existing_instrument
    existing_instrument.close()


@pytest.mark.usefixtures("close_before_and_after")
def test_failure_to_close_instrument_with_abstract_parameter_is_noted() -> None:
    class AbstractInstrument(Instrument):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self.add_parameter("voltage", abstract=True)

        def close(self) -> None:
            raise OSError("Failed to close")

    with pytest.raises(NotImplementedError, match="Abstract Parameter") as exc_info:
        AbstractInstrument("abstract")

    assert exc_info.value.__notes__ == [
        (
            "Closing the AbstractInstrument instrument after its failed "
            "initialization also failed: OSError('Failed to close')"
        )
    ]


@pytest.mark.usefixtures("close_before_and_after", "automatic_gc_disabled")
def test_instrument_stays_closed_by_garbage_collector_while_being_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Stripping the attributes of an instrument that the garbage collector is
    closing keeps its state, so that its loggers stay disabled, e.g. if
    deleting an attribute runs code that logs.
    """
    observed: list[bool] = []

    def strip_attrs_and_observe(obj: object, whitelist: Sequence[str] = ()) -> None:
        strip_attrs(obj, whitelist=whitelist)
        if isinstance(obj, Instrument):
            observed.append(obj._being_closed_by_garbage_collector)

    monkeypatch.setattr(instrument_module, "strip_attrs", strip_attrs_and_observe)
    Instrument("unclosed")

    with pytest.warns(ResourceWarning, match="unclosed instrument"):
        gc.collect()

    assert observed == [True]


@pytest.mark.usefixtures("close_before_and_after")
def test_closing_ip_instrument_again_does_not_raise() -> None:
    # not persistent, so that it does not connect
    instrument = IPInstrument(
        "ip_instrument", address="127.0.0.1", port=1, persistent=False
    )

    instrument.close()
    instrument.close()

    assert not Instrument.exist("ip_instrument")


@pytest.mark.usefixtures("close_before_and_after")
def test_closing_ip_instrument_closes_socket_that_is_not_connected() -> None:
    """
    Closing an IPInstrument closes its socket also if the socket is not
    connected, e.g. because connecting it failed, so that it cannot be shut
    down.
    """
    instrument = IPInstrument(
        "ip_instrument", address="127.0.0.1", port=1, persistent=False
    )
    unconnected_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    instrument._socket = unconnected_socket

    instrument.close()

    assert unconnected_socket.fileno() == -1
    assert not Instrument.exist("ip_instrument")


@pytest.mark.usefixtures("close_before_and_after", "automatic_gc_disabled")
def test_failure_to_close_unclosed_instrument_when_garbage_collected_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    If closing an instrument that the garbage collector collects fails, the
    error does not escape the finalizer, which would report it as an exception
    ignored in __del__.
    """
    close_attempts: list[str] = []

    class FailingToCloseInstrument(Instrument):
        def close(self) -> None:
            close_attempts.append(self.name)
            raise OSError("Failed to close")

    unraisable: list[Any] = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    FailingToCloseInstrument("failing")

    with pytest.warns(ResourceWarning, match="unclosed instrument"):
        gc.collect()

    assert close_attempts == ["failing"]
    assert unraisable == []


@pytest.mark.usefixtures("close_before_and_after")
def test_freeing_closed_instrument_after_module_globals_are_cleared() -> None:
    """
    Late in the interpreter shutdown, the globals of modules may have been
    cleared when a closed instrument is freed, which must not make its
    finalizer fail.
    """
    instrument = Instrument("instr")
    instrument.close()
    instrument_ref = weakref.ref(instrument)
    unraisable: list[Any] = []

    with pytest.MonkeyPatch.context() as monkeypatch:
        # simulate the interpreter shutdown clearing the globals of the module
        monkeypatch.setattr(instrument_module, "_InstrumentState", None)
        monkeypatch.setattr(instrument_module, "warnings", None)
        monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
        del instrument
        gc.collect()
        assert instrument_ref() is None

    assert unraisable == []


@pytest.mark.usefixtures("close_before_and_after")
def test_garbage_collecting_closed_instrument_does_not_close_it_again() -> None:
    """
    An instrument that has been closed explicitly is neither closed again nor
    warned about when garbage collected, which may happen after its name has
    been reused.
    """
    closed: list[str] = []
    instrument_class = _instrument_class_recording_close(closed)
    old_instrument = instrument_class("instr")
    old_instrument.close()
    new_instrument = instrument_class("instr")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        del old_instrument
        gc.collect()

    assert closed == ["instr"]
    assert [w for w in caught if issubclass(w.category, ResourceWarning)] == []
    assert Instrument.find_instrument("instr") is new_instrument
    new_instrument.close()


class _InstrumentLoggingWhenClosed(Instrument):
    """An instrument that logs through itself and its channel when closed."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.channel = self.add_submodule("channel", InstrumentChannel(self, "channel"))

    def close(self) -> None:
        self.log.info("closing instrument")
        self.channel.log.info("closing channel")
        super().close()


def _close_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if "closing" in record.getMessage()
    ]


@pytest.mark.usefixtures("close_before_and_after")
def test_instrument_logs_when_closed_explicitly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    instrument = _InstrumentLoggingWhenClosed("logging")

    with caplog.at_level(logging.INFO):
        instrument.close()

    assert _close_messages(caplog) == [
        "[logging(_InstrumentLoggingWhenClosed)] closing instrument",
        "[logging_channel(InstrumentChannel)] closing channel",
    ]


@pytest.mark.usefixtures("close_before_and_after", "automatic_gc_disabled")
def test_instrument_does_not_log_when_closed_by_garbage_collector(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    The garbage collector runs at an arbitrary point in time and on any
    thread, possibly in the middle of other logging, so the loggers of an
    instrument and of its channels do not log while it closes the instrument.
    """
    _InstrumentLoggingWhenClosed("logging")

    with (
        caplog.at_level(logging.INFO),
        pytest.warns(ResourceWarning, match="unclosed instrument"),
    ):
        gc.collect()

    assert not Instrument.exist("logging")
    assert _close_messages(caplog) == []


def test_snapshot_and_meta_attrs() -> None:
    """Test snapshot of InstrumentBase contains _meta_attrs attributes"""
    instr = InstrumentBase("instr", label="Label")

    assert instr.name == "instr"

    assert hasattr(instr, "_meta_attrs")
    assert instr._meta_attrs == ["name", "label"]

    snapshot = instr.snapshot()

    assert "name" in snapshot
    assert "instr" == snapshot["name"]

    assert "label" in snapshot
    assert "Label" == snapshot["label"]

    assert "__class__" in snapshot
    assert "InstrumentBase" in snapshot["__class__"]


class TestSnapshotType(Metadatable):
    __test__ = False

    def __init__(self, sample_value: int) -> None:
        super().__init__()
        self.sample_value = sample_value

    def snapshot_base(
        self,
        update: bool | SnapshotUpdate | None = True,
        params_to_skip_update: Sequence[str] | None = None,
    ) -> Snapshot:
        return {"sample_key": self.sample_value}


class TestInstrument(InstrumentBase):
    __test__ = False

    def __init__(self, name: str, label: str) -> None:
        super().__init__(name, label=label)
        self._meta_attrs.extend(["test_attribute"])
        self._test_attribute = TestSnapshotType(12)

    @property
    def test_attribute(self) -> TestSnapshotType:
        return self._test_attribute


def test_snapshot_and_meta_attrs2() -> None:
    """Test snapshot of child of InstrumentBase which contains
    _meta_attrs attribute that is itself Metadatable"""
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
