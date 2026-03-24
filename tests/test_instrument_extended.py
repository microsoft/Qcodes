"""
Extended tests for Instrument and InstrumentBase to improve coverage.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest

from qcodes.instrument import (
    Instrument,
    InstrumentBase,
    InstrumentModule,
    find_or_create_instrument,
)
from qcodes.instrument_drivers.mock_instruments import (
    DummyChannelInstrument,
    DummyInstrument,
    MockMetaParabola,
    MockParabola,
)
from qcodes.parameters import Function

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(name="dummy", scope="function")
def _dummy() -> Iterator[DummyInstrument]:
    inst = DummyInstrument(name="ext_dummy", gates=["dac1", "dac2", "dac3"])
    try:
        yield inst
    finally:
        inst.close()


@pytest.fixture(name="dummy_ch", scope="function")
def _dummy_ch() -> Iterator[DummyChannelInstrument]:
    inst = DummyChannelInstrument(name="ext_dummy_ch")
    try:
        yield inst
    finally:
        inst.close()


# ---------------------------------------------------------------------------
# Instrument.write_raw / ask_raw raise NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_write_raw_raises(dummy: DummyInstrument) -> None:
    """write_raw on bare Instrument should raise NotImplementedError."""
    # DummyBase overrides get_idn but not write_raw/ask_raw, so
    # the base Instrument.write_raw implementation applies.
    bare = Instrument("bare_instr")
    try:
        with pytest.raises(NotImplementedError, match="has not defined a write method"):
            bare.write_raw("cmd")
    finally:
        bare.close()


@pytest.mark.serial
def test_ask_raw_raises() -> None:
    """ask_raw on bare Instrument should raise NotImplementedError."""
    bare = Instrument("bare_ask")
    try:
        with pytest.raises(NotImplementedError, match="has not defined an ask method"):
            bare.ask_raw("cmd")
    finally:
        bare.close()


# ---------------------------------------------------------------------------
# Instrument.write and ask wrap errors
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_write_wraps_exception() -> None:
    """write() should wrap underlying exceptions with context."""
    bare = Instrument("bare_write_wrap")
    try:
        with pytest.raises(NotImplementedError) as exc_info:
            bare.write("SOMECMD")
        assert "writing 'SOMECMD'" in str(exc_info.value.args)
    finally:
        bare.close()


@pytest.mark.serial
def test_ask_wraps_exception() -> None:
    """ask() should wrap underlying exceptions with context."""
    bare = Instrument("bare_ask_wrap")
    try:
        with pytest.raises(NotImplementedError) as exc_info:
            bare.ask("SOMECMD")
        assert "asking 'SOMECMD'" in str(exc_info.value.args)
    finally:
        bare.close()


# ---------------------------------------------------------------------------
# Instrument.close_all
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_close_all() -> None:
    """close_all should remove all registered instruments."""
    _ = DummyInstrument(name="closeall1", gates=["g1"])
    _ = DummyInstrument(name="closeall2", gates=["g2"])

    assert Instrument.exist("closeall1")
    assert Instrument.exist("closeall2")

    Instrument.close_all()

    assert not Instrument.exist("closeall1")
    assert not Instrument.exist("closeall2")


# ---------------------------------------------------------------------------
# Instrument.find_instrument
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_find_instrument_by_name(dummy: DummyInstrument) -> None:
    """find_instrument should return the instrument by name."""
    found = Instrument.find_instrument("ext_dummy")
    assert found is dummy


@pytest.mark.serial
def test_find_instrument_not_found() -> None:
    """find_instrument should raise KeyError for non-existent names."""
    with pytest.raises(KeyError, match="does not exist"):
        Instrument.find_instrument("nonexistent_instrument_xyz")


@pytest.mark.serial
def test_find_instrument_wrong_class(dummy: DummyInstrument) -> None:
    """find_instrument should raise TypeError when class doesn't match."""
    with pytest.raises(TypeError, match="was requested"):
        Instrument.find_instrument("ext_dummy", instrument_class=MockParabola)


@pytest.mark.serial
def test_find_instrument_with_class(dummy: DummyInstrument) -> None:
    """find_instrument with matching class should succeed."""
    found = Instrument.find_instrument("ext_dummy", instrument_class=DummyInstrument)
    assert found is dummy


# ---------------------------------------------------------------------------
# Instrument.exist and is_valid
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_exist_true(dummy: DummyInstrument) -> None:
    assert Instrument.exist("ext_dummy") is True


@pytest.mark.serial
def test_exist_false() -> None:
    assert Instrument.exist("does_not_exist_xyz") is False


@pytest.mark.serial
def test_exist_with_class(dummy: DummyInstrument) -> None:
    assert Instrument.exist("ext_dummy", instrument_class=DummyInstrument) is True
    # exist() with wrong class will raise TypeError (not return False)
    with pytest.raises(TypeError):
        Instrument.exist("ext_dummy", instrument_class=MockParabola)


@pytest.mark.serial
def test_is_valid_open(dummy: DummyInstrument) -> None:
    assert Instrument.is_valid(dummy) is True


@pytest.mark.serial
def test_is_valid_after_close() -> None:
    inst = DummyInstrument(name="valid_test", gates=["g"])
    inst.close()
    assert Instrument.is_valid(inst) is False


# ---------------------------------------------------------------------------
# Instrument.__repr__ and __del__
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_repr(dummy: DummyInstrument) -> None:
    r = repr(dummy)
    assert "DummyInstrument" in r
    assert "ext_dummy" in r


@pytest.mark.serial
def test_del_closes_instrument() -> None:
    """__del__ should close the instrument without raising."""
    inst = DummyInstrument(name="del_test", gates=["g"])
    assert Instrument.exist("del_test")
    inst.__del__()
    assert not Instrument.exist("del_test")


# ---------------------------------------------------------------------------
# Instrument.close with connection attribute
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_close_with_connection() -> None:
    """close() should call connection.close() if present."""

    class FakeConnection:
        closed = False

        def close(self) -> None:
            self.closed = True

    inst = DummyInstrument(name="conn_test", gates=["g"])
    conn = FakeConnection()
    inst.connection = conn  # type: ignore[attr-defined]
    inst.close()
    assert conn.closed


# ---------------------------------------------------------------------------
# Instrument.instances / record_instance / remove_instance
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_instances_returns_list(dummy: DummyInstrument) -> None:
    """instances() should return a list containing the instrument."""
    instances = DummyInstrument.instances()
    assert dummy in instances


@pytest.mark.serial
def test_instances_empty_after_close() -> None:
    inst = DummyInstrument(name="inst_empty_test", gates=["g"])
    inst.close()
    assert inst not in DummyInstrument.instances()


@pytest.mark.serial
def test_record_instance_duplicate_name(dummy: DummyInstrument) -> None:
    """Recording an instance with a duplicate name should raise."""
    with pytest.raises(KeyError, match="Another instrument has the name"):
        DummyInstrument(name="ext_dummy", gates=["g"])


# ---------------------------------------------------------------------------
# InstrumentBase.label
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_label_default(dummy: DummyInstrument) -> None:
    """Default label should be the instrument name."""
    assert dummy.label == "ext_dummy"


@pytest.mark.serial
def test_label_set_get(dummy: DummyInstrument) -> None:
    """Label property should be settable and gettable."""
    dummy.label = "My Custom Label"
    assert dummy.label == "My Custom Label"


@pytest.mark.serial
def test_label_via_constructor() -> None:
    """Label kwarg should be respected."""
    inst = DummyInstrument(name="label_test", gates=["g"], label="Custom")
    try:
        assert inst.label == "Custom"
    finally:
        inst.close()


# ---------------------------------------------------------------------------
# InstrumentBase.add_function
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_add_function(dummy: DummyInstrument) -> None:
    dummy.add_function("rst", call_cmd="*RST")
    assert "rst" in dummy.functions
    assert isinstance(dummy.functions["rst"], Function)


@pytest.mark.serial
def test_add_function_duplicate(dummy: DummyInstrument) -> None:
    dummy.add_function("rst2", call_cmd="*RST")
    with pytest.raises(KeyError, match="Duplicate function name"):
        dummy.add_function("rst2", call_cmd="*RST")


# ---------------------------------------------------------------------------
# InstrumentBase.add_submodule
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_add_submodule(dummy: DummyInstrument) -> None:
    mod = InstrumentModule(dummy, "mymod")
    result = dummy.add_submodule("mymod", mod)
    assert result is mod
    assert "mymod" in dummy.submodules
    assert "mymod" in dummy.instrument_modules


@pytest.mark.serial
def test_add_submodule_duplicate(dummy: DummyInstrument) -> None:
    mod1 = InstrumentModule(dummy, "dupmod")
    dummy.add_submodule("dupmod", mod1)
    mod2 = InstrumentModule(dummy, "dupmod2")
    with pytest.raises(KeyError, match="Duplicate submodule name"):
        dummy.add_submodule("dupmod", mod2)


@pytest.mark.serial
def test_add_submodule_non_metadatable(dummy: DummyInstrument) -> None:
    with pytest.raises(TypeError, match="Submodules must be metadatable"):
        dummy.add_submodule("bad", "not_a_submodule")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# InstrumentBase.get_component / _get_component_by_name
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_get_component_parameter(dummy: DummyInstrument) -> None:
    comp = dummy.get_component("dac1")
    assert comp is dummy.parameters["dac1"]


@pytest.mark.serial
def test_get_component_submodule(dummy_ch: DummyChannelInstrument) -> None:
    comp = dummy_ch.get_component("A")
    assert comp is dummy_ch.submodules["A"]


@pytest.mark.serial
def test_get_component_nested(dummy_ch: DummyChannelInstrument) -> None:
    """Get a parameter within a submodule."""
    comp = dummy_ch.get_component("A_temperature")
    assert comp is dummy_ch.submodules["A"].parameters["temperature"]  # type: ignore[union-attr]


@pytest.mark.serial
def test_get_component_not_found(dummy: DummyInstrument) -> None:
    with pytest.raises(KeyError):
        dummy.get_component("nonexistent_thing")


# ---------------------------------------------------------------------------
# InstrumentBase.print_readable_snapshot
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_print_readable_snapshot(
    dummy: DummyInstrument, capsys: pytest.CaptureFixture[str]
) -> None:
    dummy.print_readable_snapshot(update=False)
    captured = capsys.readouterr()
    assert "ext_dummy:" in captured.out
    assert "dac1" in captured.out


@pytest.mark.serial
def test_print_readable_snapshot_truncation(
    dummy: DummyInstrument, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that long lines are truncated to max_chars."""
    dummy.print_readable_snapshot(update=False, max_chars=40)
    captured = capsys.readouterr()
    for line in captured.out.split("\n"):
        if line.startswith("-"):
            continue
        if line.strip() == "":
            continue
        # header lines and parameter lines
        # Lines should be at most 40 chars (or contain "...")
        if len(line) > 40:
            assert line.endswith("...")


# ---------------------------------------------------------------------------
# InstrumentBase.invalidate_cache
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_invalidate_cache(dummy: DummyInstrument) -> None:
    """invalidate_cache should mark parameters as stale."""
    dummy.dac1.set(42)
    dummy.dac1.get()
    assert dummy.dac1.cache.valid
    dummy.invalidate_cache()
    assert not dummy.dac1.cache.valid


@pytest.mark.serial
def test_invalidate_cache_with_submodules(
    dummy_ch: DummyChannelInstrument,
) -> None:
    """invalidate_cache should recurse into submodules."""
    chan_a = dummy_ch.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    chan_a.parameters["temperature"].set(100)
    chan_a.parameters["temperature"].get()
    assert chan_a.parameters["temperature"].cache.valid
    dummy_ch.invalidate_cache()
    assert not chan_a.parameters["temperature"].cache.valid


# ---------------------------------------------------------------------------
# InstrumentBase.parent / ancestors / root_instrument
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_parent_of_instrument(dummy: DummyInstrument) -> None:
    """Top-level instruments should have parent=None."""
    assert dummy.parent is None


@pytest.mark.serial
def test_parent_of_module(dummy_ch: DummyChannelInstrument) -> None:
    chan_a = dummy_ch.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    assert chan_a.parent is dummy_ch


@pytest.mark.serial
def test_ancestors_of_instrument(dummy: DummyInstrument) -> None:
    assert dummy.ancestors == (dummy,)


@pytest.mark.serial
def test_ancestors_of_module(dummy_ch: DummyChannelInstrument) -> None:
    chan_a = dummy_ch.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    assert chan_a.ancestors == (chan_a, dummy_ch)


@pytest.mark.serial
def test_root_instrument(dummy: DummyInstrument) -> None:
    assert dummy.root_instrument is dummy


@pytest.mark.serial
def test_root_instrument_of_module(dummy_ch: DummyChannelInstrument) -> None:
    chan_a = dummy_ch.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    assert chan_a.root_instrument is dummy_ch


# ---------------------------------------------------------------------------
# InstrumentBase.name_parts / full_name / short_name
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_name_parts_instrument(dummy: DummyInstrument) -> None:
    assert dummy.name_parts == ["ext_dummy"]


@pytest.mark.serial
def test_name_parts_module(dummy_ch: DummyChannelInstrument) -> None:
    chan_a = dummy_ch.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    assert chan_a.name_parts == ["ext_dummy_ch", "ChanA"]


@pytest.mark.serial
def test_full_name_module(dummy_ch: DummyChannelInstrument) -> None:
    chan_a = dummy_ch.submodules["A"]
    assert isinstance(chan_a, InstrumentModule)
    assert chan_a.full_name == "ext_dummy_ch_ChanA"


@pytest.mark.serial
def test_short_name(dummy: DummyInstrument) -> None:
    assert dummy.short_name == "ext_dummy"


@pytest.mark.serial
def test_name_equals_full_name(dummy: DummyInstrument) -> None:
    assert dummy.name == dummy.full_name


# ---------------------------------------------------------------------------
# InstrumentBase.snapshot_base
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_snapshot_base(dummy: DummyInstrument) -> None:
    snap = dummy.snapshot_base(update=False)
    assert "parameters" in snap
    assert "functions" in snap
    assert "submodules" in snap
    assert "__class__" in snap
    assert "name" in snap
    assert "label" in snap
    assert "dac1" in snap["parameters"]


# ---------------------------------------------------------------------------
# InstrumentBase.validate_status
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_validate_status(
    dummy: DummyInstrument, capsys: pytest.CaptureFixture[str]
) -> None:
    """validate_status should not raise for valid parameters."""
    dummy.dac1.set(10)
    dummy.validate_status(verbose=True)
    captured = capsys.readouterr()
    assert "dac1" in captured.out


# ---------------------------------------------------------------------------
# InstrumentBase._replace_hyphen and _is_valid_identifier
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_replace_hyphen() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        inst = DummyInstrument(name="my-inst", gates=["g"])
        try:
            assert inst.name == "my_inst"
            assert len(w) >= 1
            hyphen_warnings = [x for x in w if "Changed my-inst" in str(x.message)]
            assert len(hyphen_warnings) >= 1
        finally:
            inst.close()


@pytest.mark.serial
def test_invalid_identifier() -> None:
    with pytest.raises(ValueError, match="invalid instrument identifier"):
        DummyInstrument(name="123invalid", gates=["g"])


# ---------------------------------------------------------------------------
# InstrumentBase deprecated __getitem__, set, get, call
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_deprecated_getitem(dummy: DummyInstrument) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        val = dummy["dac1"]  # type: ignore[index]
    assert val is dummy.parameters["dac1"]


@pytest.mark.serial
def test_deprecated_set_get(dummy: DummyInstrument) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        dummy.set("dac1", 42)  # type: ignore[call-overload]
        result = dummy.get("dac1")  # type: ignore[call-overload]
    assert result == 42


@pytest.mark.serial
def test_deprecated_call(dummy: DummyInstrument) -> None:
    dummy.add_function("noop", call_cmd="*OPC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        # call() on a function backed by a string cmd will try to write,
        # which raises NotImplementedError on DummyBase. That's fine —
        # we just test that the deprecated `call` path is exercised.
        try:
            dummy.call("noop")  # type: ignore[call-overload]
        except NotImplementedError:
            pass


# ---------------------------------------------------------------------------
# InstrumentBase.__getstate__ prevents pickling
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_getstate_raises(dummy: DummyInstrument) -> None:
    with pytest.raises(RuntimeError, match="can not be pickled"):
        dummy.__getstate__()


# ---------------------------------------------------------------------------
# find_or_create_instrument
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_find_or_create_new() -> None:
    """find_or_create_instrument should create a new instrument."""
    inst = find_or_create_instrument(DummyInstrument, "foc_new", gates=["g1"])
    try:
        assert isinstance(inst, DummyInstrument)
        assert inst.name == "foc_new"
    finally:
        inst.close()


@pytest.mark.serial
def test_find_or_create_existing() -> None:
    """find_or_create_instrument should find an existing instrument."""
    inst = DummyInstrument(name="foc_exist", gates=["g1"])
    try:
        found = find_or_create_instrument(DummyInstrument, "foc_exist", gates=["g1"])
        assert found is inst
    finally:
        inst.close()


@pytest.mark.serial
def test_find_or_create_recreate() -> None:
    """find_or_create_instrument with recreate=True should recreate."""
    inst = DummyInstrument(name="foc_recreate", gates=["g1"])
    new_inst = find_or_create_instrument(
        DummyInstrument, "foc_recreate", gates=["g1"], recreate=True
    )
    try:
        assert new_inst is not inst
        assert new_inst.name == "foc_recreate"
    finally:
        new_inst.close()


# ---------------------------------------------------------------------------
# MockMetaParabola (InstrumentBase, not Instrument)
# ---------------------------------------------------------------------------


@pytest.mark.serial
def test_meta_instrument_not_tracked() -> None:
    """MockMetaParabola is InstrumentBase but not Instrument, not tracked."""
    p = MockParabola("meta_parabola_parent")
    try:
        m = MockMetaParabola("meta_test", p)
        assert isinstance(m, InstrumentBase)
        assert not isinstance(m, Instrument)
        assert not Instrument.exist("meta_test")
    finally:
        p.close()
