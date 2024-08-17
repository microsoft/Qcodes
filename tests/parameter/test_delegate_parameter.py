"""
Test suite for DelegateParameter
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import hypothesis.strategies as hst
import pytest
from hypothesis import given

import qcodes.validators as vals
from qcodes.parameters import DelegateParameter, Parameter, ParamRawDataType

from .conftest import BetterGettableParam

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


@pytest.fixture(name="numeric_val")
def _make_numeric_val() -> Generator[int, None, None]:
    yield 1


@pytest.fixture(name="simple_param")
def _make_simple_param(numeric_val: int) -> Generator[Parameter, None, None]:
    yield Parameter(
        "testparam",
        set_cmd=None,
        get_cmd=None,
        scale=2,
        offset=17,
        label="Test Parameter",
        unit="V",
        initial_value=numeric_val,
    )


class ObservableParam(Parameter):
    def __init__(self, *args: Any, **kwargs: Any):
        self.instr_val = None
        super().__init__(*args, **kwargs)

    def set_raw(self, value: ParamRawDataType) -> None:
        self.instr_val = value

    def get_raw(self) -> ParamRawDataType:
        return self.instr_val

    def get_instr_val(self) -> ParamRawDataType:
        return self.instr_val


@pytest.fixture(params=[True, False])
def make_observable_parameter(
    request: pytest.FixtureRequest,
) -> Generator[Callable[..., ObservableParam], None, None]:
    def make_parameter(
        *args: Any, override_getset: bool = True, **kwargs: Any
    ) -> ObservableParam:
        if request.param:
            if not override_getset:
                pytest.skip()
            param = ObservableParam(*args, **kwargs)
        else:
            val = None

            def set_cmd(value: Any) -> None:
                nonlocal val
                val = value

            def get_cmd() -> Any:
                nonlocal val
                return val

            p = Parameter(
                *args,
                **kwargs,  # type: ignore[misc]
                set_cmd=set_cmd,
                get_cmd=get_cmd,
            )
            param = cast(ObservableParam, p)
            param.get_instr_val = get_cmd  # type: ignore[method-assign]
        return param

    yield make_parameter


def test_observable_parameter(
    make_observable_parameter: Callable[..., ObservableParam], numeric_val: int
) -> None:
    p = make_observable_parameter("testparam")
    p(numeric_val)
    assert p.get_instr_val() == numeric_val


def test_observable_parameter_initial_value(
    make_observable_parameter: Callable[..., ObservableParam], numeric_val: int
) -> None:
    t = make_observable_parameter("observable_parameter", initial_value=numeric_val)
    assert t.get_instr_val() == numeric_val


def test_same_value(simple_param: Parameter) -> None:
    d = DelegateParameter("test_delegate_parameter", simple_param)
    assert d() == simple_param()


def test_same_label_and_unit_on_init(simple_param: Parameter) -> None:
    """
    Test that the label and unit get used from source parameter if not
    specified otherwise.
    """
    d = DelegateParameter("test_delegate_parameter", simple_param)
    assert d.label == simple_param.label
    assert d.unit == simple_param.unit


def test_overwritten_unit_on_init(simple_param: Parameter) -> None:
    d = DelegateParameter("test_delegate_parameter", simple_param, unit="Ohm")
    assert d.label == simple_param.label
    assert not d.unit == simple_param.unit
    assert d.unit == "Ohm"


def test_overwritten_label_on_init(simple_param: Parameter) -> None:
    d = DelegateParameter(
        "test_delegate_parameter", simple_param, label="Physical parameter"
    )
    assert d.unit == simple_param.unit
    assert not d.label == simple_param.label
    assert d.label == "Physical parameter"


def test_get_set_raises(simple_param: Parameter) -> None:
    """
    Test that providing a get/set_cmd kwarg raises an error.
    """
    for kwargs in ({"set_cmd": None}, {"get_cmd": None}):
        with pytest.raises(KeyError) as e:
            DelegateParameter("test_delegate_parameter", simple_param, **kwargs)
        assert str(e.value).startswith("'It is not allowed to set")


def test_scaling(simple_param: Parameter, numeric_val: int) -> None:
    scale = 5
    offset = 3
    d = DelegateParameter(
        "test_delegate_parameter", simple_param, offset=offset, scale=scale
    )

    simple_param(numeric_val)
    assert d() == (numeric_val - offset) / scale

    d(numeric_val)
    assert simple_param() == numeric_val * scale + offset


def test_scaling_delegate_initial_value(
    simple_param: Parameter, numeric_val: int
) -> None:
    scale = 5
    offset = 3
    DelegateParameter(
        "test_delegate_parameter",
        simple_param,
        offset=offset,
        scale=scale,
        initial_value=numeric_val,
    )

    assert simple_param() == numeric_val * scale + offset


def test_scaling_initial_value(simple_param: Parameter) -> None:
    scale = 5
    offset = 3
    d = DelegateParameter(
        "test_delegate_parameter", simple_param, offset=offset, scale=scale
    )
    assert d() == (simple_param() - offset) / scale


def test_snapshot() -> None:
    p = Parameter(
        "testparam", set_cmd=None, get_cmd=None, offset=1, scale=2, initial_value=1
    )
    d = DelegateParameter(
        "test_delegate_parameter", p, offset=3, scale=5, initial_value=2
    )

    delegate_snapshot = d.snapshot()
    source_snapshot = delegate_snapshot.pop("source_parameter")
    assert source_snapshot == p.snapshot()
    assert delegate_snapshot["value"] == 2
    assert source_snapshot["value"] == 13


def test_set_source_cache_changes_delegate_cache(simple_param: Parameter) -> None:
    """Setting the cached value of the source parameter changes the
    delegate parameter cache accordingly.

    """
    offset = 4
    scale = 5
    d = DelegateParameter("d", simple_param, offset=offset, scale=scale)
    new_source_value = 3
    simple_param.cache.set(new_source_value)

    assert d.cache.get() == (new_source_value - offset) / scale


def test_set_source_cache_changes_delegate_get(simple_param: Parameter) -> None:
    """When the delegate parameter's ``get`` is called, the new
    value of the source propagates.

    """
    offset = 4
    scale = 5
    d = DelegateParameter("d", simple_param, offset=offset, scale=scale)
    new_source_value = 3

    simple_param.cache.set(new_source_value)

    assert d.get() == (new_source_value - offset) / scale


def test_set_delegate_cache_changes_source_cache(simple_param: Parameter) -> None:
    offset = 4
    scale = 5
    d = DelegateParameter("d", simple_param, offset=offset, scale=scale)

    new_delegate_value = 2
    d.cache.set(new_delegate_value)

    assert simple_param.cache.get() == (new_delegate_value * scale + offset)


def test_set_delegate_cache_with_raw_value(simple_param: Parameter) -> None:
    offset = 4
    scale = 5
    d = DelegateParameter("d", simple_param, offset=offset, scale=scale)

    new_delegate_value = 2
    d.cache._set_from_raw_value(new_delegate_value * scale + offset)

    assert simple_param.cache.get() == (new_delegate_value * scale + offset)
    assert d.cache.get(get_if_invalid=False) == new_delegate_value


def test_instrument_val_invariant_under_delegate_cache_set(
    make_observable_parameter: Callable[..., ObservableParam], numeric_val: int
) -> None:
    """
    Setting the cached value of the source parameter changes the delegate
    parameter. But it has no impact on the instrument value.
    """
    initial_value = numeric_val
    t = make_observable_parameter("observable_parameter", initial_value=initial_value)
    new_source_value = 3
    t.cache.set(new_source_value)
    assert t.get_instr_val() == initial_value


def test_delegate_cache_pristine_if_not_set() -> None:
    p = Parameter("test")
    d = DelegateParameter("delegate", p)
    gotten_delegate_cache = d.cache.get(get_if_invalid=False)
    assert gotten_delegate_cache is None


def test_delegate_get_updates_cache(
    make_observable_parameter: Callable[..., ObservableParam], numeric_val: int
) -> None:
    initial_value = numeric_val
    t = make_observable_parameter("observable_parameter", initial_value=initial_value)
    d = DelegateParameter("delegate", t)

    assert d() == initial_value
    assert d.cache.get() == initial_value
    assert t.get_instr_val() == initial_value


def test_delegate_parameter_get_and_snapshot_with_none_source() -> None:
    """
    Test that a delegate parameter returns None on get and snapshot if
    the source has a value of None and an offset or scale is used.
    And returns a value if the source is remapped to a real parameter.
    """
    none_param = Parameter("None")
    source_param = Parameter("source", get_cmd=None, set_cmd=None, initial_value=2)
    delegate_param = DelegateParameter(name="delegate", source=none_param)
    delegate_param.offset = 4
    assert delegate_param.get() is None
    assert delegate_param.snapshot()["value"] is None

    delegate_param.offset = None
    delegate_param.scale = 2
    assert delegate_param.get() is None
    assert delegate_param.snapshot()["value"] is None

    parameter = delegate_param.cache._parameter  # type: ignore[attr-defined]
    assert parameter.source.cache is none_param.cache
    delegate_param.source = source_param
    assert delegate_param.get() == 1
    assert delegate_param.snapshot()["value"] == 1
    assert parameter.source.cache is source_param.cache


def test_raw_value_scaling() -> None:
    """
    The :attr:`raw_value` will be deprecated soon,
    so other tests should not use it.
    """

    p = Parameter("testparam", set_cmd=None, get_cmd=None, offset=1, scale=2)
    d = DelegateParameter("test_delegate_parameter", p, offset=3, scale=5)

    val = 1
    p(val)
    assert d() == (val - 3) / 5

    d(val)
    assert d.raw_value == val * 5 + 3
    assert d.raw_value == p()


def test_setting_initial_value_delegate_parameter() -> None:
    value = 10
    p = Parameter("testparam", set_cmd=None, get_cmd=None)
    d = DelegateParameter("test_delegate_parameter", p, initial_value=value)
    assert p.cache.get(get_if_invalid=False) == value
    assert d.cache.get(get_if_invalid=False) == value


def test_setting_initial_cache_delegate_parameter() -> None:
    value = 10
    p = Parameter("testparam", set_cmd=None, get_cmd=None)
    d = DelegateParameter("test_delegate_parameter", p, initial_cache_value=value)
    assert p.cache.get(get_if_invalid=False) == value
    assert d.cache.get(get_if_invalid=False) == value


def test_delegate_parameter_with_none_source_works_as_expected() -> None:
    delegate_param = DelegateParameter(name="delegate", source=None, scale=2, offset=1)
    _assert_none_source_is_correct(delegate_param)


@given(
    hst.floats(allow_nan=False, allow_infinity=False),
    hst.floats(allow_nan=False, allow_infinity=False).filter(lambda x: x != 0),
    hst.floats(allow_nan=False, allow_infinity=False),
)
def test_delegate_parameter_with_changed_source_snapshot_matches_value(
    value: float, scale: float, offset: float
) -> None:
    delegate_param = DelegateParameter(
        name="delegate", source=None, scale=scale, offset=offset
    )
    source_parameter = Parameter(
        name="source", get_cmd=None, set_cmd=None, initial_value=value
    )
    _assert_none_source_is_correct(delegate_param)
    delegate_param.source = source_parameter
    calc_value = (value - offset) / scale
    assert delegate_param.cache.get(get_if_invalid=False) == calc_value
    assert delegate_param.source.cache.get(get_if_invalid=False) == value
    snapshot = delegate_param.snapshot()
    # disregard timestamp that might be slightly different
    snapshot["source_parameter"].pop("ts")
    source_snapshot = source_parameter.snapshot()
    source_snapshot.pop("ts")
    assert snapshot["source_parameter"] == source_snapshot
    assert snapshot["value"] == calc_value
    assert delegate_param.get() == calc_value
    # now remove the source again
    delegate_param.source = None
    _assert_none_source_is_correct(delegate_param)
    _assert_delegate_cache_none_source(delegate_param)


def _assert_none_source_is_correct(delegate_param: DelegateParameter) -> None:
    with pytest.raises(TypeError):
        delegate_param.get()
    with pytest.raises(TypeError):
        delegate_param.set(1)
    snapshot = delegate_param.snapshot()
    assert snapshot["source_parameter"] is None
    assert "value" not in snapshot.keys()
    snapshot.pop("ts")
    updated_snapshot = delegate_param.snapshot(update=True)
    updated_snapshot.pop("ts")
    assert snapshot == updated_snapshot


def _assert_delegate_cache_none_source(delegate_param: DelegateParameter) -> None:
    with pytest.raises(TypeError):
        delegate_param.cache.set(1)
    with pytest.raises(TypeError):
        delegate_param.cache.get()
    with pytest.raises(TypeError):
        delegate_param.cache.raw_value
    assert delegate_param.cache.max_val_age is None
    assert delegate_param.cache.timestamp is None


@pytest.mark.parametrize("snapshot_value", [True, False])
@pytest.mark.parametrize("gettable,get_cmd", [(True, None), (False, False)])
@pytest.mark.parametrize("settable,set_cmd", [(True, None), (False, False)])
def test_gettable_settable_snapshotget_delegate_parameter(
    gettable: bool,
    get_cmd: Literal[False] | None,
    settable: bool,
    set_cmd: Literal[False] | None,
    snapshot_value: bool,
) -> None:
    """
    Test that gettable, settable and snapshot_get are correctly reflected
    in the DelegateParameter
    """
    source_param = Parameter(
        "source", get_cmd=get_cmd, set_cmd=set_cmd, snapshot_value=snapshot_value
    )
    delegate_param = DelegateParameter("delegate", source=source_param)
    assert delegate_param.gettable is gettable
    assert delegate_param.settable is settable
    assert delegate_param._snapshot_value is snapshot_value


@pytest.mark.parametrize("snapshot_value", [True, False])
@pytest.mark.parametrize("gettable,get_cmd", [(True, None), (False, False)])
@pytest.mark.parametrize("settable,set_cmd", [(True, None), (False, False)])
def test_gettable_settable_snapshotget_delegate_parameter_2(
    gettable: bool,
    get_cmd: Literal[False] | None,
    settable: bool,
    set_cmd: Literal[False] | None,
    snapshot_value: bool,
) -> None:
    """
    Test that gettable/settable and snapshot_get are updated correctly
    when source changes
    """
    source_param = Parameter(
        "source", get_cmd=get_cmd, set_cmd=set_cmd, snapshot_value=snapshot_value
    )
    delegate_param = DelegateParameter("delegate", source=None)
    delegate_param.source = source_param
    assert delegate_param.gettable is gettable
    assert delegate_param.settable is settable
    assert delegate_param._snapshot_value is snapshot_value


def test_initial_value_and_none_source_raises() -> None:
    with pytest.raises(
        KeyError,
        match="It is not allowed to supply"
        " 'initial_value' or"
        " 'initial_cache_value'",
    ):
        DelegateParameter("delegate", source=None, initial_value=1)
    with pytest.raises(
        KeyError,
        match="It is not allowed to supply"
        " 'initial_value' or "
        "'initial_cache_value'",
    ):
        DelegateParameter("delegate", source=None, initial_cache_value=1)


def test_delegate_parameter_change_source_reflected_in_label_and_unit() -> None:
    delegate_param = DelegateParameter("delegate", source=None)
    source_param_1 = Parameter("source1", label="source 1", unit="unit1")
    source_param_2 = Parameter("source2", label="source 2", unit="unit2")

    assert delegate_param.label == "delegate"
    assert delegate_param.unit == ""
    delegate_param.source = source_param_1
    assert delegate_param.label == "source 1"
    assert delegate_param.unit == "unit1"
    delegate_param.source = source_param_2
    assert delegate_param.label == "source 2"
    assert delegate_param.unit == "unit2"
    delegate_param.source = None
    assert delegate_param.label == "delegate"
    assert delegate_param.unit == ""


def test_delegate_parameter_fixed_label_unit_unchanged() -> None:
    delegate_param = DelegateParameter(
        "delegate", label="delegatelabel", unit="delegateunit", source=None
    )
    source_param_1 = Parameter("source1", label="source 1", unit="unit1")
    source_param_2 = Parameter("source2", label="source 2", unit="unit2")

    assert delegate_param.label == "delegatelabel"
    assert delegate_param.unit == "delegateunit"
    delegate_param.source = source_param_1
    assert delegate_param.label == "delegatelabel"
    assert delegate_param.unit == "delegateunit"
    delegate_param.source = source_param_2
    assert delegate_param.label == "delegatelabel"
    assert delegate_param.unit == "delegateunit"
    delegate_param.source = None
    assert delegate_param.label == "delegatelabel"
    assert delegate_param.unit == "delegateunit"


def test_cache_invalidation() -> None:
    value = 10
    p = BetterGettableParam("testparam", set_cmd=None, get_cmd=None)
    d = DelegateParameter("test_delegate_parameter", p, initial_cache_value=value)
    assert p._get_count == 0
    assert d.cache.get() == value
    assert p._get_count == 0

    assert d.cache.valid is True
    assert p.cache.valid is True

    d.cache.invalidate()

    assert d.cache.valid is False
    assert p.cache.valid is False

    d.cache.get()
    assert p._get_count == 1

    assert d.cache.valid is True
    assert p.cache.valid is True


def test_cache_no_source() -> None:
    d = DelegateParameter("test_delegate_parameter", source=None)

    assert d.cache.valid is False
    assert d.cache.timestamp is None
    assert d.cache.max_val_age is None

    with pytest.raises(
        TypeError,
        match="Cannot get the cache of a DelegateParameter that delegates to None",
    ):
        d.cache.get()

    d.cache.invalidate()


def test_underlying_instrument_property_for_delegate_parameter() -> None:
    p = BetterGettableParam("testparam", set_cmd=None, get_cmd=None)
    d = DelegateParameter("delegate_parameter_with_source", p)

    assert d.underlying_instrument is p.root_instrument

    d = DelegateParameter("delegate_parameter_without_source", source=None)
    assert d.underlying_instrument is None


def test_value_validation() -> None:
    source_param = Parameter("source", set_cmd=None, get_cmd=None)
    delegate_param = DelegateParameter("delegate", source=source_param)

    delegate_param.vals = vals.Numbers(-10, 10)
    source_param.vals = None
    delegate_param.validate(1)
    with pytest.raises(ValueError):
        delegate_param.validate(11)

    delegate_param.vals = None
    source_param.vals = vals.Numbers(-5, 5)
    delegate_param.validate(1)
    with pytest.raises(ValueError):
        delegate_param.validate(6)

    delegate_param.vals = vals.Numbers(-10, 10)
    source_param.vals = vals.Numbers(-5, 5)
    delegate_param.validate(1)
    with pytest.raises(ValueError):
        delegate_param.validate(6)
    with pytest.raises(ValueError):
        delegate_param.validate(11)


def test_value_validation_with_offset_and_scale() -> None:
    source_param = Parameter(
        "source", set_cmd=None, get_cmd=None, vals=vals.Numbers(-5, 5)
    )
    delegate_param = DelegateParameter(
        "delegate", source=source_param, vals=vals.Numbers(-10, 10)
    )

    source_param.offset = 100
    source_param.scale = None
    source_param.validate(0)  # raw_value = 100
    source_param.set(0)
    delegate_param.validate(0)  # raw_value = 0
    delegate_param.set(0)

    source_param.offset = None
    source_param.scale = 100
    source_param.validate(1)  # raw_value = 100
    source_param.set(1)
    delegate_param.validate(1)  # raw_value = 1
    delegate_param.set(1)

    source_param.offset = None
    source_param.scale = None

    delegate_param.offset = 100
    delegate_param.scale = None
    source_param.validate(0)  # raw_value = 0
    source_param.set(0)
    with pytest.raises(ValueError):
        delegate_param.validate(0)  # raw_value = 100
    with pytest.raises(ValueError):
        delegate_param.set(0)

    delegate_param.offset = None
    delegate_param.scale = 100
    source_param.validate(1)  # raw_value = 1
    source_param.set(1)
    with pytest.raises(ValueError):
        delegate_param.validate(1)  # raw_value = 100
    with pytest.raises(ValueError):
        delegate_param.set(1)
