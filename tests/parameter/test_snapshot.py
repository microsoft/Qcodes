from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from typing_extensions import ParamSpec

from qcodes.parameters import Parameter

from .conftest import NOT_PASSED

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
P = ParamSpec("P")


def create_parameter(
    snapshot_get: bool | Literal["NOT_PASSED"],
    snapshot_value: bool | Literal["NOT_PASSED"],
    cache_is_valid: bool,
    get_cmd: Callable[..., Any] | bool | None | Literal["NOT_PASSED"],
    offset: float | Literal["NOT_PASSED"] = NOT_PASSED,
) -> Parameter:
    kwargs: dict[str, Any] = dict(
        set_cmd=None, label="Parameter", unit="a.u.", docstring="some docs"
    )

    if offset != NOT_PASSED:
        kwargs.update(offset=offset)

    if snapshot_get != NOT_PASSED:
        kwargs.update(snapshot_get=snapshot_get)

    if snapshot_value != NOT_PASSED:
        kwargs.update(snapshot_value=snapshot_value)

    if get_cmd != NOT_PASSED:
        kwargs.update(get_cmd=get_cmd)

    p = Parameter("p", **kwargs)

    if get_cmd is not False:

        def wrap_in_call_counter(get_func: Callable[P, T]) -> Callable[P, T]:
            call_count = 0

            def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> T:
                nonlocal call_count
                call_count += 1
                return get_func(*args, **kwargs)

            wrapped_func.call_count = lambda: call_count  # type: ignore[attr-defined]

            return wrapped_func

        p.get = wrap_in_call_counter(p.get)
        # pre-condition
        assert p.get.call_count() == 0  # type: ignore[attr-defined]
    else:
        # pre-condition
        assert not hasattr(p, "get")
        assert not p.gettable

    if cache_is_valid:
        p.set(42)

    return p


def test_snapshot_contains_parameter_attributes(
    snapshot_get: bool | Literal["NOT_PASSED"],
    snapshot_value: bool | Literal["NOT_PASSED"],
    get_cmd: None | Literal[False] | Literal["NOT_PASSED"],
    cache_is_valid: bool,
    update: bool | Literal["NOT_PASSED"] | None,
) -> None:
    p = create_parameter(snapshot_get, snapshot_value, cache_is_valid, get_cmd)

    if update != NOT_PASSED:
        s = p.snapshot(update=update)
    else:
        s = p.snapshot()

    assert isinstance(s, dict)

    # Not metadata key in the snapshot because we didn't pass any metadata
    # for the parameter
    # TODO: test for parameter with metadata
    assert "metadata" not in s

    assert s["__class__"] == "qcodes.parameters.parameter.Parameter"
    assert s["full_name"] == "p"

    # The following is because the parameter does not belong to an instrument
    # TODO: test for a parameter that is attached to instrument
    assert "instrument" not in s
    assert "instrument_name" not in s

    # These attributes have value of ``None`` hence not included in the snapshot
    # TODO: test snapshot when some of these are not None
    none_attrs = ("step", "scale", "offset", "val_mapping", "vals")
    for attr in none_attrs:
        assert getattr(p, attr) is None  # pre-condition
        assert attr not in s

    # TODO: test snapshot when some of these are None
    not_none_attrs = {
        "name": p.name,
        "label": p.label,
        "unit": p.unit,
        "inter_delay": p.inter_delay,
        "post_delay": p.post_delay,
    }
    for attr, value in not_none_attrs.items():
        assert s[attr] == value


def test_snapshot_timestamp_of_non_gettable_depends_only_on_cache_validity(
    snapshot_get: bool | Literal["NOT_PASSED"],
    snapshot_value: bool | Literal["NOT_PASSED"],
    update: bool | Literal["NOT_PASSED"] | None,
    cache_is_valid: bool,
) -> None:
    p = create_parameter(snapshot_get, snapshot_value, cache_is_valid, get_cmd=False)

    t0 = p.cache.timestamp

    if update != NOT_PASSED:
        s = p.snapshot(update=update)
    else:
        s = p.snapshot()

    if cache_is_valid:
        assert t0 is not None
        ts = datetime.strptime(s["ts"], "%Y-%m-%d %H:%M:%S")
        t0_up_to_seconds = t0.replace(microsecond=0)
        assert ts >= t0_up_to_seconds
    else:
        assert t0 is None
        assert s["ts"] is None


def test_snapshot_timestamp_for_valid_cache_depends_on_cache_update(
    snapshot_get: bool | Literal["NOT_PASSED"],
    snapshot_value: bool | Literal["NOT_PASSED"],
    update: bool | Literal["NOT_PASSED"] | None,
) -> None:
    p = create_parameter(
        snapshot_get, snapshot_value, get_cmd=lambda: 69, cache_is_valid=True
    )

    # Hack cache's timestamp to simplify this test
    timestamp = p.cache.timestamp
    assert timestamp is not None
    p.cache._timestamp = timestamp - timedelta(days=31)  # type: ignore[attr-defined]

    tu = datetime.now()
    assert p.cache.timestamp is not None
    assert p.cache.timestamp < tu  # pre-condition
    if update != NOT_PASSED:
        s = p.snapshot(update=update)
    else:
        s = p.snapshot()

    ts = datetime.strptime(s["ts"], "%Y-%m-%d %H:%M:%S")
    tu_up_to_seconds = tu.replace(microsecond=0)

    cache_gets_updated_on_snapshot_call = (
        snapshot_value is not False and snapshot_get is not False and update is True
    )

    if cache_gets_updated_on_snapshot_call:
        assert ts >= tu_up_to_seconds
    else:
        assert ts < tu_up_to_seconds


def test_snapshot_timestamp_for_invalid_cache_depends_only_on_snapshot_flags(
    snapshot_get: bool | Literal["NOT_PASSED"],
    snapshot_value: bool | Literal["NOT_PASSED"],
    update: bool | Literal["NOT_PASSED"] | None,
) -> None:
    p = create_parameter(
        snapshot_get, snapshot_value, get_cmd=lambda: 69, cache_is_valid=False
    )

    cache_gets_updated_on_snapshot_call = (
        snapshot_value is not False
        and snapshot_get is not False
        and update is not False
        and update != NOT_PASSED
    )

    if cache_gets_updated_on_snapshot_call:
        tu = datetime.now()
    else:
        tu = None

    if update != NOT_PASSED:
        s = p.snapshot(update=update)
    else:
        s = p.snapshot()

    if cache_gets_updated_on_snapshot_call:
        ts = datetime.strptime(s["ts"], "%Y-%m-%d %H:%M:%S")
        assert tu is not None
        tu_up_to_seconds = tu.replace(microsecond=0)
        assert ts >= tu_up_to_seconds
    else:
        assert s["ts"] is None


def test_snapshot_when_snapshot_value_is_false(
    snapshot_get: bool | Literal["NOT_PASSED"],
    get_cmd: Literal[False, "NOT_PASSED"] | None,
    cache_is_valid: bool,
    update: bool | Literal["NOT_PASSED"] | None,
) -> None:
    p = create_parameter(
        snapshot_get=snapshot_get,
        snapshot_value=False,
        get_cmd=get_cmd,
        cache_is_valid=cache_is_valid,
    )

    if update != NOT_PASSED:
        s = p.snapshot(update=update)
    else:
        s = p.snapshot()

    assert "value" not in s
    assert "raw_value" not in s

    if get_cmd is not False:
        assert p.get.call_count() == 0  # type: ignore[attr-defined]


def test_snapshot_value_is_true_by_default(
    snapshot_get: bool | Literal["NOT_PASSED"],
    get_cmd: Literal[False, "NOT_PASSED"] | None,
) -> None:
    p = create_parameter(
        snapshot_value=NOT_PASSED,
        snapshot_get=snapshot_get,
        get_cmd=get_cmd,
        cache_is_valid=True,
    )
    assert p._snapshot_value is True


def test_snapshot_get_is_true_by_default(
    snapshot_value: bool | Literal["NOT_PASSED"],
    get_cmd: Literal[False, "NOT_PASSED"] | None,
) -> None:
    p = create_parameter(
        snapshot_get=NOT_PASSED,
        snapshot_value=snapshot_value,
        get_cmd=get_cmd,
        cache_is_valid=True,
    )
    assert p._snapshot_get is True


def test_snapshot_when_snapshot_get_is_false(
    get_cmd: Literal[False, "NOT_PASSED"] | None,
    update: bool | Literal["NOT_PASSED"] | None,
    cache_is_valid: bool,
) -> None:
    p = create_parameter(
        snapshot_get=False,
        snapshot_value=True,
        get_cmd=get_cmd,
        cache_is_valid=cache_is_valid,
        offset=4,
    )

    if update != NOT_PASSED:
        s = p.snapshot(update=update)
    else:
        s = p.snapshot()

    if cache_is_valid:
        assert s["value"] == 42
        assert s["raw_value"] == 46
    else:
        assert s["value"] is None
        assert s["raw_value"] is None

    if get_cmd is not False:
        assert p.get.call_count() == 0  # type: ignore[attr-defined]


def test_snapshot_of_non_gettable_parameter_mirrors_cache(
    update: bool | Literal["NOT_PASSED"] | None, cache_is_valid: bool
) -> None:
    p = create_parameter(
        snapshot_get=True,
        snapshot_value=True,
        get_cmd=False,
        cache_is_valid=cache_is_valid,
        offset=4,
    )

    if update != NOT_PASSED:
        s = p.snapshot(update=update)
    else:
        s = p.snapshot()

    if cache_is_valid:
        assert s["value"] == 42
        assert s["raw_value"] == 46
    else:
        assert s["value"] is None
        assert s["raw_value"] is None


def test_snapshot_of_gettable_parameter_depends_on_update(
    update: bool | Literal["NOT_PASSED"] | None, cache_is_valid: bool
) -> None:
    p = create_parameter(
        snapshot_get=True,
        snapshot_value=True,
        get_cmd=lambda: 69,
        cache_is_valid=cache_is_valid,
        offset=4,
    )

    if update != NOT_PASSED:
        s = p.snapshot(update=update)
    else:
        s = p.snapshot()

    if update is not True and cache_is_valid:
        assert s["value"] == 42
        assert s["raw_value"] == 46
        assert p.get.call_count() == 0  # type: ignore[attr-defined]
    elif update is False or update == NOT_PASSED:
        assert s["value"] is None
        assert s["raw_value"] is None
        assert p.get.call_count() == 0  # type: ignore[attr-defined]
    else:
        assert s["value"] == 65
        assert s["raw_value"] == 69
        assert p.get.call_count() == 1  # type: ignore[attr-defined]


def test_snapshot_value() -> None:
    p_snapshot = Parameter(
        "no_snapshot", set_cmd=None, get_cmd=None, snapshot_value=True
    )
    p_snapshot(42)
    snap = p_snapshot.snapshot()
    assert "value" in snap
    assert "raw_value" in snap
    assert "ts" in snap
    p_no_snapshot = Parameter(
        "no_snapshot", set_cmd=None, get_cmd=None, snapshot_value=False
    )
    p_no_snapshot(42)
    snap = p_no_snapshot.snapshot()
    assert "value" not in snap
    assert "raw_value" not in snap
    assert "ts" in snap
