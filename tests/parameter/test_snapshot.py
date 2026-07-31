from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pytest
from typing_extensions import ParamSpec

from qcodes.metadatable import normalize_snapshot_update
from qcodes.parameters import Parameter

from .conftest import NOT_PASSED

if TYPE_CHECKING:
    from collections.abc import Callable

    from qcodes.metadatable import SnapshotUpdate

T = TypeVar("T")
P = ParamSpec("P")


def create_parameter(
    snapshot_get: bool | Literal["NOT_PASSED"],
    snapshot_value: bool | Literal["NOT_PASSED"],
    cache_is_valid: bool,
    get_cmd: Callable[..., Any] | bool | Literal["NOT_PASSED"] | None,
    offset: float | Literal["NOT_PASSED"] = NOT_PASSED,
) -> Parameter:
    kwargs: dict[str, Any] = {
        "set_cmd": None,
        "label": "Parameter",
        "unit": "a.u.",
        "docstring": "some docs",
    }

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
    get_cmd: Literal[False, "NOT_PASSED"] | None,
    cache_is_valid: bool,
    update: SnapshotUpdate | Literal["NOT_PASSED"],
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
    update: SnapshotUpdate | Literal["NOT_PASSED"],
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
        ts = datetime.fromisoformat(s["ts"])
        t0_up_to_seconds = t0.replace(microsecond=0)
        assert ts >= t0_up_to_seconds
    else:
        assert t0 is None
        assert s["ts"] is None


def test_snapshot_timestamp_for_valid_cache_depends_on_cache_update(
    snapshot_get: bool | Literal["NOT_PASSED"],
    snapshot_value: bool | Literal["NOT_PASSED"],
    update: SnapshotUpdate | Literal["NOT_PASSED"],
) -> None:
    p = create_parameter(
        snapshot_get, snapshot_value, get_cmd=lambda: 69, cache_is_valid=True
    )

    # Hack cache's timestamp to simplify this test
    timestamp = p.cache.timestamp
    assert timestamp is not None
    p.cache._timestamp = timestamp - timedelta(days=31)  # type: ignore[attr-defined]

    tu = datetime.now(UTC)
    assert p.cache.timestamp is not None
    assert p.cache.timestamp < tu  # pre-condition
    if update != NOT_PASSED:
        s = p.snapshot(update=update)
    else:
        s = p.snapshot()

    ts = datetime.fromisoformat(s["ts"])
    tu_up_to_seconds = tu.replace(microsecond=0)

    effective = "Only_invalid" if update == NOT_PASSED else update
    cache_gets_updated_on_snapshot_call = (
        snapshot_value is not False and snapshot_get is not False and effective == "All"
    )

    if cache_gets_updated_on_snapshot_call:
        assert ts >= tu_up_to_seconds
    else:
        assert ts < tu_up_to_seconds


def test_snapshot_timestamp_for_invalid_cache_depends_only_on_snapshot_flags(
    snapshot_get: bool | Literal["NOT_PASSED"],
    snapshot_value: bool | Literal["NOT_PASSED"],
    update: SnapshotUpdate | Literal["NOT_PASSED"],
) -> None:
    p = create_parameter(
        snapshot_get, snapshot_value, get_cmd=lambda: 69, cache_is_valid=False
    )

    effective = "Only_invalid" if update == NOT_PASSED else update
    cache_gets_updated_on_snapshot_call = (
        snapshot_value is not False
        and snapshot_get is not False
        and effective != "Never"
    )

    if cache_gets_updated_on_snapshot_call:
        tu = datetime.now(UTC)
    else:
        tu = None

    if update != NOT_PASSED:
        s = p.snapshot(update=update)
    else:
        s = p.snapshot()

    if cache_gets_updated_on_snapshot_call:
        ts = datetime.fromisoformat(s["ts"])
        assert tu is not None
        tu_up_to_seconds = tu.replace(microsecond=0)
        assert ts >= tu_up_to_seconds
    else:
        assert s["ts"] is None


def test_snapshot_when_snapshot_value_is_false(
    snapshot_get: bool | Literal["NOT_PASSED"],
    get_cmd: Literal[False, "NOT_PASSED"] | None,
    cache_is_valid: bool,
    update: SnapshotUpdate | Literal["NOT_PASSED"],
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
    update: SnapshotUpdate | Literal["NOT_PASSED"],
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
    update: SnapshotUpdate | Literal["NOT_PASSED"], cache_is_valid: bool
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
    update: SnapshotUpdate | Literal["NOT_PASSED"], cache_is_valid: bool
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

    effective = "Only_invalid" if update == NOT_PASSED else update
    should_get = effective == "All" or (
        effective == "Only_invalid" and not cache_is_valid
    )

    if should_get:
        assert s["value"] == 65
        assert s["raw_value"] == 69
        assert p.get.call_count() == 1  # type: ignore[attr-defined]
    elif cache_is_valid:
        assert s["value"] == 42
        assert s["raw_value"] == 46
        assert p.get.call_count() == 0  # type: ignore[attr-defined]
    else:
        assert s["value"] is None
        assert s["raw_value"] is None
        assert p.get.call_count() == 0  # type: ignore[attr-defined]


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


def test_normalize_snapshot_update_maps_to_canonical_values() -> None:
    # canonical string values are returned unchanged
    assert normalize_snapshot_update("All") == "All"
    assert normalize_snapshot_update("Only_invalid") == "Only_invalid"
    assert normalize_snapshot_update("Never") == "Never"
    # legacy values are mapped to the canonical string values
    assert normalize_snapshot_update(True) == "All"
    assert normalize_snapshot_update(None) == "Only_invalid"
    assert normalize_snapshot_update(False) == "Never"


def test_normalize_snapshot_update_rejects_unknown_string() -> None:
    with pytest.raises(ValueError, match="Invalid value for snapshot"):
        normalize_snapshot_update("bogus")  # type: ignore[arg-type]


def test_snapshot_update_all_always_calls_get() -> None:
    p = create_parameter(
        snapshot_get=True,
        snapshot_value=True,
        get_cmd=lambda: 69,
        cache_is_valid=True,
    )
    s = p.snapshot(update="All")
    assert s["value"] == 69
    assert p.get.call_count() == 1  # type: ignore[attr-defined]


def test_snapshot_update_never_never_calls_get() -> None:
    p = create_parameter(
        snapshot_get=True,
        snapshot_value=True,
        get_cmd=lambda: 69,
        cache_is_valid=False,
    )
    s = p.snapshot(update="Never")
    assert s["value"] is None
    assert p.get.call_count() == 0  # type: ignore[attr-defined]


def test_snapshot_update_only_invalid_calls_get_when_cache_invalid() -> None:
    p = create_parameter(
        snapshot_get=True,
        snapshot_value=True,
        get_cmd=lambda: 69,
        cache_is_valid=False,
    )
    s = p.snapshot(update="Only_invalid")
    assert s["value"] == 69
    assert p.get.call_count() == 1  # type: ignore[attr-defined]


def test_snapshot_update_only_invalid_skips_get_when_cache_valid() -> None:
    p = create_parameter(
        snapshot_get=True,
        snapshot_value=True,
        get_cmd=lambda: 69,
        cache_is_valid=True,
    )
    s = p.snapshot(update="Only_invalid")
    # the cached (set) value is used, ``get`` is not called
    assert s["value"] == 42
    assert p.get.call_count() == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("legacy", "string"),
    ((True, "All"), (None, "Only_invalid"), (False, "Never")),
)
def test_snapshot_update_string_matches_legacy_value(
    legacy: bool | None,
    string: Literal["All", "Only_invalid", "Never"],
    cache_is_valid: bool,
) -> None:
    p_legacy = create_parameter(
        snapshot_get=True,
        snapshot_value=True,
        get_cmd=lambda: 69,
        cache_is_valid=cache_is_valid,
    )
    p_string = create_parameter(
        snapshot_get=True,
        snapshot_value=True,
        get_cmd=lambda: 69,
        cache_is_valid=cache_is_valid,
    )

    # ``update=legacy`` intentionally uses the deprecated bool/None values to
    # confirm they still map to the new canonical behavior.
    s_legacy = p_legacy.snapshot(update=legacy)  # pyright: ignore[reportDeprecated]
    s_string = p_string.snapshot(update=string)

    assert s_legacy["value"] == s_string["value"]
    assert s_legacy["raw_value"] == s_string["raw_value"]
    assert (
        p_legacy.get.call_count()  # type: ignore[attr-defined]
        == p_string.get.call_count()  # type: ignore[attr-defined]
    )


def test_snapshot_rejects_unknown_update_value() -> None:
    p = create_parameter(
        snapshot_get=True,
        snapshot_value=True,
        get_cmd=lambda: 69,
        cache_is_valid=True,
    )
    with pytest.raises(ValueError, match="Invalid value for snapshot"):
        p.snapshot(update="bogus")  # type: ignore[arg-type]
