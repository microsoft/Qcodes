from typing import TYPE_CHECKING, Any, NamedTuple

# Unbound parameters or Instrument parameters
ParameterKey = str | tuple[str, str]

type ParameterDict[T] = dict[ParameterKey, T]
Snapshot = dict[str, Any]


class ParameterDiff(NamedTuple):
    # Cannot be generic in Python < 3.7:
    # https://stackoverflow.com/questions/50530959/generic-namedtuple-in-python-3-6
    left_only: ParameterDict[Any]
    right_only: ParameterDict[Any]
    changed: ParameterDict[tuple[Any, Any]]

    def __str__(self) -> str:
        return format_parameter_diff(self)


def _format_parameter_key(key: ParameterKey) -> str:
    if isinstance(key, tuple):
        return ".".join(key)
    return key


def format_parameter_diff(
    diff: ParameterDiff,
    left_name: str = "left",
    right_name: str = "right",
) -> str:
    """
    Render a :class:`ParameterDiff` as a human-readable multi-line string.

    Args:
        diff: the difference to render.
        left_name: name used to refer to the left hand side snapshot.
        right_name: name used to refer to the right hand side snapshot.

    Returns:
        A human-readable representation of the differences.

    """
    lines: list[str] = []

    if diff.changed:
        lines.append(f"Changed parameters ({left_name} -> {right_name}):")
        lines.extend(
            f"  {_format_parameter_key(key)}: {left!r} -> {right!r}"
            for key, (left, right) in sorted(
                diff.changed.items(), key=lambda item: _format_parameter_key(item[0])
            )
        )
    if diff.left_only:
        lines.append(f"Parameters only in {left_name}:")
        lines.extend(
            f"  {_format_parameter_key(key)}: {value!r}"
            for key, value in sorted(
                diff.left_only.items(), key=lambda item: _format_parameter_key(item[0])
            )
        )
    if diff.right_only:
        lines.append(f"Parameters only in {right_name}:")
        lines.extend(
            f"  {_format_parameter_key(key)}: {value!r}"
            for key, value in sorted(
                diff.right_only.items(), key=lambda item: _format_parameter_key(item[0])
            )
        )

    if not lines:
        return "No differences between the two snapshots."

    return "\n".join(lines)


def extract_param_values(snapshot: Snapshot) -> dict[ParameterKey, Any]:
    """
    Given a snapshot, returns a dictionary from
    instrument and parameter names onto parameter values.
    """
    parameters = {}
    snapshot = snapshot.get("station", snapshot)
    for param_name, param in snapshot["parameters"].items():
        parameters[param_name] = param["value"]
    if "instruments" in snapshot:
        for instrument_name, instrument in snapshot["instruments"].items():
            for param_name, param in instrument["parameters"].items():
                if "value" in param:
                    parameters[instrument_name, param_name] = param["value"]

    return parameters


def diff_param_values(
    left_snapshot: Snapshot, right_snapshot: Snapshot
) -> ParameterDiff:
    """
    Given two snapshots, returns the differences between parameter values
    in each.
    """
    left_params, right_params = map(
        extract_param_values, (left_snapshot, right_snapshot)
    )
    left_keys, right_keys = (
        set(params.keys()) for params in (left_params, right_params)
    )
    common_keys = left_keys.intersection(right_keys)

    return ParameterDiff(
        left_only={key: left_params[key] for key in left_keys.difference(common_keys)},
        right_only={
            key: right_params[key] for key in right_keys.difference(common_keys)
        },
        changed={
            key: (left_params[key], right_params[key])
            for key in common_keys
            if left_params[key] != right_params[key]
        },
    )


if not TYPE_CHECKING:
    from typing import TypeVar

    from qcodes.utils.deprecate import _make_deprecated_typevars_getattr

    __getattr__ = _make_deprecated_typevars_getattr(
        __name__,
        {
            "T": TypeVar("T"),
        },
    )
