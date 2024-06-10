from typing import Any, NamedTuple, TypeVar

T = TypeVar("T")

# Unbound parameters or Instrument parameters
ParameterKey = str | tuple[str, str]

ParameterDict = dict[ParameterKey, T]
Snapshot = dict[str, Any]


class ParameterDiff(NamedTuple):
    # Cannot be generic in Python < 3.7:
    # https://stackoverflow.com/questions/50530959/generic-namedtuple-in-python-3-6
    left_only: ParameterDict[Any]
    right_only: ParameterDict[Any]
    changed: ParameterDict[tuple[Any, Any]]


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
