from typing import (Any, Dict, NamedTuple, NewType, Sequence, Tuple, TypeVar,
                    Union, Optional)

from .helpers import deep_update

T = TypeVar('T')
# NB: At the moment, the Snapshot type is a bit weak, as the Any
#     for the value type doesn't tell us anything about the schema
#     followed by snapshots.
#     This is needed, however, since snapshots are Dict instances with
#     homogeneous keys and heterogeneous values, something that
#     recent Python versions largely replace with features like
#     typing.NamedTuple and @dataclass.
#     As those become more widely available, the weakness of this
#     type constraint will become less of an issue.
Snapshot = Dict[str, Any]
ParameterKey = Union[
    # Unbound parameters
    str,
    # Instrument parameters
    Tuple[str, str]
]
ParameterDict = Dict[ParameterKey, T]
RunId = NewType('RunId', int)


class Metadatable:
    def __init__(self, metadata=None):
        self.metadata = {}
        self.load_metadata(metadata or {})

    def load_metadata(self, metadata: dict) -> None:
        """
        Load metadata into this classes metadata dictionary.

        Args:
            metadata: Metadata to load.
        """
        deep_update(self.metadata, metadata)

    def snapshot(self, update: Optional[bool] = False) -> Dict:
        """
        Decorate a snapshot dictionary with metadata.
        DO NOT override this method if you want metadata in the snapshot
        instead, override :meth:`snapshot_base`.

        Args:
            update: Passed to snapshot_base.

        Returns:
            Base snapshot.
        """

        snap = self.snapshot_base(update=update)

        if len(self.metadata):
            snap['metadata'] = self.metadata

        return snap

    def snapshot_base(
            self, update: Optional[bool] = False,
            params_to_skip_update: Optional[Sequence[str]] = None) -> Dict:
        """
        Override this with the primary information for a subclass.
        """
        return {}


class ParameterDiff(NamedTuple):
    # Cannot be generic in Python < 3.7:
    # https://stackoverflow.com/questions/50530959/generic-namedtuple-in-python-3-6
    left_only: ParameterDict[Any]
    right_only: ParameterDict[Any]
    changed: ParameterDict[Tuple[Any, Any]]

## FUNCTIONS ##

def extract_param_values(snapshot: Snapshot) -> Dict[ParameterKey, Any]:
    """
    Given a snapshot, returns a dictionary from
    instrument and parameter names onto parameter values.
    """
    parameters = {}
    snapshot = snapshot.get('station', snapshot)
    for param_name, param in snapshot['parameters'].items():
        parameters[param_name] = param['value']
    if 'instruments' in snapshot:
        for instrument_name, instrument in snapshot['instruments'].items():
            for param_name, param in instrument['parameters'].items():
                if 'value' in param:
                    parameters[instrument_name, param_name] = param['value']

    return parameters



def diff_param_values(left_snapshot: Snapshot,
                      right_snapshot: Snapshot
                      ) -> ParameterDiff:
    """
    Given two snapshots, returns the differences between parameter values
    in each.
    """
    left_params, right_params = map(extract_param_values, (left_snapshot, right_snapshot))
    left_keys, right_keys = [set(params.keys()) for params in (left_params, right_params)]
    common_keys = left_keys.intersection(right_keys)

    return ParameterDiff(
        left_only={
            key: left_params[key]
            for key in left_keys.difference(common_keys)
        },
        right_only={
            key: right_params[key]
            for key in right_keys.difference(common_keys)
        },
        changed={
            key: (left_params[key], right_params[key])
            for key in common_keys
            if left_params[key] != right_params[key]
        }
    )


def diff_param_values_by_id(left_id: RunId, right_id: RunId) -> ParameterDiff:
    """
    Given the IDs of two datasets, returns the differences between
    parameter values in each of their snapshots.
    """
    # Local import to reduce load time and
    # avoid circular references.
    from qcodes.dataset.data_set import load_by_id

    left_snapshot = load_by_id(left_id).snapshot
    right_snapshot = load_by_id(right_id).snapshot

    if left_snapshot is None or right_snapshot is None:
        if left_snapshot is None:
            empty = left_id
        else:
            empty = right_id
        raise RuntimeError(f"Tried to compare two snapshots"
                           f"but the snapshot of {empty} "
                           f"is empty.")

    return diff_param_values(left_snapshot, right_snapshot)
