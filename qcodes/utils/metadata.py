import json

from typing import Sequence
from .helpers import deep_update
from functools import partial

from typing import Dict, Tuple, Any, NamedTuple, Generic, TypeVar
T = TypeVar('T')
Snapshot = Dict[str, T] # TODO: represent known keys in typing
ParameterDict = Dict[Tuple[str, str], T]

class Metadatable:
    def __init__(self, metadata=None):
        self.metadata = {}
        self.load_metadata(metadata or {})

    def load_metadata(self, metadata):
        """
        Load metadata

        Args:
            metadata (dict): metadata to load
        """
        deep_update(self.metadata, metadata)

    def snapshot(self, update=False):
        """
        Decorate a snapshot dictionary with metadata.
        DO NOT override this method if you want metadata in the snapshot
        instead, override snapshot_base.

        Args:
            update (bool): Passed to snapshot_base

        Returns:
            dict: base snapshot
        """

        snap = self.snapshot_base(update=update)

        if len(self.metadata):
            snap['metadata'] = self.metadata

        return snap

    def snapshot_base(self, update: bool=False,
                      params_to_skip_update: Sequence[str]=None):
        """
        override this with the primary information for a subclass
        """
        return {}


class ParameterDiff(NamedTuple):
    # Cannot be generic in Python < 3.7:
    # https://stackoverflow.com/questions/50530959/generic-namedtuple-in-python-3-6
    left_only : ParameterDict[Any]
    right_only : ParameterDict[Any]
    changed : ParameterDict[Tuple[Any, Any]]

## FUNCTIONS ##

def extract_param_values(snapshot: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
    """
    Given a snapshot, returns a dictionary from
    instrument and parameter names onto parameter values.
    """
    parameters = {}
    for instrument_name, instrument in snapshot['station']['instruments'].items():
        for param_name, param in instrument['parameters'].items():
            if 'value' in param:
                parameters[instrument_name, param_name] = param['value']
        
    return parameters

def diff_param_values(left_snapshot: Dict[str, Any],
                      right_snapshot: Dict[str, Any]
                     ) -> Dict[str, Dict[Tuple[str, str], Any]]:
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

def diff_param_values_by_id(left_id, right_id):
    # Local import to reduce load time and
    # avoid circular references.
    from qcodes.dataset.data_set import load_by_id
    """
    Given the IDs of two datasets, returns the differences between
    parameter values in each of their snapshots.
    """
    return diff_param_values(
        json.loads(load_by_id(left_id).get_metadata('snapshot')),
        json.loads(load_by_id(right_id).get_metadata('snapshot'))
    )

