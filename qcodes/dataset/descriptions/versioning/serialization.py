"""
The storage-facing module that handles serializations and deserializations
into different versions of the top-level object, the RunDescriber.

Serialization is implemented in two steps: converting RunDescriber objects to
plain python dicts first, and then converting them to plain formats such as
json or yaml.

Moreover this module introduces the following terms for the versions of
RunDescriber object:

- native version: the actual version of a given RunDescriber object or the
actual version encoded in its json (or other) representation
- current version: the version of RunDescriber object that is used within
QCoDeS
- storage version: the version of RunDescriber object that is used by the
data storage infrastructure of QCoDeS
"""
import io
import json
from typing import Any, Dict, Type, Union

from qcodes.utils.helpers import YAML
import qcodes.dataset.descriptions.rundescriber as current
import qcodes.dataset.descriptions.versioning.v0 as v0
from qcodes.dataset.descriptions.versioning.converters import (
    v0_to_v1, v1_to_v0)

CURRENT_VERSION = 1
STORAGE_VERSION = 0


SomeRunDescriber = Union[current.RunDescriber, v0.RunDescriber]
SomeRunDescriberType = Union[Type[v0.RunDescriber],
                             Type[current.RunDescriber]]

# keys: (from_version, to_version)
_converters = {(0, 0): lambda x: x,
               (0, 1): v0_to_v1,
               (1, 0): v1_to_v0,
               (1, 1): lambda x: x}


def deserialize(ser: Dict[str, Any]) -> SomeRunDescriber:
    """
    Deserialize a dict (usually coming from json.loads) into a RunDescriber
    object according to the version specified in the dict
    """
    run_describers: Dict[int, SomeRunDescriberType]
    run_describers = {0: v0.RunDescriber,
                      1: current.RunDescriber}

    return run_describers[ser['version']].deserialize(ser)


def deserialize_to_current(ser: Dict[str, Any]) -> current.RunDescriber:
    """
    Deserialize a dict into a RunDescriber of the current version
    """
    desc = deserialize(ser)

    return _converters[(desc.version, CURRENT_VERSION)](desc)


def serialize_to_version(desc: SomeRunDescriber,
                         version: int) -> Dict[str, Any]:
    """
    Serialize a RunDescriber to a particular version
    """
    return _converters[(desc.version, version)](desc).serialize()


def serialize_to_storage(desc: SomeRunDescriber) -> Dict[str, Any]:
    """
    Serialize a RunDescriber into the storage version
    """
    return serialize_to_version(desc, STORAGE_VERSION)


def make_json_for_storage(desc: SomeRunDescriber) -> str:
    """
    Serialize a RunDescriber to the storage version and dump that as a JSON
    string
    """
    return json.dumps(serialize_to_storage(desc))


def make_json_in_version(desc: SomeRunDescriber, version: int) -> str:
    """
    Serialize a RunDescriber to a particular version and JSON dump that.
    Only to be used in tests and upgraders
    """
    return json.dumps(serialize_to_version(desc, version))


def read_json_to_current(json_str: str) -> current.RunDescriber:
    """
    Load a dict from JSON string and deserialize it into a RunDescriber of
    the current version
    """
    return deserialize_to_current(json.loads(json_str))


def read_json_to_native_version(json_str: str) -> SomeRunDescriber:
    """
    Load a dict from JSON string and deserialize it into a RunDescriber of the
    version given in the JSON (native version)
    """
    return deserialize(json.loads(json_str))


def make_yaml_for_storage(desc: SomeRunDescriber) -> str:
    """
    Serialize a RunDescriber to the storage version and dump that as a yaml
    string
    """
    yaml = YAML()
    with io.StringIO() as stream:
        yaml.dump(serialize_to_storage(desc), stream=stream)
        output = stream.getvalue()

    return output


def read_yaml_to_current(yaml_str: str) -> current.RunDescriber:
    """
    Load a dict from yaml string and deserialize it into a RunDescriber of
    the current version
    """
    yaml = YAML()
    # yaml.load returns an OrderedDict, but we need a dict
    ser = dict(yaml.load(yaml_str))
    return deserialize_to_current(ser)
