"""
The storage-facing module that handles serializations and deserializations
of the top-level object, the RunDescriber into and from different versions.

Note that we require strict backwards and forwards compatibility such that
the current RunDescriber must always be deserializable from any older or newer
serialization.

This means that a new version cannot delete/omit previously included fields
from the serialization and the deserialization must be written such that it
can handle that any new field may be missing.

The above excludes v1 that only serialized for a short amount of time.
See py:module`.database_fix_functions` to convert v1 RunDescribers that has
been written to the db.

Serialization is implemented in two steps: converting RunDescriber objects to
plain python dicts first, and then converting them to plain formats such as
json or yaml. The dict representation of the ``RunDescriber`` is defined in
py:module`.rundescribertypes`

Moreover this module introduces the following terms for the versions of
RunDescriber object:

- storage version: the version of RunDescriber serialization that is used
by the data storage infrastructure of QCoDeS.

The names of the functions in this module follow the "to_*"/"from_*"
convention where "*" stands for the storage format. Also note the
"as_version", "for_storage", and "to_current" suffixes.
"""
from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any, cast

from .. import rundescriber as current
from .converters import (
    v0_to_v1,
    v0_to_v2,
    v0_to_v3,
    v1_to_v0,
    v1_to_v2,
    v1_to_v3,
    v2_to_v0,
    v2_to_v1,
    v2_to_v3,
    v3_to_v0,
    v3_to_v1,
    v3_to_v2,
)
from .rundescribertypes import (
    RunDescriberDicts,
    RunDescriberV0Dict,
    RunDescriberV1Dict,
    RunDescriberV2Dict,
    RunDescriberV3Dict,
)

if TYPE_CHECKING:
    from collections.abc import Callable

STORAGE_VERSION = 3
# the version of :class:`RunDescriber` object that is used by the data storage
# infrastructure of :mod:`qcodes`

# keys: (from_version, to_version)
_converters: dict[tuple[int, int], Callable[..., Any]] = {
    (0, 0): lambda x: x,
    (0, 1): v0_to_v1,
    (0, 2): v0_to_v2,
    (0, 3): v0_to_v3,
    (1, 0): v1_to_v0,
    (1, 1): lambda x: x,
    (1, 2): v1_to_v2,
    (1, 3): v1_to_v3,
    (2, 0): v2_to_v0,
    (2, 1): v2_to_v1,
    (2, 2): lambda x: x,
    (2, 3): v2_to_v3,
    (3, 0): v3_to_v0,
    (3, 1): v3_to_v1,
    (3, 2): v3_to_v2,
    (3, 3): lambda x: x,

}


def from_dict_to_current(dct: RunDescriberDicts) -> current.RunDescriber:
    """
    Convert a dict into a RunDescriber of the current version
    """
    dct_version = dct['version']
    if dct_version == 0:
        return current.RunDescriber._from_dict(cast(RunDescriberV0Dict, dct))
    elif dct_version == 1:
        return current.RunDescriber._from_dict(cast(RunDescriberV1Dict, dct))
    elif dct_version == 2:
        return current.RunDescriber._from_dict(cast(RunDescriberV2Dict, dct))
    elif dct_version >= 3:
        return current.RunDescriber._from_dict(cast(RunDescriberV3Dict, dct))
    else:
        raise RuntimeError(f"Unknown version of run describer dictionary, can't deserialize. The dictionary is {dct!r}")


def to_dict_as_version(desc: current.RunDescriber,
                       version: int) -> RunDescriberDicts:
    """
    Convert the given RunDescriber into a dictionary that represents a
    RunDescriber of the given version
    """
    input_version = desc.version
    input_dict = desc._to_dict()
    output_dict = _converters[(input_version, version)](input_dict)
    return output_dict


def to_dict_for_storage(desc: current.RunDescriber) -> RunDescriberDicts:
    """
    Convert a RunDescriber into a dictionary that represents the
    RunDescriber of the storage version
    """
    return to_dict_as_version(desc, STORAGE_VERSION)


# JSON


def to_json_for_storage(desc: current.RunDescriber) -> str:
    """
    Serialize the given RunDescriber to JSON as a RunDescriber of the
    version for storage
    """
    return json.dumps(to_dict_for_storage(desc))


def to_json_as_version(desc: current.RunDescriber, version: int) -> str:
    """
    Serialize the given RunDescriber to JSON as a RunDescriber of the
    given version. Only to be used in tests and upgraders
    """
    return json.dumps(to_dict_as_version(desc, version))


def from_json_to_current(json_str: str) -> current.RunDescriber:
    """
    Deserialize a JSON string into a RunDescriber of the current version
    """

    data = json.loads(json_str)
    # json maps both list and tuple to list
    # since we always expects shapes to be a tuple
    # convert it back to a tuple here
    shapes = data.get("shapes", None)
    if shapes is not None:
        for name, shapelist in shapes.items():
            shapes[name] = tuple(shapelist)

    return from_dict_to_current(data)


rundescriber_from_json = from_json_to_current


# YAML


def to_yaml_for_storage(desc: current.RunDescriber) -> str:
    """
    Serialize the given RunDescriber to YAML as a RunDescriber of the
    version for storage
    """
    import ruamel.yaml  # lazy import

    yaml = ruamel.yaml.YAML()
    with io.StringIO() as stream:
        yaml.dump(to_dict_for_storage(desc), stream=stream)
        output = stream.getvalue()

    return output


def from_yaml_to_current(yaml_str: str) -> current.RunDescriber:
    """
    Deserialize a YAML string into a RunDescriber of the current version
    """
    import ruamel.yaml  # lazy import

    yaml = ruamel.yaml.YAML()
    # yaml.load returns an OrderedDict, but we need a dict
    ser = cast(RunDescriberDicts, dict(yaml.load(yaml_str)))
    return from_dict_to_current(ser)
