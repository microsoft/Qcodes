import io
from typing import Dict, Any, Union, Optional
import json
from copy import deepcopy

from qcodes.utils.helpers import YAML
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.versioning.v0 import InterDependencies
from qcodes.dataset.descriptions.versioning.converters import (
    new_to_old, old_to_new)


class RunDescriber:
    """
    The object that holds the description of each run in the database. This
    object serialises itself to a string and is found under the run_description
    column in the runs table

    Extension of this object is planned for the future, for now it holds the
    parameter interdependencies. Extensions should be objects that can
    serialize themselves added as attributes to the RunDescriber , such that
    the RunDescriber can iteratively serialize its attributes when serializing
    itself.
    """

    def __init__(self, interdeps: InterDependencies_) -> None:

        if not isinstance(interdeps, InterDependencies_):
            raise ValueError('The interdeps arg must be of type: '
                             'InterDependencies_. '
                             f'Got {type(interdeps)}.')

        self.interdeps = interdeps

        # we operate with two version numbers
        # _version: the version of objects used inside this class
        # _written_version: the version of objects we write to the DB
        #
        # We can not simply write the current version to disk, as some
        # third-party applications may not handle that too well
        #
        # Version log:
        #
        # 0: The run_describer has a single attribute, interdeps, which is
        # an instance of InterDependencies (which contains ParamSpecs)
        #
        # 1: The run_describer has a single attribute, interdeps, which is
        # an instance of InterDependencies_ (which contains ParamSpecBases)
        #

        self._version = 1
        self._written_version = 0

        # key: tuple of (from_version, to_version)
        self._serializers = {(1, 0): self._serialize_1_as_0,
                             (1, 1): self._serialize_1_as_1}

    def serialize(self, version: Optional[int] = None) -> Dict[str, Any]:
        """
        Serialize this object into a dictionary
        """
        write_v = version if version is not None else self._written_version

        key = (self._version, write_v)
        return self._serializers[key](self)

    @classmethod
    def deserialize(cls, ser: Dict[str, Any]) -> 'RunDescriber':
        """
        Make a RunDescriber object based on a serialized version of it
        """

        idp: Union[InterDependencies, InterDependencies_]

        # None corresponds to old, unversioned RunDescribers

        deserializers = {None: cls._deserialize_from_v_0,
                         0: cls._deserialize_from_v_0,
                         1: cls._deserialize_from_v_1}

        version = ser.get('version', None)

        rundesc = deserializers[version](ser)

        return rundesc

    @classmethod
    def _deserialize_from_v_0(cls, ser: Dict[str, Any]) -> 'RunDescriber':
        """
        Make a RunDescriber object of the current version from a version 0
        serialization
        """
        idps = old_to_new(
            InterDependencies.deserialize(ser['interdependencies']))
        return cls(idps)

    @classmethod
    def _deserialize_from_v_1(cls, ser: Dict[str, Any]) -> 'RunDescriber':
        """
        Make a RunDescriber object of the current version from a version 0
        serialization
        """
        idps = InterDependencies_.deserialize(ser['interdependencies'])
        return cls(idps)

    def to_yaml(self, version: Optional[int] = None) -> str:
        """
        Output the run description as a yaml string
        """
        yaml = YAML()
        with io.StringIO() as stream:
            yaml.dump(self.serialize(version=version), stream=stream)
            output = stream.getvalue()

        return output

    def to_json(self, version: Optional[int] = None) -> str:
        """
        Output the run describtion as a JSON string
        """
        return json.dumps(self.serialize(version=version))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'RunDescriber':
        """
        Parse a yaml string (the return of `to_yaml`) into a RunDescriber
        object
        """
        yaml = YAML()
        # yaml.load returns an OrderedDict, but we need a dict
        ser = dict(yaml.load(yaml_str))
        return cls.deserialize(ser)

    @classmethod
    def from_json(cls, json_str: str) -> 'RunDescriber':
        """
        Parse a JSON string (the return value of `to_json`) into a
        RunDescriber object
        """
        return cls.deserialize(json.loads(json_str))

    def __eq__(self, other):
        if not isinstance(other, RunDescriber):
            return False
        if self.interdeps != other.interdeps:
            return False
        return True

    def __repr__(self) -> str:
        return f"RunDescriber({self.interdeps})"

    @staticmethod
    def _serialize_1_as_0(desc: 'RunDescriber') -> Dict[str, Any]:
        """
        Serialize a RunDescriber object of version 1 as version 0
        """
        if desc._version != 1:
            raise ValueError('Invalid RunDescriber version. Got version '
                            f'{desc._version}, expected version 1')
        new_desc = deepcopy(desc)
        new_desc.interdeps = new_to_old(new_desc.interdeps)  # type: ignore
        new_desc._version = 0

        ser: Dict[str, Any] = {}
        ser['version'] = new_desc._version
        ser['interdependencies'] = new_desc.interdeps.serialize()


        return ser

    @staticmethod
    def _serialize_1_as_1(desc: 'RunDescriber') -> Dict[str, Any]:
        """
        Serialize a RunDescriber object of version 1 as version 0
        """
        if desc._version != 1:
            raise ValueError('Invalid RunDescriber version. Got version '
                            f'{desc._version}, expected version 1')

        ser: Dict[str, Any] = {}
        ser['version'] = desc._version
        ser['interdependencies'] = desc.interdeps.serialize()

        return ser
