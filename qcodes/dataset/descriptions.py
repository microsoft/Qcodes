import io
from typing import Dict, Any, Union, cast
import json

from qcodes.utils.helpers import YAML
from qcodes.dataset.dependencies import (InterDependencies,
                                         InterDependencies_,
                                         new_to_old, old_to_new)
SomeDeps = Union[InterDependencies, InterDependencies_]


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

    # we operate with two version numbers
    # _version: the version of objects used inside this class
    # _written_version: the version of objects we write to the DB
    #
    # We can not simply write the current version to disk, as some third-party
    # applications may not handle that too well

    _version = 1
    _written_version = 0

    def __init__(self, interdeps: SomeDeps) -> None:

        if not isinstance(interdeps, InterDependencies_):
            raise ValueError('The interdeps arg must be of type: '
                             'InterDependencies_. '
                             f'Got {type(interdeps)}.')

        self.interdeps = interdeps

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize this object into a dictionary
        """
        ser = {}
        old_interdeps: InterDependencies

        new_interdeps = cast(InterDependencies_, self.interdeps)
        old_interdeps = new_to_old(new_interdeps)

        ser['interdependencies'] = old_interdeps.serialize()
        return ser

    @classmethod
    def deserialize(cls, ser: Dict[str, Any]) -> 'RunDescriber':
        """
        Make a RunDescriber object based on a serialized version of it
        """

        idp: Union[InterDependencies, InterDependencies_]

        if cls._is_description_old_style(ser['interdependencies']):
            idp = old_to_new(
                InterDependencies.deserialize(ser['interdependencies']))
        else:
            idp = InterDependencies_.deserialize(ser['interdependencies'])
        rundesc = cls(interdeps=idp)

        return rundesc

    @staticmethod
    def _is_description_old_style(serialized_object: Dict[str, Any]) -> bool:
        """
        Returns True if an old style description is encountered
        """

        # NOTE: we should probably think carefully about versioning; keeping
        # the runs description in sync with the API (this file)

        if 'paramspecs' in serialized_object.keys():
            return True
        else:
            return False

    def to_yaml(self) -> str:
        """
        Output the run description as a yaml string
        """
        yaml = YAML()
        with io.StringIO() as stream:
            yaml.dump(self.serialize(), stream=stream)
            output = stream.getvalue()

        return output

    def to_json(self) -> str:
        """
        Output the run describtion as a JSON string
        """
        return json.dumps(self.serialize())

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
