import io
from typing import Dict, Any
import json

from qcodes.dataset.dependencies import InterDependencies


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

    def __init__(self, interdeps: InterDependencies) -> None:

        if not isinstance(interdeps, InterDependencies):
            raise ValueError('The interdeps arg must be of type: '
                             f'InterDependencies. Got {type(interdeps)}.')

        self.interdeps = interdeps

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize this object into a dictionary
        """
        ser = {}
        ser['interdependencies'] = self.interdeps.serialize()
        return ser

    @classmethod
    def deserialize(cls, ser: Dict[str, Any]) -> 'RunDescriber':
        """
        Make a RunDescriber object based on a serialized version of it
        """
        idp = InterDependencies.deserialize(ser['interdependencies'])
        rundesc = cls(interdeps=idp)

        return rundesc

    @staticmethod
    def _ruamel_importer():
        try:
            from ruamel_yaml import YAML
        except ImportError:
            try:
                from ruamel.yaml import YAML
            except ImportError:
                raise ImportError('No ruamel module found. Please install '
                                  'either ruamel.yaml or ruamel_yaml to '
                                  'use the methods to_yaml and from_yaml')
        return YAML

    def to_yaml(self) -> str:
        """
        Output the run description as a yaml string
        """

        YAML = self._ruamel_importer()

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

        YAML = cls._ruamel_importer()

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
