import io
from typing import Dict, Any

from ruamel.yaml import YAML

from qcodes.dataset.dependencies import InterDependencies


class RunDescriber:

    def __init__(self, interdeps: InterDependencies) -> None:
        self.interdeps = interdeps

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize this object into a dictionary
        """
        ser = {}
        ser['Parameters'] = self.interdeps.serialize()
        return ser

    @classmethod
    def deserialize(cls, ser: Dict[str, Any]) -> 'RunDescriber':
        """
        Make a RunDescriber object based on a serialized version of it
        """
        idp = InterDependencies.deserialize(ser['Parameters'])
        rundesc = cls(interdeps=idp)

        return rundesc

    def to_yaml(self):
        """
        Output the run description as a yaml string
        """
        yaml = YAML()
        stream = io.StringIO()
        yaml.dump(self.serialize(), stream=stream)
        output = stream.getvalue()
        stream.close()
        return output

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

    def __eq__(self, other):
        if not isinstance(other, RunDescriber):
            return False
        if self.interdeps != other.interdeps:
            return False
        return True
