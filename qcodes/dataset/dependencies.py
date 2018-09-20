from typing import Dict, Any

from ruamel.yaml import YAML

from qcodes.dataset.param_spec import ParamSpec


class InterDependencies:
    """
    An object holding the same information as the yaml file plus
    methods for validation and extraction of data

    My idea is to have the helper functions for plotting (get_layout, get_XX)
    not call the SQLite database but this object instead. That is to say,
    the dependencies text will be read out, but all further processing happens
    via this object
    """

    def __init__(self, *paramspecs: ParamSpec) -> None:
        self.paramspecs = paramspecs

    def __repr__(self) -> str:
        output = self.__class__.__name__
        output += '('
        for ii, paramspec in enumerate(self.paramspecs):
            if ii == 0:
                output += f'{paramspec}'
            else:
                output += f', {paramspec}'
        output += ')'
        return output

    def __eq__(self, other) -> bool:
        if not isinstance(other, InterDependencies):
            return False
        if not self.paramspecs == other.paramspecs:
            return False
        return True

    def serialize(self) -> Dict[str, Any]:
        """
        Return a serialized version of this object instance
        """
        ser = {}
        ser['paramspecs'] = tuple(ps.serialize() for ps in self.paramspecs)
        return ser

    @classmethod
    def deserialize(cls, ser: Dict[str, Any]) -> 'InterDependencies':
        """
        Create an InterDependencies object from a serialization of an
        instance
        """
        paramspecs = [ParamSpec.deserialize(sps) for sps in ser['paramspecs']]
        idp = cls(*paramspecs)
        return idp


def yaml_to_interdeps(yaml_str: str) -> InterDependencies:
    yaml = YAML()
    par_dict_list = yaml.load(yaml_str)['Parameters']

    ser_ps = tuple({a: b for (a, b) in par_dict.items()}
                   for par_dict in par_dict_list)

    paramspecs = tuple(ParamSpec.deserialize(sps) for sps in ser_ps)

    return InterDependencies(*paramspecs)
