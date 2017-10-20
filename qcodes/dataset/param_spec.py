from typing import List
from qcodes.instrument.parameter import _BaseParameter


# TODO: we should validate type somehow
# we can't accept everything (or we can but crash at runtime?)
# we only support the types in VALUES type
class ParamSpec():
    def __init__(self, name: str, type: str,
                 label: str=None,
                 unit: str=None,
                 inferred_from: List['ParamSpec']=None,
                 depends_on: List['ParamSpec']=None,
                 **metadata) -> None:
        """
        Args:
            name: name of the parameter
            type: type of the parameter
            label: label of the parameter
            inferred_from: the parameters that this parameter is inferred_from
            depends_on: the parameters that this parameter depends on
        """
        self.name = name
        self.type = type
        self.label = '' if label is None else label
        self.unit = '' if unit is None else unit
        if inferred_from:
            self.inferred_from = ', '.join([ps.name for ps in inferred_from])
        else:
            self.inferred_from = ''

        if depends_on:
            self.depends_on = ', '.join([ps.name for ps in depends_on])
        else:
            self.depends_on = ''

        if metadata:
            self.metadata = metadata

    def sql_repr(self):
        return f"{self.name} {self.type}"

    def __repr__(self):
        return f"{self.name} ({self.type})"


def param_spec(parameter: _BaseParameter, type: str) -> ParamSpec:
    """ Generates a ParamSpec from a qcodes parameter

    Args:
        - parameter: the qcodes parameter to make a spec

    """
    return ParamSpec(parameter.name, type, **parameter.metadata)
