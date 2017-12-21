from typing import List, Union, cast, Sequence
from qcodes.instrument.parameter import _BaseParameter


# TODO: we should validate type somehow
# we can't accept everything (or we can but crash at runtime?)
# we only support the types in VALUES type
class ParamSpec():
    def __init__(self, name: str,
                 paramtype: str,
                 label: str=None,
                 unit: str=None,
                 inferred_from: Sequence[Union['ParamSpec', str]]=None,
                 depends_on: Sequence[Union['ParamSpec', str]]=None,
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
        self.type = paramtype
        self.label = '' if label is None else label
        self.unit = '' if unit is None else unit

        # a bit of footwork to allow for entering either strings or ParamSpecs
        if inferred_from:
            temp_inf_from = []
            for inff in inferred_from:
                if hasattr(inff, 'name'):
                    inff = cast('ParamSpec', inff)
                    temp_inf_from.append(inff.name)
                else:
                    inff = cast(str, inff)
                    temp_inf_from.append(inff)
            self.inferred_from = ', '.join(temp_inf_from)
        else:
            self.inferred_from = ''

        if depends_on:
            temp_dep_on = []
            for dpn in depends_on:
                if hasattr(dpn, 'name'):
                    dpn = cast('ParamSpec', dpn)
                    temp_dep_on.append(dpn.name)
                else:
                    dpn = cast(str, dpn)
                    temp_dep_on.append(dpn)
            self.depends_on = ', '.join(temp_dep_on)
        else:
            self.depends_on = ''

        if metadata:
            self.metadata = metadata

    def sql_repr(self):
        return f"{self.name} {self.type}"

    def __repr__(self):
        return f"{self.name} ({self.type})"


def param_spec(parameter: _BaseParameter, paramtype: str) -> ParamSpec:
    """ Generates a ParamSpec from a qcodes parameter

    Args:
        - parameter: the qcodes parameter to make a spec

    """
    return ParamSpec(parameter.name, paramtype, **parameter.metadata)
