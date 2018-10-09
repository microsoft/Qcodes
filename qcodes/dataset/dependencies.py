from typing import Dict, Any

from qcodes.dataset.param_spec import ParamSpec


class InterDependencies:
    """
    Object containing the ParamSpecs of a given run
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
