from typing import Dict, Any, List

from qcodes.dataset.param_spec import ParamSpec


class InterDependencies:
    """
    Object containing the ParamSpecs of a given run
    """

    def __init__(self, *paramspecs: ParamSpec) -> None:

        for paramspec in paramspecs:
            if not isinstance(paramspec, ParamSpec):
                raise ValueError('Got invalid input. All paramspecs must be '
                                 f'ParamSpecs, but {paramspec} is of type '
                                 f'{type(paramspec)}.')

        self.paramspecs = paramspecs

    def __repr__(self) -> str:
        output = self.__class__.__name__
        tojoin = (str(paramspec) for paramspec in self.paramspecs)
        output += f'({", ".join(tojoin)})'
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

    # also: method to check for cycles (and other invalid stuff)

    @staticmethod
    def _are_dependencies_met(*params) -> bool:
        """
        Determine whether all dependencies are met, i.e. that for every
        parameter that has dependencies, those dependencies are also present
        """
        needed: List[str] = []
        present: List[str] = []

        for param in params:
            if param.name in needed:
                needed.remove(param.name)
            present.append(param.name)
            param_deps = [sp for sp in param.depends_on.split(', ')
                          if sp != '']
            param_infs = [sp for sp in param.inferred_from.split(', ')
                          if sp != '']
            for must_have in [param_deps, param_infs]:
                needed += [sp for sp in must_have if sp not in present]

        if len(needed) > 0:
            return False
        else:
            return True

    @classmethod
    def deserialize(cls, ser: Dict[str, Any]) -> 'InterDependencies':
        """
        Create an InterDependencies object from a serialization of an
        instance
        """
        paramspecs = [ParamSpec.deserialize(sps) for sps in ser['paramspecs']]
        idp = cls(*paramspecs)
        return idp
