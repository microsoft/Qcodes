from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..param_spec import ParamSpec

if TYPE_CHECKING:
    from .rundescribertypes import InterDependenciesDict


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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, InterDependencies):
            return False
        ours = sorted(self.paramspecs, key=lambda ps: ps.name)
        theirs = sorted(other.paramspecs, key=lambda ps: ps.name)
        if not ours == theirs:
            return False
        return True

    def _to_dict(self) -> InterDependenciesDict:
        """
        Return a dictionary representation of this object instance
        """

        return {'paramspecs': tuple(ps._to_dict() for ps in self.paramspecs)}


    @classmethod
    def _from_dict(cls, ser: InterDependenciesDict) -> InterDependencies:
        """
        Create an InterDependencies object from a dictionary
        """
        paramspecs = [ParamSpec._from_dict(sps) for sps in ser['paramspecs']]
        idp = cls(*paramspecs)
        return idp
