from typing import Dict, Any

from qcodes.dataset.descriptions.param_spec import ParamSpec


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

    def _to_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary representation of this object instance
        """
        ser = {}
        ser['paramspecs'] = tuple(ps._to_dict() for ps in self.paramspecs)
        return ser

    @classmethod
    def _from_dict(cls, ser: Dict[str, Any]) -> 'InterDependencies':
        """
        Create an InterDependencies object from a dictionary
        """
        paramspecs = [ParamSpec._from_dict(sps) for sps in ser['paramspecs']]
        idp = cls(*paramspecs)
        return idp


class RunDescriber:
    """
    The object that holds the description of each run in the database. This
    object serialises itself to a string and is found under the run_description
    column in the runs table

    Extension of this object is planned for the future, for now it holds the
    parameter interdependencies. Extensions should be objects that can
    convert themselves to dictionary and added as attributes to the
    RunDescriber, such that the RunDescriber can iteratively convert its
    attributes when converting itself to dictionary.
    """

    def __init__(self, interdeps: InterDependencies):

        if not isinstance(interdeps, InterDependencies):
            raise ValueError('The interdeps arg must be of type: '
                             'InterDependencies. '
                             f'Got {type(interdeps)}.')

        self._version = 0
        self.interdeps = interdeps

    @property
    def version(self) -> int:
        return self._version

    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert this object into a dictionary. This method is intended to
        be used only by the serialization routines.
        """
        ser: Dict[str, Any] = {}
        ser['version'] = self._version
        ser['interdependencies'] = self.interdeps._to_dict()

        return ser

    @classmethod
    def _from_dict(cls, ser: Dict[str, Any]) -> 'RunDescriber':
        """
        Make a RunDescriber object from a dictionary. This method is
        intended to be used only by the deserialization routines.
        """

        return cls(InterDependencies._from_dict(ser['interdependencies']))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RunDescriber):
            return False
        if not self.interdeps == other.interdeps:
            return False
        return True
