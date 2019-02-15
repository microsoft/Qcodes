from typing import (Dict, Any, Tuple, Sequence, Optional, FrozenSet, List,
                    cast, Type)

from qcodes.dataset.param_spec import ParamSpecBase, ParamSpec

ParamSpecTree = Dict[ParamSpecBase, Tuple[ParamSpecBase, ...]]
ErrorTuple = Tuple[Type[Exception], str]


class InterDependencies_:
    """
    Object containing a group of ParamSpecs and the information about their
    internal relations to each other
    """

    def __init__(self,
                 dependencies: Optional[ParamSpecTree] = None,
                 inferences: Optional[ParamSpecTree] = None,
                 standalones: Tuple[ParamSpecBase, ...] = ()):

        dependencies = dependencies or {}
        inferences = inferences or {}

        deps_error = self.validate_paramspectree(dependencies)
        if not deps_error is None:
            old_error = cast(type, deps_error[0])
            self._raise_from(ValueError, 'Invalid dependencies',
                             old_error, deps_error[1])

        inffs_error = self.validate_paramspectree(inferences)
        if not inffs_error is None:
            old_error = cast(type, inffs_error[0])
            self._raise_from(ValueError, 'Invalid inferences',
                             old_error, inffs_error[1])

        for ps in standalones:
            if not isinstance(ps, ParamSpecBase):
                error: type = TypeError
                message: str = ('Standalones must be a sequence of '
                                'ParamSpecs')
                self._raise_from(ValueError, 'Invalid standalones',
                                 error, message)

        self.dependencies: ParamSpecTree = dependencies
        self.inferences: ParamSpecTree = inferences
        self.standalones: FrozenSet[ParamSpecBase] = frozenset(standalones)

    @staticmethod
    def _raise_from(new_error: Type[Exception], new_mssg: str,
                    old_error: Type[Exception], old_mssg: str) -> None:
        """
        Helper function to raise an error with a cause in a way that our test
        suite can digest
        """
        try:
            raise old_error(old_mssg)
        except old_error as e:
            raise new_error(new_mssg) from e

    @staticmethod
    def validate_paramspectree(
        paramspectree: ParamSpecTree) -> Optional[ErrorTuple]:
        """
        Validate a ParamSpecTree. Apart from adhering to the type, a
        ParamSpecTree must not have any cycles.

        Returns:
            A tuple with an exception type and an error message or None, if
            the paramtree is valid
        """

        # Validate the type

        if not isinstance(paramspectree, dict):
            return (TypeError, 'ParamSpecTree must be a dict')

        for key, values in paramspectree.items():
            if not isinstance(key, ParamSpecBase):
                return (TypeError, 'ParamSpecTree must have ParamSpecs as keys')
            if not isinstance(values, tuple):
                return (TypeError, 'ParamSpecTree must have tuple values')
            for value in values:
                if not isinstance(value, ParamSpecBase):
                    return (TypeError,
                            ('ParamSpecTree can only have tuples of '
                             'ParamSpecs as values'))

        # check for cycles

        roots = set(paramspectree.keys())
        leafs = set(ps for tup in paramspectree.values() for ps in tup)

        if roots.intersection(leafs) != set():
            return (ValueError, 'ParamSpecTree can not have cycles')

        return None


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

    @classmethod
    def deserialize(cls, ser: Dict[str, Any]) -> 'InterDependencies':
        """
        Create an InterDependencies object from a serialization of an
        instance
        """
        paramspecs = [ParamSpec.deserialize(sps) for sps in ser['paramspecs']]
        idp = cls(*paramspecs)
        return idp


def old_to_new(idps: InterDependencies) -> InterDependencies_:
    """
    Create a new InterDependencies_ object (new style) from an existing
    InterDependencies object (old style). Leaves the original object unchanged.
    Incidentally, this function can serve as a validator of the original object
    """
    namedict: Dict[str, ParamSpec] = {ps.name: ps for ps in idps.paramspecs}

    dependencies = {}
    inferences = {}
    standalones_mut = []
    root_paramspecs: List[ParamSpecBase] = []

    for ps in idps.paramspecs:
        deps = tuple(namedict[n].base_version() for n in ps.depends_on_)
        inffs = tuple(namedict[n].base_version() for n in ps.inferred_from_)
        if len(deps) > 0:
            dependencies.update({ps.base_version(): deps})
            root_paramspecs += list(deps)
        if len(inffs) > 0:
            inferences.update({ps.base_version(): inffs})
            root_paramspecs += list(inffs)
        if len(deps) == len(inffs) == 0:
            standalones_mut.append(ps.base_version())

    standalones = tuple(set(standalones_mut).difference(set(root_paramspecs)))

    idps_ = InterDependencies_(dependencies=dependencies,
                               inferences=inferences,
                               standalones=standalones)
    return idps_

