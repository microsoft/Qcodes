from typing import Dict, Any, Tuple, Sequence, Optional, Set

from qcodes.dataset.param_spec import ParamSpecBase, ParamSpec

ParamSpecTree = Dict[ParamSpecBase, Tuple[ParamSpecBase, ...]]


class InterDependencies_:
    """
    Object containing a group of ParamSpecs and the information about their
    internal relations to each other
    """

    error_codes = {1: {'error': TypeError,
                       'message': 'ParamSpecTree must be a dict'},
                   2: {'error': TypeError,
                       'message': 'ParamSpecTree must have ParamSpecs as keys'},
                   3: {'error': TypeError,
                       'message': 'ParamSpecTree must have tuple values'},
                   4: {'error': TypeError,
                       'message': ('ParamSpecTree can only have tuples of '
                                   'ParamSpecs as values')},
                   5: {'error': ValueError,
                       'message': 'ParamSpecTree can not have cycles'}}

    def __init__(self,
                 dependencies: Optional[ParamSpecTree] = None,
                 inferences: Optional[ParamSpecTree] = None,
                 standalones: Tuple[ParamSpecBase, ...] = ()):

        dependencies = dependencies or {}
        inferences = inferences or {}

        deps_code = self.validate_paramspectree(dependencies)
        if not deps_code == 0:
            err = self.error_codes[deps_code]
            self._raise_from(ValueError, 'Invalid dependencies',
                             err['error'], err['message'])

        inffs_code = self.validate_paramspectree(inferences)
        if not inffs_code == 0:
            err = self.error_codes[inffs_code]
            self._raise_from(ValueError, 'Invalid inferences',
                             err['error'], err['message'])

        for ps in standalones:
            if not isinstance(ps, ParamSpecBase):
                err = {'error': TypeError,
                       'message': ('Standalones must be a sequence of '
                                   'ParamSpecs')}
                self._raise_from(ValueError, 'Invalid standalones',
                                 err['error'], err['message'])

        self.dependencies: ParamSpecTree = dependencies
        self.inferences: ParamSpecTree = inferences
        self.standalones: Set[ParamSpecBase] = set(standalones)

    @staticmethod
    def _raise_from(new_error: Exception, new_mssg: str,
                    old_error: Exception, old_mssg: str) -> None:
        """
        Helper function to raise an error with a cause in a way that our test
        suite can digest
        """
        try:
            raise old_error(old_mssg)
        except old_error as e:
            raise new_error(new_mssg) from e

    @staticmethod
    def validate_paramspectree(paramspectree: ParamSpecTree) -> int:
        """
        Validate a ParamSpecTree. Apart from adhering to the type, a
        ParamSpecTree must not have any cycles.

        Returns:
            An error code describing what is wrong with the tree (or 0, if the
            tree is valid).
            1: The passed tree is not a dict
            2: The dict's keys are not all ParamSpecs
            3: The dict's values are not all tuples
            4: The dict's values are tuples containing something that
            is not a ParamSpec
            5: There is at least one ParamSpec that is present as a key and
            inside a value (i.e there is a cycle in the tree)
        """

        # Validate the type

        if not isinstance(paramspectree, dict):
            return 1

        for key, values in paramspectree.items():
            if not isinstance(key, ParamSpecBase):
                return 2
            if not isinstance(values, tuple):
                return 3
            for value in values:
                if not isinstance(value, ParamSpecBase):
                    return 4

        # check for cycles

        roots = set(paramspectree.keys())
        leafs = set(ps for tup in paramspectree.values() for ps in tup)

        if roots.intersection(leafs) != set():
            return 5

        return 0


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
    root_paramspecs = []

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

    standalones = (set(standalones_mut)
                       .difference(set(root_paramspecs)))

    idps_ = InterDependencies_(dependencies=dependencies,
                               inferences=inferences,
                               standalones=standalones)
    return idps_

