from typing import Dict, Any, Tuple, Sequence

from qcodes.dataset.param_spec import ParamSpec

ParamSpecTree = Dict[ParamSpec, Tuple[ParamSpec, ...]]


class InterDependencies_:
    """
    Object containing a group of ParamSpecs and the information about their
    internal relations to each other
    """

    error_codes = {1: TypeError('ParamSpecTree must be a dict'),
                   2: TypeError('ParamSpecTree must have ParamSpecs as keys'),
                   3: TypeError('ParamSpecTree must have tuple values'),
                   4: TypeError('ParamSpecTree can only have tuples of '
                                'ParamSpecs as values'),
                   5: ValueError('ParamSpecTree can not have cycles')}

    def __init__(self,
                 dependencies: ParamSpecTree = {},
                 inferences: ParamSpecTree = {},
                 standalones: Tuple[ParamSpec, ...] = ()):

        deps_code = self.validate_paramspectree(dependencies)
        if not deps_code == 0:
            e = self.error_codes[deps_code]
            raise ValueError('Invalid dependencies') from e

        inffs_code = self.validate_paramspectree(inferences)
        if not inffs_code == 0:
            e = self.error_codes[inffs_code]
            raise ValueError('Invalid inferences') from e

        for ps in standalones:
            if not isinstance(ps, ParamSpec):
                e = TypeError('Standalones must be a sequence of ParamSpecs')
                raise ValueError('Invalid standalones') from e

        self.dependencies = dependencies
        self.inferences = inferences
        self.standalones = standalones

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
            if not isinstance(key, ParamSpec):
                return 2
            if not isinstance(values, tuple):
                return 3
            for value in values:
                if not isinstance(value, ParamSpec):
                    return 4

        # check for cycles

        roots = set(paramspectree.keys())
        leafs = set(ps for tup in paramspectree.items() for ps in tup)

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
