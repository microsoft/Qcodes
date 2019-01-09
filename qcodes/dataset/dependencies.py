from typing import Dict, Any, Set, Union

from qcodes.dataset.param_spec import ParamSpec


class UnknownParameterError(Exception):
    pass


class MissingDependencyError(Exception):
    pass


class DuplicateParameterError(Exception):
    pass


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
    def _missing_dependencies(*params: ParamSpec) -> Set[str]:
        """
        Return a list of the names of the missing dependencies paramspecs
        (including both depends_on and inferred_from)

        Note that multiple occurences of the same parameter is allowed in the
        input.
        """

        needed = set()
        present = set()

        for param in params:
            if param.name in needed:
                needed.remove(param.name)
            present.add(param.name)
            param_deps = [sp for sp in param.depends_on.split(', ')
                          if sp != '']
            param_infs = [sp for sp in param.inferred_from.split(', ')
                          if sp != '']
            for must_have in [param_deps, param_infs]:
                also_needed = set([sp for sp in must_have if sp not in present])
                needed = needed.union(also_needed)

        return needed

    @staticmethod
    def _are_dependencies_met(*params: ParamSpec) -> bool:
        """
        Determine whether all dependencies are met, i.e. that for every
        parameter that has dependencies, those dependencies are also present.

        Note that multiple occurences of the same parameter is allowed
        """
        needed = InterDependencies._missing_dependencies(*params)

        if len(needed) > 0:
            return False
        else:
            return True

    def validate_subset(self, *params: Union[str, ParamSpec]) -> None:
        """
        Validate that the given input is a valid subset of the parameters
        of this instance. A valid subset is a non-strict subset that has no
        missing dependencies.
        """

        error_unknown_param: bool = False

        # Step 1: validate that all params are known to this instance
        # (and turn all strings into paramspecs for further validation)

        paramspecs = []
        for param in params:
            if isinstance(param, str):
                if not param in (p.name for p in self.paramspecs):
                    error_unknown_param = True
                else:
                    tp = [p for p in self.paramspecs if p.name == param]
                    paramspecs += tp
            elif isinstance(param, ParamSpec):
                if not param in self.paramspecs:
                    error_unknown_param = True
                else:
                    paramspecs.append(param)
            else:
                raise ValueError(f'Received parameter of invalid type: '
                                 f'{param}, ({type(param)}). Parameters '
                                 'must be strings or ParamSpecs.')
            if error_unknown_param:
                raise UnknownParameterError(f'Unknown parameter: {param}')

        # Step 2: Validate that there are no duplicates
        # NOTE: this would be easier with hashable paramspecs

        param_names = tuple(p.name for p in paramspecs)
        if len(set(param_names)) != len(param_names):
            duplicates = set([p for p in param_names
                              if param_names.count(p) > 1])
            raise DuplicateParameterError(f'Duplicate parameter(s): '
                                           '{duplicates}')

        # Step 3: validate that there are no missing dependencies

        missing = self._missing_dependencies(*paramspecs)
        if missing != set():
            raise MissingDependencyError(f'Missing parameter(s): {missing}')

    @classmethod
    def deserialize(cls, ser: Dict[str, Any]) -> 'InterDependencies':
        """
        Create an InterDependencies object from a serialization of an
        instance
        """
        paramspecs = [ParamSpec.deserialize(sps) for sps in ser['paramspecs']]
        idp = cls(*paramspecs)
        return idp
