from typing import (Dict, Any, Tuple, Optional, FrozenSet, List,
                    cast, Type, Sequence)

from qcodes.dataset.param_spec import ParamSpecBase, ParamSpec

ParamSpecTree = Dict[ParamSpecBase, Tuple[ParamSpecBase, ...]]
ErrorTuple = Tuple[Type[Exception], str]


class DependencyError(Exception):
    pass


class InferenceError(Exception):
    pass


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

        self._id_to_paramspec: Dict[str, ParamSpecBase] = {}
        for tree in (self.dependencies, self.inferences):
            for ps, ps_tup in tree.items():
                self._id_to_paramspec.update({self._id(ps): ps})
                self._id_to_paramspec.update({self._id(pst): pst
                                              for pst in ps_tup})
        for ps in self.standalones:
            self._id_to_paramspec.update({self._id(ps): ps})
        self._paramspec_to_id = {v: k for k, v in self._id_to_paramspec.items()}

    @staticmethod
    def _id(ps: ParamSpecBase) -> str:
        """
        Function to generate an ID from/for a ParamSpecBase
        """
        return ps.name

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

    def serialize(self) -> Dict[str, Any]:
        """
        Write out this object as a dictionary
        """
        output: Dict[str, Any] = {}
        output['parameters'] = {key: value.serialize() for key, value in
                                self._id_to_paramspec.items()}

        trees = ['dependencies', 'inferences']
        for tree in trees:
            output[tree] = {}
            for key, value in getattr(self, tree).items():
                ps_id = self._paramspec_to_id[key]
                ps_ids = [self._paramspec_to_id[ps] for ps in value]
                output[tree].update({ps_id: ps_ids})

        output['standalones'] = [self._paramspec_to_id[ps] for ps in
                                 self.standalones]

        return output

    @property
    def paramspecs(self) -> Tuple[ParamSpecBase, ...]:
        """
        Return the ParamSpecBase objects of this instance
        """
        return tuple(self._paramspec_to_id.keys())

    def _extend_with_paramspec(self, ps: ParamSpec) -> 'InterDependencies_':
        """
        Create a new InterDependencies_ object extended with the provided
        ParamSpec. A helper function for DataSet's add_parameter function.
        Note that this function will only work as expected if the ParamSpecs
        are extended into the InterDependencies_ in the "logical order", i.e.
        independent ParamSpecs before dependent ones.
        """
        base_ps = ps.base_version()

        old_standalones = set(self.standalones.copy())
        new_standalones: Tuple[ParamSpecBase, ...]

        if len(ps.depends_on_) > 0:
            deps_list = [self._id_to_paramspec[name] for name in ps.depends_on_]
            new_deps = {base_ps: tuple(deps_list)}
            old_standalones = old_standalones.difference(set(deps_list))
        else:
            new_deps = {}

        if len(ps.inferred_from_) > 0:
            inffs_list = [self._id_to_paramspec[name]
                          for name in ps.inferred_from_]
            new_inffs = {base_ps: tuple(inffs_list)}
            old_standalones = old_standalones.difference(set(inffs_list))
        else:
            new_inffs = {}

        if new_deps == new_inffs == {}:
            new_standalones = (base_ps,)
        else:
            old_standalones = old_standalones.difference({base_ps})
            new_standalones = ()

        new_deps.update(self.dependencies.copy())
        new_inffs.update(self.inferences.copy())
        new_standalones = tuple(list(new_standalones) + list(old_standalones))

        new_idps = InterDependencies_(dependencies=new_deps,
                                      inferences=new_inffs,
                                      standalones=new_standalones)

        return new_idps

    def validate_subset(self, parameters: Sequence[ParamSpecBase]) -> None:
        """
        Validate that the given parameters form a valid subset of the
        parameters of this instance, meaning that all the given parameters are
        actually found in this instance and that there are no missing
        dependencies/inferences.

        Args:
            params: The collection of ParamSpecBases to validate

        Raises:
            DependencyError, if a dependency is missing
            InferenceError, if an inference is missing
        """
        params = set(parameters)

        for param in params:
            deps = self.dependencies.get(param, ())
            if set(deps).difference(params):
                # pylint: disable=unused-variable
                missing_names = [p.name for p in set(deps).difference(params)]
                raise DependencyError(f'{param.name} has the following '
                                      'dependencies that are missing: '
                                      'f{missing_names}')

            inffs = self.inferences.get(param, ())
            if set(inffs).difference(params):
                # pylint: disable=unused-variable
                missing_names = [p.name for p in set(inffs).difference(params)]
                raise InferenceError(f'{param.name} has the following '
                                     'inferences that are missing: '
                                     'f{missing_names}')

    @classmethod
    def deserialize(cls, ser: Dict[str, Any]) -> 'InterDependencies_':
        """
        Construct an InterDependencies_ object from a serialization of such
        an object
        """
        params = ser['parameters']
        deps = {}
        for key, value in ser['dependencies'].items():
            deps_key = ParamSpecBase.deserialize(params[key])
            deps_vals = tuple(ParamSpecBase.deserialize(params[val]) for
                              val in value)
            deps.update({deps_key: deps_vals})

        inffs = {}
        for key, value in ser['inferences'].items():
            inffs_key = ParamSpecBase.deserialize(params[key])
            inffs_vals = tuple(ParamSpecBase.deserialize(params[val]) for
                              val in value)
            inffs.update({inffs_key: inffs_vals})

        stdls = tuple(ParamSpecBase.deserialize(params[ps_id]) for
                      ps_id in ser['standalones'])

        return cls(dependencies=deps, inferences=inffs, standalones=stdls)

    def __repr__(self) -> str:
        rep = (f"InterDependencies_(dependencies={self.dependencies}, "
               f"inferences={self.inferences}, "
               f"standalones={self.standalones})")
        return rep

    def __eq__(self, other):
        if not isinstance(other, InterDependencies_):
            return False
        if not self.dependencies == other.dependencies:
            return False
        if not self.inferences == other.inferences:
            return False
        if not self.standalones == other.standalones:
            return False

        return True


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


def new_to_old(idps: InterDependencies_) -> InterDependencies:
    """
    Create a new InterDependencies object (old style) from an existing
    InterDependencies_ object (new style). Leaves the original object
    unchanged. Only meant to be used for ensuring backwards-compatibility
    until we update sqlite_base to forget about ParamSpecs
    """

    paramspecs: Dict[str, ParamSpec] = {}

    # first the independent parameters
    for indeps in idps.dependencies.values():
        for indep in indeps:
            paramspecs.update({indep.name: ParamSpec(name=indep.name,
                                                     paramtype=indep.type,
                                                     label=indep.label,
                                                     unit=indep.unit)})

    for inffs in idps.inferences.values():
        for inff in inffs:
            paramspecs.update({inff.name: ParamSpec(name=inff.name,
                                                     paramtype=inff.type,
                                                     label=inff.label,
                                                     unit=inff.unit)})

    for ps_base in idps._paramspec_to_id.keys():
        paramspecs.update({ps_base.name: ParamSpec(name=ps_base.name,
                                                   paramtype=ps_base.type,
                                                   label=ps_base.label,
                                                   unit=ps_base.unit)})

    for ps, indeps in idps.dependencies.items():
        for indep in indeps:
            paramspecs[ps.name]._depends_on.append(indep.name)
    for ps, inffs in idps.inferences.items():
        for inff in inffs:
            paramspecs[ps.name]._inferred_from.append(inff.name)

    return InterDependencies(*tuple(paramspecs.values()))
