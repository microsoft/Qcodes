"""
This module holds the objects that describe the intra-run relationships
between the parameters of that run. Most importantly, the information about
which parameters depend on each other is handled here.
"""
from copy import deepcopy
from functools import reduce
from collections import ChainMap
from typing import (Dict, Any, Tuple, Optional, FrozenSet, List, Set,
                    Type, Sequence)

from qcodes.dataset.param_spec import ParamSpecBase, ParamSpec

Dependencies = Dict[ParamSpecBase, Tuple[ParamSpecBase, ...]]
ParamNameTree = Dict[str, Tuple[str, ...]]
ErrorTuple = Tuple[Type[Exception], str]


class ParamSpecTree:
    """
    Class to represent a parameter tree, i.e. one parameter at the top (root)
    level depending on N leaf parameters.
    """

    def __init__(self, root: ParamSpecBase, *leafs: ParamSpecBase):
        if not isinstance(root, ParamSpecBase):
            raise ValueError('Root of ParamSpecTree must be of type '
                             'ParamSpecBase. Got root of type '
                             f'{type(root)}.')
        for leaf in leafs:
            if not isinstance(leaf, ParamSpecBase):
                raise ValueError('Leafs of ParamSpecTree must be of type '
                                 'ParamSpecBase. Got leaf of type '
                                 f'{type(leaf)}.')
        self.root = root
        self._leafs_set = set(leafs)
        self._leafs_tuple = leafs
        self._leaf_names = set(leaf.name for leaf in leafs)
        self._as_dict = {root: leafs}
        self._as_dict_str = {root.name: self.leaf_names}

    @property
    def leaf_names(self) -> Set[str]:
        return self._leaf_names

    @property
    def leafs(self) -> Set[ParamSpecBase]:
        return self._leafs_set

    def as_dict(self) -> Dict[ParamSpecBase, Tuple[ParamSpecBase, ...]]:
        return self._as_dict

    def as_dict_str(self) -> Dict[str, str]:
        return self._as_dict_str


class ParamSpecGrove:

    def __init__(self, *trees: ParamSpecTree):
        for tree in trees:
            if not isinstance(tree, ParamSpecTree):
                raise ValueError('ParamSpecGrove can only contain '
                                 'ParamSpecTrees, but received a/an '
                                 f'{type(tree)} instead.')
        # validate input
        root_names_raw = tuple(tree.root.name for tree in trees)
        root_names = set(root_names_raw)
        if len(root_names_raw) > len(root_names):
            raise ValueError('Supplied trees do not have unique root names')
        leaf_names: Set[str] = reduce(set.union,
                                      (tree.leaf_names for tree in trees),
                                      set())
        if root_names.intersection(leaf_names):
            raise ValueError('Cycles detected!')

        self._trees = set(trees)

        self._roots: Set[ParamSpecBase] = set(tree.root for tree in trees)
        self._leafs: Set[ParamSpecBase] = reduce(set.union,
                                                (tree.leafs for tree in trees),
                                                set())

        self._trees_as_dict = dict(ChainMap(*(t.as_dict()
                                              for t in self._trees)))
        self._trees_as_dict_str = dict(ChainMap(*(t.as_dict_str()
                                                  for t in self._trees)))

        self._trees_as_dict_inv = self._invert_grove()

    def as_dict(self) -> Dict[ParamSpecBase, Tuple[ParamSpecBase, ...]]:
        return self._trees_as_dict

    def as_dict_inv(self) -> Dict[ParamSpecBase, Tuple[ParamSpecBase, ...]]:
        return self._trees_as_dict_inv

    @property
    def roots(self) -> Set[ParamSpecBase]:
        return self._roots

    @property
    def leafs(self) -> Set[ParamSpecBase]:
        return self._leafs

    def extend(self, new: ParamSpecTree) -> 'ParamSpecGrove':
        return ParamSpecGrove(*self._trees, new)

    def _invert_grove(self) -> Dict[ParamSpecBase, Tuple[ParamSpecBase, ...]]:
        """
        Helper function to invert the dict representation of the grove.
        Will turn {A: (B, C)} into {B: (A,), C: (A,)}
        """
        grovedict = self._trees_as_dict

        indeps: Set[ParamSpecBase] = set()
        for indep_tup in grovedict.values():
            indeps.update(indep_tup)

        inverted: Dependencies = {}
        for indep in indeps:
            deps = tuple(ps for ps in grovedict if indep in grovedict[ps])
            inverted[indep] = deps

        return inverted

    def __add__(self, other_grove: 'ParamSpecGrove') -> 'ParamSpecGrove':
        if not isinstance(other_grove, ParamSpecGrove):
            raise TypeError(f'Must be ParamSpecGrove, not {type(other_grove)}')

        new_trees = deepcopy(self._trees).union(deepcopy(other_grove._trees))
        return ParamSpecGrove(*new_trees)

    def __getitem__(self, ps: ParamSpecBase) -> Tuple[ParamSpecBase, ...]:
        try:
            ret_val = self.as_dict()[ps]
            return ret_val
        except KeyError:
            try:
                ret_val = self.as_dict_inv()[ps]
                return ret_val
            except KeyError:
                pass
        raise KeyError(f'{ps} not found in this grove')

    def __contains__(self, ps: ParamSpecBase) -> bool:
        return ps in self._trees_as_dict or ps in self._trees_as_dict_inv


class DependencyError(Exception):
    def __init__(self,
                 param_name: str,
                 missing_params: Set[str],
                 *args):
        super().__init__(*args)
        self._param_name = param_name
        self._missing_params = missing_params

    def __str__(self) -> str:
        return (f'{self._param_name} has the following dependencies that are '
                f'missing: {self._missing_params}')


class InferenceError(Exception):
    def __init__(self,
                 param_name: str,
                 missing_params: Set[str],
                 *args):
        super().__init__(*args)
        self._param_name = param_name
        self._missing_params = missing_params

    def __str__(self):
        return (f'{self._param_name} has the following inferences that are '
                f'missing: {self._missing_params}')


class InterDependencies_:
    """
    Object containing a group of ParamSpecs and the information about their
    internal relations to each other
    """

    def __init__(self,
                 dependencies: Optional[Dependencies] = None,
                 inferences: Optional[Dependencies] = None,
                 standalones: Tuple[ParamSpecBase, ...] = ()):

        dependencies = dependencies or {}
        inferences = inferences or {}

        try:
            deps_gen = (ParamSpecTree(k, *v)
                            for k, v in dependencies.items())
            deps_grove = ParamSpecGrove(*deps_gen)
        except (AttributeError, ValueError, TypeError) as e:
            raise ValueError('Invalid dependencies') from e

        try:
            inffs_gen = (ParamSpecTree(k, *v)
                             for k, v in inferences.items())
            inffs_grove = ParamSpecGrove(*inffs_gen)
        except (AttributeError, ValueError, TypeError) as e:
            raise ValueError('Invalid inferences') from e

        link_error = self._validate_double_links(deps_grove, inffs_grove)
        if link_error is not None:
            error, mssg = link_error
            raise error(mssg)

        for ps in standalones:
            if not isinstance(ps, ParamSpecBase):
                base_error = TypeError('Standalones must be a sequence of '
                                       'ParamSpecs')

                raise ValueError('Invalid standalones') from base_error

        self._remove_duplicates(dependencies)
        self._remove_duplicates(inferences)

        self.dependencies: Dependencies = dependencies
        self.inferences: Dependencies = inferences
        self.standalones: FrozenSet[ParamSpecBase] = frozenset(standalones)

        self.deps_grove = deps_grove
        self.inffs_grove = inffs_grove

        # The object is now complete, but for convenience, we form some more
        # private attributes for easy look-up

        self._id_to_paramspec: Dict[str, ParamSpecBase] = {}
        for tree in (self.dependencies, self.inferences):
            for ps, ps_tup in tree.items():
                self._id_to_paramspec.update({ps.name: ps})
                self._id_to_paramspec.update({pst.name: pst
                                              for pst in ps_tup})
        for ps in self.standalones:
            self._id_to_paramspec.update({ps.name: ps})
        self._paramspec_to_id = {v: k for k, v in self._id_to_paramspec.items()}

        self._dependencies_inv: Dependencies = self._invert_tree(
            self.dependencies)
        self._inferences_inv: Dependencies = self._invert_tree(
            self.inferences)

        # Set operations are ~2x (or more) faster on strings than on hashable
        # ParamSpecBase objects, hence the need for use of the following
        # representation
        self._deps_names: ParamNameTree = self._tree_of_names(
            self.dependencies)
        self._infs_names: ParamNameTree = self._tree_of_names(
            self.inferences)

    @staticmethod
    def _tree_of_names(tree: Dependencies) -> ParamNameTree:
        """
        Helper function to convert a Dependencies-kind of tree where all
        ParamSpecBases are substituted with their ``.name`` s.
        Will turn {A: (B, C)} into {A.name: (B.name, C.name)}
        """
        name_tree: ParamNameTree = {}
        for ps, ps_tuple in tree.items():
            name_tree[ps.name] = tuple(p.name for p in ps_tuple)
        return name_tree

    @staticmethod
    def _invert_tree(tree: Dependencies) -> Dependencies:
        """
        Helper function to invert a Dependencies. Will turn {A: (B, C)} into
        {B: (A,), C: (A,)}
        """
        indeps: Set[ParamSpecBase] = set()
        for indep_tup in tree.values():
            indeps.update(indep_tup)

        inverted: Dependencies = {}
        for indep in indeps:
            deps = tuple(ps for ps in tree if indep in tree[ps])
            inverted[indep] = deps

        return inverted

    @staticmethod
    def _remove_duplicates(tree: Dependencies) -> None:
        """
        Helper function to remove duplicate entries from a Dependencies while
        preserving order. Will turn {A: (B, B, C)} into {A: (B, C)}
        """
        for ps, tup in tree.items():
            specs: List[ParamSpecBase] = []
            for p in tup:
                if p not in specs:
                    specs.append(p)
            tree[ps] = tuple(specs)

    @staticmethod
    def _validate_double_links(grove1: ParamSpecGrove,
                               grove2: ParamSpecGrove) -> Optional[ErrorTuple]:
        """
        Validate that two groves do not contain double links. A double link
        is a link between two nodes that exists in both groves. E.g. if the
        first grove has {A: (B, C)}, the second may not have {B: (A,)} etc.
        """
        for root, leafs in grove1.as_dict().items():
            for leaf in leafs:
                one_way_error = root in grove2.as_dict().get(leaf, ())
                other_way_error = leaf in grove2.as_dict().get(root, ())
                if one_way_error or other_way_error:
                    mssg = (f"Invalid dependencies/inferences. {root} and "
                            f"{leaf} have an ill-defined relationship.")
                    return (ValueError, mssg)

        return None

    def what_depends_on(self, ps: ParamSpecBase) -> Tuple[ParamSpecBase, ...]:
        """
        Return a tuple of the parameters that depend on the given parameter.
        Returns an empty tuple if nothing depends on the given parameter

        Args:
            ps: the parameter to look up

        Raises:
            ValueError if the parameter is not part of this object
        """
        if ps not in self.deps_grove and ps not in self.inffs_grove:
            raise ValueError(f'Unknown parameter: {ps}')
        try:
            return self.deps_grove[ps]
        except KeyError:
            return ()

    def what_is_inferred_from(self,
                              ps: ParamSpecBase) -> Tuple[ParamSpecBase, ...]:
        """
        Return a tuple of the parameters thatare inferred from the given
        parameter. Returns an empty tuple if nothing is inferred from the given
        parameter

        Args:
            ps: the parameter to look up

        Raises:
            ValueError if the parameter is not part of this object
        """
        if ps not in self.deps_grove and ps not in self.inffs_grove:
            raise ValueError(f'Unknown parameter: {ps}')
        try:
            return self.inffs_grove[ps]
        except KeyError:
            return ()

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

    @property
    def names(self) -> Tuple[str, ...]:
        """
        Return all the names of the parameters of this instance
        """
        return tuple(self._id_to_paramspec.keys())

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

    def extend(
            self,
            dependencies: Optional[Dependencies] = None,
            inferences: Optional[Dependencies] = None,
            standalones: Tuple[ParamSpecBase, ...] = ()) -> 'InterDependencies_':
        """
        Create a new InterDependencies_ object that is an extension of this
        instance with the provided input
        """

        dependencies = {} if dependencies is None else dependencies
        inferences = {} if inferences is None else inferences

        # first step: remove parameters from standalones if they no longer
        # stand alone

        depended_on = (ps for tup in dependencies.values() for ps in tup)
        inferred_from = (ps for tup in inferences.values() for ps in tup)

        standalones_mut = set(deepcopy(self.standalones))
        standalones_mut = (standalones_mut.difference(set(dependencies))
                                          .difference(set(inferences))
                                          .difference(set(depended_on))
                                          .difference(set(inferred_from)))

        # then update deps and inffs
        new_deps = deepcopy(self.dependencies)
        for ps in set(dependencies).intersection(set(new_deps)):
            new_deps[ps] = tuple(set(list(new_deps[ps]) +
                                     list(dependencies[ps])))
        for ps in set(dependencies).difference(set(new_deps)):
            new_deps.update({deepcopy(ps): dependencies[ps]})

        new_inffs = deepcopy(self.inferences.copy())
        for ps in set(inferences).intersection(set(new_inffs)):
            new_inffs[ps] = tuple(set(list(new_inffs[ps]) +
                                      list(inferences[ps])))
        for ps in set(inferences).difference(set(new_inffs)):
            new_inffs.update({deepcopy(ps): inferences[ps]})

        # add new standalones
        new_standalones = tuple(standalones_mut.union(set(standalones)))

        new_idps =  InterDependencies_(dependencies=new_deps,
                                       inferences=new_inffs,
                                       standalones=new_standalones)

        return new_idps

    def remove(self, parameter: ParamSpecBase) -> 'InterDependencies_':
        """
        Create a new InterDependencies_ object that is similar to this
        instance, but has the given parameter removed.
        """
        if parameter not in self:
            raise ValueError(f'Unknown parameter: {parameter}.')

        if parameter in self._dependencies_inv:
            raise ValueError(f'Cannot remove {parameter.name}, other '
                                'parameters depend on it.')
        if parameter in self._inferences_inv:
            raise ValueError(f'Cannot remove {parameter.name}, other '
                                'parameters are inferred from it.')

        if parameter in self.standalones:
            new_standalones = tuple(deepcopy(self.standalones).
                                    difference({parameter}))
            new_deps = deepcopy(self.dependencies)
            new_inffs = deepcopy(self.inferences)
        elif parameter in self.dependencies or parameter in self.inferences:
            new_deps = deepcopy(self.dependencies)
            new_inffs = deepcopy(self.inferences)
            # figure out whether removing this parameter will make any other
            # parameters standalone
            new_standalones_l = []
            old_standalones = deepcopy(self.standalones)
            for indep in self.dependencies.get(parameter, []):
                if not(indep in self._inferences_inv or
                       indep in self.inferences):
                    new_standalones_l.append(indep)
            for basis in self.inferences.get(parameter, []):
                if not(basis in self._dependencies_inv or
                       basis in self.dependencies):
                    new_standalones_l.append(basis)
            new_deps.pop(parameter, None)
            new_inffs.pop(parameter, None)
            new_standalones = tuple(set(new_standalones_l)
                                    .union(old_standalones))

        idps = InterDependencies_(dependencies=new_deps, inferences=new_inffs,
                                  standalones=new_standalones)
        return idps

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
        params = set(p.name for p in parameters)

        for param in params:
            ps = self._id_to_paramspec.get(param, None)
            if ps is None:
                raise ValueError(f'Unknown parameter: {param}')

            deps = set(self._deps_names.get(param, ()))
            missing_deps = deps.difference(params)
            if missing_deps:
                raise DependencyError(param, missing_deps)

            inffs = set(self._infs_names.get(param, ()))
            missing_inffs = inffs.difference(params)
            if missing_inffs:
                raise InferenceError(param, missing_inffs)

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

        def sorter(inp: Any) -> List[Any]:
            return sorted(inp, key=lambda ps: ps.name)

        if not isinstance(other, InterDependencies_):
            return False

        for dic in ['dependencies', 'inferences']:
            our_keys = sorter(getattr(self, dic).keys())
            their_keys = sorter(getattr(other, dic).keys())
            if our_keys != their_keys:
                return False
            for key in our_keys:
                our_values = sorter(getattr(self, dic)[key])
                their_values = sorter(getattr(other, dic)[key])
                if our_values != their_values:
                    return False

        if not self.standalones == other.standalones:
            return False

        return True

    def __contains__(self, ps: ParamSpecBase) -> bool:
        return ps in self._paramspec_to_id

    def __getitem__(self, name: str) -> ParamSpecBase:
        return self._id_to_paramspec[name]


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
        ours = sorted(self.paramspecs, key=lambda ps: ps.name)
        theirs = sorted(other.paramspecs, key=lambda ps: ps.name)
        if not ours == theirs:
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
