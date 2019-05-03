"""
This module holds the objects that describe the intra-run relationships
between the parameters of that run. Most importantly, the information about
which parameters depend on each other is handled here.
"""
from copy import deepcopy
from functools import reduce
from collections import ChainMap
from typing import (Dict, Any, Tuple, Optional, FrozenSet, List, Set,
                    Type, Sequence, Union, cast)

from qcodes.dataset.param_spec import ParamSpecBase, ParamSpec

SerializedTree = Dict[str, Union[Dict[str, str], Tuple[Dict[str, str], ...]]]
SerializedGrove = Tuple[SerializedTree, ...]
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
        if len(leafs) > len(self._leaf_names):
            raise ValueError('Supplied leafs do not have unique names; '
                             f'got {[l.name for l in leafs]}')

        self._as_dict = {root: leafs}
        self._as_dict_str = {root.name: self.leaf_names}
        self._is_stub = not(bool(self._leafs_set))

    @property
    def leaf_names(self) -> Set[str]:
        return self._leaf_names

    @property
    def leafs(self) -> Tuple[ParamSpecBase, ...]:
        return self._leafs_tuple

    @property
    def is_stub(self) -> bool:
        return self._is_stub

    def as_set_of_stubs(self) -> Set['ParamSpecTree']:
        return set(ParamSpecTree(p) for p in (self.root,) + self._leafs_tuple)

    def as_dict(self) -> Dict[ParamSpecBase, Tuple[ParamSpecBase, ...]]:
        return self._as_dict

    def as_dict_str(self) -> Dict[str, Set[str]]:
        return self._as_dict_str

    def serialize(self) -> SerializedTree:
        return {"root": self.root.serialize(),
                "leafs": tuple(leaf.serialize() for leaf in self._leafs_tuple)}

    @classmethod
    def deserialize(cls, ser: SerializedTree):
        deser = ParamSpecBase.deserialize
        root_ser = cast(Dict[str, str], ser['root'])
        root = deser(root_ser)
        leafs_ser = cast(Tuple[Dict[str, str], ...], ser['leafs'])
        leafs = tuple(deser(ser_leaf) for ser_leaf in leafs_ser)
        return cls(root, *leafs)

    def __repr__(self) -> str:
        rpr = f'ParamSpecTree({self.root}'
        if self.is_stub:
            rpr += ')'
        else:
            rpr += f', {", ".join([str(l) for l in self._leafs_tuple])})'
        return rpr

    def __eq__(self, other) -> bool:
        if not isinstance(other, ParamSpecTree):
            return False
        if self.root != other.root:
            return False
        if self._leafs_tuple != other._leafs_tuple:
            return False
        return True

    def __hash__(self) -> int:
        return hash((self.root, self._leafs_tuple))

    def __iter__(self):
        yield from (self.root,) + self._leafs_tuple


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

        self._trees = trees

        self._roots: Tuple[ParamSpecBase, ...]
        self._roots = tuple(tree.root for tree in trees)
        self._leafs: Tuple[ParamSpecBase, ...]
        self._leafs = reduce(tuple.__add__,
                             tuple(tree.leafs for tree in trees), ())

        self._trees_as_dict = dict(ChainMap(*(t.as_dict()
                                              for t in self._trees)))
        self._trees_as_dict_str = dict(ChainMap(*(t.as_dict_str()
                                                  for t in self._trees)))

        self._trees_as_dict_inv = self._invert_grove()

        self._names_to_paramspec = {
            ps.name: ps for ps in self._roots + self._leafs}

        self._stubs: Tuple[ParamSpecTree, ...] = tuple(
            t for t in self._trees if t.is_stub)

    def as_dict(self) -> Dict[ParamSpecBase, Tuple[ParamSpecBase, ...]]:
        return self._trees_as_dict

    def as_dict_inv(self) -> Dict[ParamSpecBase, Tuple[ParamSpecBase, ...]]:
        return self._trees_as_dict_inv

    def as_dict_str(self) -> Dict[str, Set[str]]:
        return self._trees_as_dict_str

    @property
    def trees(self) -> Tuple[ParamSpecTree, ...]:
        return self._trees

    @property
    def roots(self) -> Tuple[ParamSpecBase, ...]:
        return self._roots

    @property
    def leafs(self) -> Tuple[ParamSpecBase, ...]:
        return self._leafs

    def extend(self, new: ParamSpecTree) -> 'ParamSpecGrove':
        return ParamSpecGrove(*self._trees, new)

    def remove_stubs(self, *stubs: ParamSpecTree) -> 'ParamSpecGrove':
        for stub in stubs:
            if stub not in self._stubs:
                raise ValueError(f'Cannot remove stub {stub}, not a stub '
                                 'of this grove.')
        new_trees = tuple(t for t in self._trees if t not in stubs)
        return ParamSpecGrove(*new_trees)

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

    def serialize(self) -> SerializedGrove:
        return tuple(t.serialize() for t in self._trees)

    @classmethod
    def deserialize(cls, ser_trees: SerializedGrove) -> 'ParamSpecGrove':
        return cls(*(ParamSpecTree.deserialize(ser_t) for ser_t in ser_trees))

    def __add__(self, other_grove: 'ParamSpecGrove') -> 'ParamSpecGrove':
        if not isinstance(other_grove, ParamSpecGrove):
            raise TypeError(f'Must be ParamSpecGrove, not {type(other_grove)}')

        new_trees = (set(deepcopy(self._trees))
                     .union(set(deepcopy(other_grove._trees))))
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

    def __iter__(self):
        yield from self._trees

    def __eq__(self, other) -> bool:
        if not isinstance(other, ParamSpecGrove):
            return False
        if self._trees != other._trees:
            return False
        return True

    def __repr__(self) -> str:
        str_trees = [str(t) for t in self._trees]
        return 'ParamSpecGrove(' + ', '.join(str_trees) + ')'


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
                 dependencies: Sequence[ParamSpecTree] = (),
                 inferences: Sequence[ParamSpecTree] = ()):

        deps_grove = ParamSpecGrove(*dependencies)
        inffs_grove = ParamSpecGrove(*inferences)

        link_error = self._validate_double_links(deps_grove, inffs_grove)
        if link_error is not None:
            error, mssg = link_error
            raise error(mssg)

        self.dependencies: ParamSpecGrove = deps_grove
        self.inferences: ParamSpecGrove = inffs_grove

        # The object is now complete, but for convenience, we form some more
        # private attributes for easy look-up

        self._id_to_paramspec: Dict[str, ParamSpecBase] = {}
        for tree in (self.dependencies.as_dict(), self.inferences.as_dict()):
            for ps, ps_tup in tree.items():
                self._id_to_paramspec.update({ps.name: ps})
                self._id_to_paramspec.update({pst.name: pst
                                              for pst in ps_tup})

        self._dependencies_inv: Dependencies = self.dependencies.as_dict_inv()
        self._inferences_inv: Dependencies = self.inferences.as_dict_inv()

        # Set operations are ~2x (or more) faster on strings than on hashable
        # ParamSpecBase objects, hence the need for use of the following
        # representation
        self._deps_names = self.dependencies.as_dict_str()
        self._infs_names = self.inferences.as_dict_str()

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
        if ps not in self.dependencies and ps not in self.inferences:
            raise ValueError(f'Unknown parameter: {ps}')
        try:
            return self.dependencies[ps]
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
        if ps not in self.dependencies and ps not in self.inferences:
            raise ValueError(f'Unknown parameter: {ps}')
        try:
            return self.inferences[ps]
        except KeyError:
            return ()

    @property
    def paramspecs(self) -> Tuple[ParamSpecBase, ...]:
        """
        Return the ParamSpecBase objects of this instance
        """
        paramspecs = reduce(lambda x, y: x + y, [self.dependencies.roots,
                                                 self.dependencies.leafs,
                                                 self.inferences.roots,
                                                 self.inferences.leafs])
        return paramspecs

    @property
    def names(self) -> Tuple[str, ...]:
        """
        Return all the names of the parameters of this instance
        """
        return tuple(self._id_to_paramspec.keys())

    def extend_with_tree(self,
                         new_tree: ParamSpecTree,
                         deps_or_inffs: str
                         ) -> 'InterDependencies_':
        """
        Create a new InterDependencies_ object that is an extension of this
        instance with the provided input

        Args:
            new_tree: The tree to extend one of the two groves with
            deps_or_inffs: A string, either 'deps' or 'inffs' telling declaring
                which grove to extend
        """

        # first step: remove the stubs that are to become leafs or non-stub
        # roots
        stubs_set = set(self.dependencies._stubs)
        stubs_to_remove = stubs_set.intersection(new_tree.as_set_of_stubs())

        new_deps_grove = self.dependencies.remove_stubs(*stubs_to_remove)

        # second step: extend the relevant collection of trees
        #  with the new tree
        if deps_or_inffs == 'deps':
            new_deps = tuple(new_deps_grove._trees) + (new_tree,)
            new_inffs = tuple(self.inferences._trees)
        elif deps_or_inffs == 'inffs':
            new_deps = tuple(new_deps_grove._trees)
            new_inffs = tuple(self.inferences._trees) + (new_tree,)
        else:
            raise ValueError(f'Invalid deps_or_inffs string: {deps_or_inffs}')

        return InterDependencies_(dependencies=new_deps,
                                  inferences=new_inffs)

    def remove(self, parameter: ParamSpecBase) -> 'InterDependencies_':
        """
        Create a new InterDependencies_ object that is similar to this
        instance, but has the given parameter removed.
        """
        if parameter not in self:
            raise ValueError(f'Unknown parameter: {parameter}.')

        if parameter in self.dependencies.leafs:
                raise ValueError(f'Cannot remove {parameter.name}, other '
                                 'parameters depend on it.')

        if parameter in self.inferences.leafs:
            raise ValueError(f'Cannot remove {parameter.name}, other '
                                'parameters are inferred from it.')

        orphaned_leafs: List[ParamSpecBase] = []
        deps_trees_to_kill = []
        inffs_trees_to_kill = []

        for tree in self.dependencies:
            if parameter == tree.root:
                orphaned_leafs += list(l for l in tree.leafs if
                                       l not in self.inferences.roots)
                deps_trees_to_kill.append(tree)

        for tree in self.inferences:
            if parameter == tree.root:
                orphaned_leafs += list(l for l in tree.leafs if
                                       l not in self.dependencies.roots)
                inffs_trees_to_kill.append(tree)

        new_trees = [ParamSpecTree(new_root) for new_root in orphaned_leafs]

        new_deps_trees = [t for t in self.dependencies.trees if
                          t not in deps_trees_to_kill]
        new_deps_trees += new_trees

        new_inffs_trees =  [t for t in self.inferences.trees if
                            t not in inffs_trees_to_kill]

        idps = InterDependencies_(dependencies=new_deps_trees,
                                  inferences=new_inffs_trees)

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

    def serialize(self) -> Dict[str, SerializedGrove]:
        """
        Write out this object as a dictionary
        """
        ser: Dict[str, SerializedGrove] = {}
        ser['dependencies'] = self.dependencies.serialize()
        ser['inferences'] = self.inferences.serialize()
        return ser

    @classmethod
    def deserialize(cls,
                    ser: Dict[str, SerializedGrove]) -> 'InterDependencies_':
        """
        Construct an InterDependencies_ object from a serialization of such
        an object
        """
        deps = ParamSpecGrove.deserialize(ser['dependencies'])
        inffs = ParamSpecGrove.deserialize(ser['inferences'])

        return cls(dependencies=tuple(deps._trees),
                   inferences=tuple(inffs._trees))

    def __repr__(self) -> str:
        rep = (f"InterDependencies_(dependencies={self.dependencies}, "
               f"inferences={self.inferences}, ")
        return rep

    def __eq__(self, other) -> bool:

        if not isinstance(other, InterDependencies_):
            return False

        if self.dependencies != other.dependencies:
            return False

        if self.inferences != other.inferences:
            return False

        return True

    def __contains__(self, ps: ParamSpecBase) -> bool:
        containers = (self.dependencies.roots, self.dependencies.leafs,
                      self.inferences.roots, self.inferences.leafs)
        return any(ps in cont for cont in containers)

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

    dependencies = []
    inferences = []

    already_added = []

    tree_like = (ps for ps in namedict.values() if
                 len(ps._depends_on) > 0 or len(ps._inferred_from) > 0)

    for ps in tree_like:
        root = ps.base_version()
        deps = tuple(namedict[n].base_version() for n in ps.depends_on_)
        inffs = tuple(namedict[n].base_version() for n in ps.inferred_from_)
        if len(deps) > 0:
            dependencies.append(ParamSpecTree(root, *deps))
            for name in (ps.name for ps in (root,) + deps):
                already_added.append(name)
        if len(inffs) > 0:
            inferences.append(ParamSpecTree(root, *inffs))
            for name in (ps.name for ps in (root,) + inffs):
                already_added.append(name)

    missing = [name for name in namedict.keys() if name not in already_added]

    for mps in missing:
        dependencies.append(ParamSpecTree(namedict[mps].base_version()))

    idps_ = InterDependencies_(dependencies=dependencies,
                               inferences=inferences)
    return idps_


def new_to_old(idps: InterDependencies_) -> InterDependencies:
    """
    Create a new InterDependencies object (old style) from an existing
    InterDependencies_ object (new style). Leaves the original object
    unchanged. Only meant to be used for ensuring backwards-compatibility
    until we update sqlite_base to forget about ParamSpecs
    """

    paramspecs: Dict[str, ParamSpec] = {}


    for dep_tree in idps.dependencies:
        root = dep_tree.root
        leafs = dep_tree._leafs_tuple
        paramspecs.update({root.name: ParamSpec(name=root.name,
                                                paramtype=root.type,
                                                label=root.label,
                                                unit=root.unit)})
        for leaf in leafs:
            paramspecs.update({leaf.name: ParamSpec(name=leaf.name,
                                                    paramtype=leaf.type,
                                                    label=leaf.label,
                                                    unit=leaf.unit)})
            paramspecs[root.name]._depends_on.append(leaf.name)

    for inff_tree in idps.inferences:
        root = inff_tree.root
        leafs = inff_tree._leafs_tuple

        # it's very unlikely that root is not already present, but it can
        # in principle happen
        if root.name not in paramspecs:
            paramspecs.update({root.name: ParamSpec(name=root.name,
                                                   paramtype=root.type,
                                                   label=root.label,
                                                   unit=root.unit)})

        for leaf in leafs:
            if leaf.name not in paramspecs:
                paramspecs.update({leaf.name: ParamSpec(name=leaf.name,
                                                        paramtype=leaf.type,
                                                        label=leaf.label,
                                                        unit=leaf.unit)})
            paramspecs[root.name]._inferred_from.append(leaf.name)


    return InterDependencies(*tuple(paramspecs.values()))
