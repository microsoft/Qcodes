"""
This module holds the objects that describe the intra-run relationships
between the parameters of that run. Most importantly, the information about
which parameters depend on each other is handled here.
"""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from .param_spec import ParamSpec, ParamSpecBase

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .versioning.rundescribertypes import InterDependencies_Dict

ParamSpecTree = dict[ParamSpecBase, tuple[ParamSpecBase, ...]]
ParamNameTree = dict[str, tuple[str, ...]]
ErrorTuple = tuple[type[Exception], str]


class DependencyError(Exception):
    def __init__(self, param_name: str, missing_params: set[str], *args: Any):
        super().__init__(*args)
        self._param_name = param_name
        self._missing_params = missing_params

    def __str__(self) -> str:
        return (f'{self._param_name} has the following dependencies that are '
                f'missing: {self._missing_params}')


class InferenceError(Exception):
    def __init__(self, param_name: str, missing_params: set[str], *args: Any):
        super().__init__(*args)
        self._param_name = param_name
        self._missing_params = missing_params

    def __str__(self) -> str:
        return (f'{self._param_name} has the following inferences that are '
                f'missing: {self._missing_params}')


class InterDependencies_:
    """
    Object containing a group of ParamSpecs and the information about their
    internal relations to each other
    """

    def __init__(
        self,
        dependencies: ParamSpecTree | None = None,
        inferences: ParamSpecTree | None = None,
        standalones: tuple[ParamSpecBase, ...] = (),
    ):

        dependencies = dependencies or {}
        inferences = inferences or {}

        deps_error = self.validate_paramspectree(dependencies)
        if deps_error is not None:
            try:
                raise deps_error[0](deps_error[1])
            except Exception as old_error:
                raise ValueError("Invalid dependencies") from old_error

        inffs_error = self.validate_paramspectree(inferences)
        if inffs_error is not None:
            try:
                raise inffs_error[0](inffs_error[1])
            except Exception as old_error:
                raise ValueError("Invalid inferences") from old_error

        link_error = self._validate_double_links(dependencies, inferences)
        if link_error is not None:
            error, mssg = link_error
            raise error(mssg)

        for ps in standalones:
            if not isinstance(ps, ParamSpecBase):
                try:
                    raise TypeError("Standalones must be a sequence of ParamSpecs")
                except TypeError as base_error:
                    raise ValueError("Invalid standalones") from base_error

        self._remove_duplicates(dependencies)
        self._remove_duplicates(inferences)

        self.dependencies: ParamSpecTree = dependencies
        self.inferences: ParamSpecTree = inferences
        self.standalones: frozenset[ParamSpecBase] = frozenset(standalones)

        # The object is now complete, but for convenience, we form some more
        # private attributes for easy look-up

        self._id_to_paramspec: dict[str, ParamSpecBase] = {}
        for tree in (self.dependencies, self.inferences):
            for ps, ps_tup in tree.items():
                self._id_to_paramspec.update({ps.name: ps})
                self._id_to_paramspec.update({pst.name: pst
                                              for pst in ps_tup})
        for ps in self.standalones:
            self._id_to_paramspec.update({ps.name: ps})
        self._paramspec_to_id = {v: k for k, v in self._id_to_paramspec.items()}

        self._dependencies_inv: ParamSpecTree = self._invert_tree(
            self.dependencies)
        self._inferences_inv: ParamSpecTree = self._invert_tree(
            self.inferences)

        # Set operations are ~2x (or more) faster on strings than on hashable
        # ParamSpecBase objects, hence the need for use of the following
        # representation
        self._deps_names: ParamNameTree = self._tree_of_names(
            self.dependencies)
        self._infs_names: ParamNameTree = self._tree_of_names(
            self.inferences)

    @staticmethod
    def _tree_of_names(tree: ParamSpecTree) -> ParamNameTree:
        """
        Helper function to convert a ParamSpecTree-kind of tree where all
        ParamSpecBases are substituted with their ``.name`` s.
        Will turn {A: (B, C)} into {A.name: (B.name, C.name)}
        """
        name_tree: ParamNameTree = {}
        for ps, ps_tuple in tree.items():
            name_tree[ps.name] = tuple(p.name for p in ps_tuple)
        return name_tree

    @staticmethod
    def _invert_tree(tree: ParamSpecTree) -> ParamSpecTree:
        """
        Helper function to invert a ParamSpecTree. Will turn {A: (B, C)} into
        {B: (A,), C: (A,)}
        """
        indeps: set[ParamSpecBase] = set()
        for indep_tup in tree.values():
            indeps.update(indep_tup)

        inverted: ParamSpecTree = {}
        for indep in indeps:
            deps = tuple(ps for ps in tree if indep in tree[ps])
            inverted[indep] = deps

        return inverted

    @staticmethod
    def _remove_duplicates(tree: ParamSpecTree) -> None:
        """
        Helper function to remove duplicate entries from a ParamSpecTree while
        preserving order. Will turn {A: (B, B, C)} into {A: (B, C)}
        """
        for ps, tup in tree.items():
            specs: list[ParamSpecBase] = []
            for p in tup:
                if p not in specs:
                    specs.append(p)
            tree[ps] = tuple(specs)

    @staticmethod
    def validate_paramspectree(paramspectree: ParamSpecTree) -> ErrorTuple | None:
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
        leafs = {ps for tup in paramspectree.values() for ps in tup}

        if roots.intersection(leafs) != set():
            return (ValueError, 'ParamSpecTree can not have cycles')

        return None

    @staticmethod
    def _validate_double_links(
        tree1: ParamSpecTree, tree2: ParamSpecTree
    ) -> ErrorTuple | None:
        """
        Validate that two trees do not contain double links. A double link
        is a link between two nodes that exists in both trees. E.g. if the
        first tree has {A: (B, C)}, the second may not have {B: (A,)} etc.
        """
        for ps, tup in tree1.items():
            for val in tup:
                if ps in tree2.get(val, ()):
                    mssg = (f"Invalid dependencies/inferences. {ps} and "
                            f"{val} have an ill-defined relationship.")
                    return (ValueError, mssg)

        return None

    def what_depends_on(self, ps: ParamSpecBase) -> tuple[ParamSpecBase, ...]:
        """
        Return a tuple of the parameters that depend on the given parameter.
        Returns an empty tuple if nothing depends on the given parameter

        Args:
            ps: the parameter to look up

        Raises:
            ValueError if the parameter is not part of this object
        """
        if ps not in self:
            raise ValueError(f'Unknown parameter: {ps}')
        return self._dependencies_inv.get(ps, ())

    def what_is_inferred_from(self, ps: ParamSpecBase) -> tuple[ParamSpecBase, ...]:
        """
        Return a tuple of the parameters thatare inferred from the given
        parameter. Returns an empty tuple if nothing is inferred from the given
        parameter

        Args:
            ps: the parameter to look up

        Raises:
            ValueError if the parameter is not part of this object
        """
        if ps not in self:
            raise ValueError(f'Unknown parameter: {ps}')
        return self._inferences_inv.get(ps, ())

    def _to_dict(self) -> InterDependencies_Dict:
        """
        Write out this object as a dictionary
        """
        parameters = {key: value._to_dict() for key, value in
                      self._id_to_paramspec.items()}
        dependencies = self._construct_subdict('dependencies')
        inferences = self._construct_subdict('inferences')
        standalones = [self._paramspec_to_id[ps] for ps in
                       self.standalones]
        output: InterDependencies_Dict = {"parameters": parameters,
                                          "dependencies": dependencies,
                                          "inferences": inferences,
                                          "standalones": standalones}
        return output

    def _empty_data_dict(self) -> dict[str, dict[str, np.ndarray]]:
        """
        Create an dictionary with empty numpy arrays as values
        matching the expected output of ``DataSet``'s ``get_parameter_data`` /
        ``cache.data`` so that the order of keys in the returned dictionary
        is the same as the order of parameters in the interdependencies
        in this class.
        """

        output: dict[str, dict[str, np.ndarray]] = {}
        for dependent, independents in self.dependencies.items():
            dependent_name = dependent.name
            output[dependent_name] = {dependent_name: np.array([])}
            for independent in independents:
                output[dependent_name][independent.name] = np.array([])
        for standalone in (ps.name for ps in self.standalones):
            output[standalone] = {}
            output[standalone][standalone] = np.array([])
        return output

    def _construct_subdict(self, treename: str) -> dict[str, Any]:
        output = {}
        for key, value in getattr(self, treename).items():
            ps_id = self._paramspec_to_id[key]
            ps_ids = [self._paramspec_to_id[ps] for ps in value]
            output.update({ps_id: ps_ids})
        return output

    @property
    def paramspecs(self) -> tuple[ParamSpecBase, ...]:
        """
        Return the ParamSpecBase objects of this instance
        """
        return tuple(self._paramspec_to_id.keys())

    @property
    def non_dependencies(self) -> tuple[ParamSpecBase, ...]:
        """
        Return all parameters that are not dependencies of other parameters,
        i.e. return the top level parameters. Returned tuple is sorted by
        parameter names.
        """
        non_dependencies = tuple(self.standalones) \
                         + tuple(self.dependencies.keys())
        non_dependencies_sorted_by_name = tuple(
            sorted(non_dependencies, key=lambda ps: ps.name))
        return non_dependencies_sorted_by_name

    @property
    def names(self) -> tuple[str, ...]:
        """
        Return all the names of the parameters of this instance
        """
        return tuple(self._id_to_paramspec.keys())

    def _extend_with_paramspec(self, ps: ParamSpec) -> InterDependencies_:
        """
        Create a new InterDependencies_ object extended with the provided
        ParamSpec. A helper function for DataSet's add_parameter function.
        Note that this function will only work as expected if the ParamSpecs
        are extended into the InterDependencies_ in the "logical order", i.e.
        independent ParamSpecs before dependent ones.
        """
        base_ps = ps.base_version()

        old_standalones = set(self.standalones.copy())
        new_standalones: tuple[ParamSpecBase, ...]

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
        dependencies: ParamSpecTree | None = None,
        inferences: ParamSpecTree | None = None,
        standalones: tuple[ParamSpecBase, ...] = (),
    ) -> InterDependencies_:
        """
        Create a new :class:`InterDependencies_` object
        that is an extension of this instance with the provided input
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

        new_idps = InterDependencies_(dependencies=new_deps,
                                      inferences=new_inffs,
                                      standalones=new_standalones)

        return new_idps

    def remove(self, parameter: ParamSpecBase) -> InterDependencies_:
        """
        Create a new :class:`InterDependencies_` object that is similar
        to this instance, but has the given parameter removed.
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
        else:
            raise ValueError(f'Inconsistent InterDependencies_ object '
                             f'parameter: {parameter} is in list of'
                             f'parameters but is neither a "standalone", '
                             f'"dependency" or "inference"')

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
            parameters: The collection of ParamSpecBases to validate

        Raises:
            DependencyError, if a dependency is missing
            InferenceError, if an inference is missing
        """
        params = {p.name for p in parameters}

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
    def _from_dict(cls, ser: InterDependencies_Dict) -> InterDependencies_:
        """
        Construct an InterDependencies_ object from a dictionary
        representation of such an object
        """
        params = ser['parameters']
        deps = cls._extract_deps_from_dict(ser)

        inffs = cls._extract_inffs_from_dict(ser)

        stdls = tuple(ParamSpecBase._from_dict(params[ps_id]) for
                      ps_id in ser['standalones'])

        return cls(dependencies=deps, inferences=inffs, standalones=stdls)

    @classmethod
    def _extract_inffs_from_dict(cls,
                                 ser: InterDependencies_Dict) -> ParamSpecTree:
        params = ser['parameters']
        inffs = {}
        for key, value in ser['inferences'].items():
            inffs_key = ParamSpecBase._from_dict(params[key])
            inffs_vals = tuple(ParamSpecBase._from_dict(params[val]) for
                               val in value)
            inffs.update({inffs_key: inffs_vals})
        return inffs

    @classmethod
    def _extract_deps_from_dict(cls,
                                ser: InterDependencies_Dict) -> ParamSpecTree:
        params = ser['parameters']
        deps = {}
        for key, value in ser['dependencies'].items():
            deps_key = ParamSpecBase._from_dict(params[key])
            deps_vals = tuple(ParamSpecBase._from_dict(params[val]) for
                              val in value)
            deps.update({deps_key: deps_vals})
        return deps

    def __repr__(self) -> str:
        rep = (f"InterDependencies_(dependencies={self.dependencies}, "
               f"inferences={self.inferences}, "
               f"standalones={self.standalones})")
        return rep

    def __eq__(self, other: Any) -> bool:

        def sorter(inp: Iterable[ParamSpecBase]) -> list[ParamSpecBase]:
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
