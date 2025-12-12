"""
This module holds the objects that describe the intra-run relationships
between the parameters of that run. Most importantly, the information about
which parameters depend on each other is handled here.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from itertools import chain, product
from typing import TYPE_CHECKING, Any, Literal, cast

import networkx as nx
from typing_extensions import deprecated

from qcodes.parameters import ParamSpecBase
from qcodes.utils import QCoDeSDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .versioning.rundescribertypes import InterDependencies_Dict
_LOGGER = logging.getLogger(__name__)
ParamSpecTree = dict[ParamSpecBase, tuple[ParamSpecBase, ...]]
ParamNameTree = dict[str, list[str]]
ErrorTuple = tuple[type[Exception], str]
_InterDepType = Literal["depends_on", "inferred_from"]


class IncompleteSubsetError(Exception):
    def __init__(self, subset_params: set[str], missing_params: set[str], *args: Any):
        super().__init__(*args)
        self._subset_params = subset_params
        self._missing_params = missing_params

    def __str__(self) -> str:
        return (
            f"{self._subset_params} is not a complete subset. The following interdependencies are "
            f"missing: {self._missing_params}"
        )


class InterDependencies_:  # noqa: PLW1641
    # todo: not clear if this should implement __hash__.
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
        self._graph: nx.DiGraph[str] = nx.DiGraph()
        self.add_dependencies(dependencies)
        self.add_inferences(inferences)
        self.add_standalones(standalones)

    def add_paramspecs(self, paramspecs: Sequence[ParamSpecBase]) -> None:
        for paramspec in paramspecs:
            if (
                paramspec.name in self.graph.nodes
                and self.graph.nodes[paramspec.name]["value"] != paramspec
            ):
                raise ValueError(
                    f"A ParamSpecBase with name {paramspec.name} already exists in the graph and\n"
                    f"{paramspec} != {self.graph.nodes[paramspec.name]['value']} "
                )
            self._graph.add_node(paramspec.name, value=paramspec)

    def _add_interdeps_by_type(
        self,
        links: Sequence[tuple[ParamSpecBase, ParamSpecBase]],
        interdep_type: _InterDepType,
    ) -> None:
        for link in links:
            paramspec_from, paramspec_to = link
            if self._graph.has_edge(paramspec_to.name, paramspec_from.name) or (
                self._graph.has_edge(paramspec_from.name, paramspec_to.name)
                and self.graph[paramspec_from.name][paramspec_to.name]["interdep_type"]
                != interdep_type
            ):
                raise ValueError(
                    f"An edge between {paramspec_from.name} and {paramspec_to.name} already exists. \n"
                    "The relationship between them is not well-defined"
                )
            self._graph.add_edge(
                paramspec_from.name, paramspec_to.name, interdep_type=interdep_type
            )

    def add_dependencies(self, dependencies: ParamSpecTree | None) -> None:
        if dependencies is None or dependencies == {}:
            return
        self.validate_paramspectree(dependencies, interdep_type="dependencies")
        self._add_interdeps(dependencies, interdep_type="depends_on")

    def add_inferences(self, inferences: ParamSpecTree | None) -> None:
        if inferences is None or inferences == {}:
            return
        self.validate_paramspectree(inferences, interdep_type="inferences")
        self._add_interdeps(inferences, interdep_type="inferred_from")

    def add_standalones(self, standalones: tuple[ParamSpecBase, ...]) -> None:
        for ps in standalones:
            if not isinstance(ps, ParamSpecBase):
                raise ValueError("Invalid standalones") from TypeError(
                    "Standalones must be a sequence of ParamSpecs"
                )
        self.add_paramspecs(list(standalones))

    def _add_interdeps(
        self, interdeps: ParamSpecTree, interdep_type: _InterDepType
    ) -> None:
        for spec_dep, spec_indeps in interdeps.items():
            flat_specs = list(chain.from_iterable([(spec_dep,), spec_indeps]))
            flat_deps = list(product((spec_dep,), spec_indeps))
            self.add_paramspecs(flat_specs)
            self._add_interdeps_by_type(flat_deps, interdep_type=interdep_type)
        self._validate_interdependencies(interdeps)

    def _validate_interdependencies(self, interdeps: ParamSpecTree) -> None:
        self._validate_acyclic(interdeps)
        self._validate_no_chained_dependencies(interdeps)

    def _validate_acyclic(self, interdeps: ParamSpecTree) -> None:
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError(
                f"Adding these interdependencies {interdeps} caused the graph to become cyclic"
            )

    def _validate_no_chained_dependencies(self, interdeps: ParamSpecTree) -> None:
        for node, in_degree in self._dependency_subgraph.in_degree:
            out_degree = self._dependency_subgraph.out_degree(node)
            if in_degree > 0 and out_degree > 0:
                depends_on_nodes = list(self._dependency_subgraph.successors(node))
                depended_on_nodes = list(self._dependency_subgraph.predecessors(node))
                raise ValueError(
                    f"Paramspec {node} both depends on {depends_on_nodes} and is depended upon by {depended_on_nodes} \n"
                    f"This was caused while adding these interdependencies {interdeps}"
                )

    @property
    def _dependency_subgraph(self) -> nx.DiGraph[str]:
        depends_on_edges = [
            edge
            for edge in self.graph.edges
            if self.graph.edges[edge]["interdep_type"] == "depends_on"
        ]
        # the type annotations does not currently encode that edge_subgraph of a DiGraph
        # is a DiGraph
        return cast("nx.DiGraph[str]", self.graph.edge_subgraph(depends_on_edges))

    @property
    def _inference_subgraph(self) -> nx.DiGraph[str]:
        inferred_from_edges = [
            edge
            for edge in self.graph.edges
            if self.graph.edges[edge]["interdep_type"] == "inferred_from"
        ]
        # the type annotations does not currently encode that edge_subgraph of a DiGraph
        # is a DiGraph
        return cast("nx.DiGraph[str]", self.graph.edge_subgraph(inferred_from_edges))

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
        new_interdependencies = InterDependencies_._from_graph(deepcopy(self.graph))

        new_interdependencies.add_dependencies(dependencies)
        new_interdependencies.add_inferences(inferences)
        new_interdependencies.add_standalones(standalones)
        return new_interdependencies

    def _paramspec_tree_by_type(self, interdep_type: _InterDepType) -> ParamSpecTree:
        paramspec_tree_list: dict[ParamSpecBase, list[ParamSpecBase]] = defaultdict(
            list
        )
        for node_from, node_to, edge_data in self.graph.out_edges(data=True):
            if edge_data["interdep_type"] == interdep_type:
                paramspec_tree_list[self._node_to_paramspec(node_from)].append(
                    self._node_to_paramspec(node_to)
                )
        return {key: tuple(val) for key, val in paramspec_tree_list.items()}

    def _node_to_paramspec(self, node_id: str) -> ParamSpecBase:
        return self.graph.nodes[node_id]["value"]

    def _paramspec_predecessors_by_type(
        self, paramspec: ParamSpecBase, interdep_type: _InterDepType
    ) -> tuple[ParamSpecBase, ...]:
        return tuple(
            self._node_to_paramspec(node_from)
            for node_from, _, edge_data in self.graph.in_edges(
                paramspec.name, data=True
            )
            if edge_data["interdep_type"] == interdep_type
        )

    @property
    def dependencies(self) -> ParamSpecTree:
        return self._paramspec_tree_by_type("depends_on")

    def what_depends_on(self, ps: ParamSpecBase) -> tuple[ParamSpecBase, ...]:
        """
        Return a tuple of the parameters that depend on the given parameter.
        Returns an empty tuple if nothing depends on the given parameter

        Args:
            ps: the parameter to look up

        Raises:
            ValueError: If the parameter is not part of this object

        """
        return self._paramspec_predecessors_by_type(ps, interdep_type="depends_on")

    def what_is_inferred_from(self, ps: ParamSpecBase) -> tuple[ParamSpecBase, ...]:
        """
        Return a tuple of the parameters that are inferred from the given
        parameter. Returns an empty tuple if nothing is inferred from the given
        parameter

        Args:
            ps: the parameter to look up

        Raises:
            ValueError: If the parameter is not part of this object

        """
        return self._paramspec_predecessors_by_type(ps, interdep_type="inferred_from")

    @property
    def inferences(self) -> ParamSpecTree:
        return self._paramspec_tree_by_type("inferred_from")

    @property
    def standalones(self) -> frozenset[ParamSpecBase]:
        return frozenset(
            [
                self._node_to_paramspec(node_id)
                for node_id, degree in self.graph.degree
                if degree == 0
            ]
        )

    @property
    def names(self) -> tuple[str, ...]:
        """
        Return all the names of the parameters of this instance
        """
        return tuple(self.graph)

    @property
    def paramspecs(self) -> tuple[ParamSpecBase, ...]:
        """
        Return the ParamSpecBase objects of this instance
        """
        return tuple(paramspec for _, paramspec in self.graph.nodes(data="value"))

    @property
    @deprecated(
        "non_dependencies returns incorrect results and is deprecated. Use top_level_parameters as an alternative.",
        category=QCoDeSDeprecationWarning,
    )
    def non_dependencies(self) -> tuple[ParamSpecBase, ...]:
        """
        Return all parameters that are not dependencies of other parameters,
        i.e. return the top level parameters. Returned tuple is sorted by
        parameter names.
        """
        non_dependencies = tuple(self.standalones) + tuple(self.dependencies.keys())
        non_dependencies_sorted_by_name = tuple(
            sorted(non_dependencies, key=lambda ps: ps.name)
        )
        return non_dependencies_sorted_by_name

    @property
    def top_level_parameters(self) -> tuple[ParamSpecBase, ...]:
        """
        Return all parameters that are not dependencies or inferred from other parameters,
        i.e. return the top level parameters.

        Returns:
            A tuple of top level parameters sorted by their names.

        """

        # is is not sufficient to find all parameters with in_degree == 0
        # since some of the inferred parameters might be included in the dependency tree
        # of another parameter since we include inferred parameters both ways.
        # see test_dependency_on_middle_parameter for a test that illustrates this.
        inference_top_level = {
            self._node_to_paramspec(node_id)
            for node_id, in_degree in self._inference_subgraph.in_degree
            if in_degree == 0
        }
        dependency_top_level = {
            self._node_to_paramspec(node_id)
            for node_id, in_degree in self._dependency_subgraph.in_degree
            if in_degree == 0
        }
        standalone_top_level = {
            self._node_to_paramspec(node_id)
            for node_id, degree in self._graph.degree
            if degree == 0
        }

        all_paramspecs_in_dependency_tree = set(
            chain.from_iterable(
                [self.find_all_parameters_in_tree(ps) for ps in dependency_top_level]
            )
        )

        inference_top_level_not_in_dependency_tree = inference_top_level.difference(
            all_paramspecs_in_dependency_tree
        )

        all_params = (
            dependency_top_level
            | inference_top_level_not_in_dependency_tree
            | standalone_top_level
        )

        return tuple(sorted(all_params, key=lambda ps: ps.name))

    def remove(self, paramspec: ParamSpecBase) -> InterDependencies_:
        """
        Create a new :class:`InterDependencies_` object that is similar
        to this instance, but has the given parameter removed.
        """
        paramspec_in_degree = self.graph.in_degree(paramspec.name)
        if paramspec_in_degree > 0:
            raise ValueError(
                f"Cannot remove {paramspec.name}, other parameters depend on or are inferred from it"
            )
        new_graph = deepcopy(self.graph)
        new_graph.remove_node(paramspec.name)
        return InterDependencies_._from_graph(new_graph)

    def __repr__(self) -> str:
        rep = (
            f"InterDependencies_(dependencies={self.dependencies}, "
            f"inferences={self.inferences}, "
            f"standalones={self.standalones})"
        )
        return rep

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InterDependencies_):
            return False
        return nx.utils.graphs_equal(self.graph, other.graph)

    def __contains__(self, ps: ParamSpecBase) -> bool:
        return ps.name in self.graph

    def __getitem__(self, name: str) -> ParamSpecBase:
        return self._node_to_paramspec(name)

    @property
    def graph(self) -> nx.DiGraph[str]:
        return self._graph

    def to_ipycytoscape_json(self) -> dict[str, list[dict[str, Any]]]:
        graph_json: dict[str, list[dict[str, Any]]] = nx.cytoscape_data(self.graph)[
            "elements"
        ]
        # TODO: Add different node types?
        for edge_dict in graph_json["edges"]:
            edge_dict["classes"] = edge_dict["data"]["interdep_type"]
        return graph_json

    @staticmethod
    def validate_paramspectree(
        paramspectree: ParamSpecTree,
        interdep_type: Literal["dependencies", "inferences", "ParamSpecTree"]
        | None = None,
    ) -> None:
        """
        Validate a ParamSpecTree. Apart from adhering to the type, a
        ParamSpecTree must not have any cycles.

        Returns:
            A tuple with an exception type and an error message or None, if
            the paramtree is valid

        """
        interdep_type_internal = interdep_type or "ParamSpecTree"
        cause: str | None = None

        # Validate the type
        if not isinstance(paramspectree, dict):
            cause = "ParamSpecTree must be a dict"
        if cause is None:
            for key, values in paramspectree.items():
                if not isinstance(key, ParamSpecBase):
                    cause = "ParamSpecTree must have ParamSpecs as keys"
                    break
                if not isinstance(values, tuple):
                    cause = "ParamSpecTree must have tuple values"
                    break
                for value in values:
                    if not isinstance(value, ParamSpecBase):
                        cause = (
                            "ParamSpecTree can only have tuples of ParamSpecs as values"
                        )
                        break

        if cause is None:
            # check for cycles
            roots = set(paramspectree.keys())
            leafs = {ps for tup in paramspectree.values() for ps in tup}

            if roots.intersection(leafs) != set():
                raise ValueError(f"Invalid {interdep_type_internal}") from ValueError(
                    "ParamSpecTree can not have cycles"
                )
        else:
            raise ValueError(f"Invalid {interdep_type_internal}") from TypeError(cause)

    def _invalid_subsets(
        self, paramspecs: Sequence[ParamSpecBase]
    ) -> tuple[set[str], set[str]] | None:
        subset_nodes = {paramspec.name for paramspec in paramspecs}
        for subset_node in subset_nodes:
            descendant_nodes_per_subset_node = nx.descendants(self.graph, subset_node)
            if missing_nodes := descendant_nodes_per_subset_node.difference(
                subset_nodes
            ):
                return (subset_nodes, missing_nodes)
        return None

    def validate_subset(self, paramspecs: Sequence[ParamSpecBase]) -> None:
        """
        Validate that the given parameters form a valid subset of the
        parameters of this instance, meaning that all the given parameters are
        actually found in this instance and that there are no missing
        dependencies/inferences.

        Args:
            paramspecs: The collection of ParamSpecBases to validate

        Raises:
            InterdependencyError: If a dependency or inference is missing

        """
        invalid_subset = self._invalid_subsets(paramspecs)
        if invalid_subset is not None:
            raise IncompleteSubsetError(
                subset_params=invalid_subset[0], missing_params=invalid_subset[1]
            )

    @classmethod
    def _from_graph(cls, graph: nx.DiGraph[str]) -> InterDependencies_:
        new_interdependencies = cls()
        new_interdependencies._graph = graph
        return new_interdependencies

    def find_all_parameters_in_tree(
        self, initial_param: ParamSpecBase
    ) -> set[ParamSpecBase]:
        """
        Collect all parameters that are transitively related to the initial parameter.

        This includes dependencies of the initial parameter and parameters that are inferred from
        the initial parameter, as well as parameters that are inferred from its dependencies.

        Args:
            initial_param: The parameter to start the traversal from.

        Returns:
            Set of all parameters transitively related to the initial parameter

        Raises:
            ValueError: If the initial parameter is not part of the graph.

        """

        # Use NetworkX to find all nodes reachable from initial parameters
        collected_nodes: set[str] = set()

        if initial_param.name not in self.graph:
            available_params = ", ".join(self.graph.nodes)
            raise ValueError(
                f"Parameter '{initial_param.name}' is not part of the graph. "
                f"Available parameters are: {available_params}. "
                f"Please check if the parameter name is correct or if the graph has been properly initialized."
            )

        # Add the parameter itself
        collected_nodes.add(initial_param.name)

        # find all parameters that this parameter depends on
        if initial_param.name in self._dependency_subgraph:
            dep_descendants = nx.descendants(
                self._dependency_subgraph, initial_param.name
            )
            collected_nodes.update(dep_descendants)

        # find all parameters that are inferred from the parameter or its dependencies

        for param_name in collected_nodes.copy():
            if param_name in self._inference_subgraph:
                descendants = nx.descendants(self._inference_subgraph, param_name)
                ancestors = nx.ancestors(self._inference_subgraph, param_name)
                collected_nodes.update(descendants)
                collected_nodes.update(ancestors)

        # Convert node names back to ParamSpecBase objects
        collected_params: set[ParamSpecBase] = set()
        for node_name in collected_nodes:
            collected_params.add(self._node_to_paramspec(node_name))
        return collected_params

    def all_parameters_in_tree_by_group(
        self, initial_param: ParamSpecBase
    ) -> tuple[ParamSpecBase, tuple[ParamSpecBase, ...], tuple[ParamSpecBase, ...]]:
        """
        Collect all parameters that are transitively related to the initial parameter
        and organize them into three groups.

        This includes dependencies of the initial parameter and parameters that are inferred from
        the initial parameter, as well as parameters that are inferred from its dependencies.
        The parameter must be part of the interdependency graph.

        Args:
            initial_param: The parameter to start the traversal from.

        Returns:
            A tuple containing:
            - The initial parameter
            - A tuple of direct dependencies of the initial parameter
            - A tuple of parameters inferred from the initial parameter and its dependencies (sorted by name).

        Raises:
            ValueError: If the initial parameter is not part of the graph.

        """
        collected_params = self.find_all_parameters_in_tree(initial_param)

        collected_params.remove(initial_param)

        dependencies = self.dependencies.get(initial_param, ())

        for dep in dependencies:
            collected_params.remove(dep)

        # Sort the remaining parameters by their names to ensure a consistent order
        remaining_params_sorted = sorted(collected_params, key=lambda ps: ps.name)

        return initial_param, tuple(dependencies), tuple(remaining_params_sorted)

    @classmethod
    def _from_dict(cls, ser: InterDependencies_Dict) -> InterDependencies_:
        """
        Construct an InterDependencies_ object from a dictionary
        representation of such an object
        """
        params = ser["parameters"]
        deps = cls._extract_deps_from_dict(ser)

        inffs = cls._extract_inffs_from_dict(ser)

        stdls = tuple(
            ParamSpecBase._from_dict(params[ps_id]) for ps_id in ser["standalones"]
        )

        return cls(dependencies=deps, inferences=inffs, standalones=stdls)

    @classmethod
    def _extract_inffs_from_dict(cls, ser: InterDependencies_Dict) -> ParamSpecTree:
        params = ser["parameters"]
        inffs = {}
        for key, value in ser["inferences"].items():
            inffs_key = ParamSpecBase._from_dict(params[key])
            inffs_vals = tuple(ParamSpecBase._from_dict(params[val]) for val in value)
            inffs.update({inffs_key: inffs_vals})
        return inffs

    @classmethod
    def _extract_deps_from_dict(cls, ser: InterDependencies_Dict) -> ParamSpecTree:
        params = ser["parameters"]
        deps = {}
        for key, value in ser["dependencies"].items():
            deps_key = ParamSpecBase._from_dict(params[key])
            deps_vals = tuple(ParamSpecBase._from_dict(params[val]) for val in value)
            deps.update({deps_key: deps_vals})
        return deps

    def _to_dict(self) -> InterDependencies_Dict:
        """
        Write out this object as a dictionary
        """
        parameters = {
            node_id: data["value"]._to_dict()
            for node_id, data in self.graph.nodes(data=True)
        }

        dependencies = paramspec_tree_to_param_name_tree(self.dependencies)
        inferences = paramspec_tree_to_param_name_tree(self.inferences)
        standalones = [paramspec.name for paramspec in self.standalones]
        output: InterDependencies_Dict = {
            "parameters": parameters,
            "dependencies": dependencies,
            "inferences": inferences,
            "standalones": standalones,
        }
        return output

    @property
    def _id_to_paramspec(self) -> dict[str, ParamSpecBase]:
        return {node_id: data["value"] for node_id, data in self.graph.nodes(data=True)}

    @property
    def _paramspec_to_id(self) -> dict[ParamSpecBase, str]:
        return {data["value"]: node_id for node_id, data in self.graph.nodes(data=True)}


def paramspec_tree_to_param_name_tree(
    paramspec_tree: ParamSpecTree,
) -> ParamNameTree:
    return {
        key.name: [item.name for item in items] for key, items in paramspec_tree.items()
    }


class FrozenInterDependencies_(InterDependencies_):  # noqa: PLW1641
    # todo: not clear if this should implement __hash__.
    """
    A frozen version of InterDependencies_ that is immutable and caches
    expensive lookups. This is used exclusively while running a measurement
    to minimize the overhead of dependency lookups for each data operation.

    Args:
        interdeps: An InterDependencies_ instance to freeze

    """

    def __init__(self, interdeps: InterDependencies_):
        self._graph = interdeps.graph.copy()
        nx.freeze(self._graph)
        self._top_level_parameters_cache: tuple[ParamSpecBase, ...] | None = None
        self._dependencies_cache: ParamSpecTree | None = None
        self._inferences_cache: ParamSpecTree | None = None
        self._standalones_cache: frozenset[ParamSpecBase] | None = None
        self._find_all_parameters_in_tree_cache: dict[
            ParamSpecBase, set[ParamSpecBase]
        ] = {}
        self._invalid_subsets_cache: dict[
            tuple[ParamSpecBase, ...], tuple[set[str], set[str]] | None
        ] = {}
        self._id_to_paramspec_cache: dict[str, ParamSpecBase] | None = None
        self._paramspec_to_id_cache: dict[ParamSpecBase, str] | None = None

    def add_dependencies(self, dependencies: ParamSpecTree | None) -> None:
        raise TypeError("FrozenInterDependencies_ is immutable")

    def add_inferences(self, inferences: ParamSpecTree | None) -> None:
        raise TypeError("FrozenInterDependencies_ is immutable")

    def add_standalones(self, standalones: tuple[ParamSpecBase, ...]) -> None:
        raise TypeError("FrozenInterDependencies_ is immutable")

    def add_paramspecs(self, paramspecs: Sequence[ParamSpecBase]) -> None:
        raise TypeError("FrozenInterDependencies_ is immutable")

    def remove(self, paramspec: ParamSpecBase) -> InterDependencies_:
        raise TypeError("FrozenInterDependencies_ is immutable")

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
        # We need to unfreeze the graph for the new instance
        new_graph = nx.DiGraph(self.graph)
        new_interdependencies = InterDependencies_._from_graph(new_graph)

        new_interdependencies.add_dependencies(dependencies)
        new_interdependencies.add_inferences(inferences)
        new_interdependencies.add_standalones(standalones)
        return new_interdependencies

    @property
    def top_level_parameters(self) -> tuple[ParamSpecBase, ...]:
        if self._top_level_parameters_cache is None:
            self._top_level_parameters_cache = super().top_level_parameters
        return self._top_level_parameters_cache

    @property
    def dependencies(self) -> ParamSpecTree:
        if self._dependencies_cache is None:
            self._dependencies_cache = super().dependencies
        return self._dependencies_cache.copy()

    @property
    def inferences(self) -> ParamSpecTree:
        if self._inferences_cache is None:
            self._inferences_cache = super().inferences
        return self._inferences_cache.copy()

    @property
    def standalones(self) -> frozenset[ParamSpecBase]:
        if self._standalones_cache is None:
            self._standalones_cache = super().standalones
        return self._standalones_cache

    def find_all_parameters_in_tree(
        self, initial_param: ParamSpecBase
    ) -> set[ParamSpecBase]:
        if initial_param not in self._find_all_parameters_in_tree_cache:
            self._find_all_parameters_in_tree_cache[initial_param] = (
                super().find_all_parameters_in_tree(initial_param)
            )
        return self._find_all_parameters_in_tree_cache[initial_param].copy()

    @classmethod
    def _from_dict(cls, ser: InterDependencies_Dict) -> FrozenInterDependencies_:
        interdeps = InterDependencies_._from_dict(ser)
        return cls(interdeps)

    @classmethod
    def _from_graph(cls, graph: nx.DiGraph[str]) -> FrozenInterDependencies_:
        interdeps = InterDependencies_._from_graph(graph)
        return cls(interdeps)

    def validate_subset(self, paramspecs: Sequence[ParamSpecBase]) -> None:
        paramspecs_tuple = tuple(paramspecs)
        if paramspecs_tuple not in self._invalid_subsets_cache:
            self._invalid_subsets_cache[paramspecs_tuple] = self._invalid_subsets(
                paramspecs_tuple
            )
        invalid_subset = self._invalid_subsets_cache[paramspecs_tuple]
        if invalid_subset is not None:
            raise IncompleteSubsetError(
                subset_params=invalid_subset[0], missing_params=invalid_subset[1]
            )

    @property
    def _id_to_paramspec(self) -> dict[str, ParamSpecBase]:
        if self._id_to_paramspec_cache is None:
            self._id_to_paramspec_cache = {
                node_id: data["value"] for node_id, data in self.graph.nodes(data=True)
            }
        return self._id_to_paramspec_cache

    @property
    def _paramspec_to_id(self) -> dict[ParamSpecBase, str]:
        if self._paramspec_to_id_cache is None:
            self._paramspec_to_id_cache = {
                data["value"]: node_id for node_id, data in self.graph.nodes(data=True)
            }
        return self._paramspec_to_id_cache

    def __repr__(self) -> str:
        rep = (
            f"FrozenInterDependencies_(dependencies={self.dependencies}, "
            f"inferences={self.inferences}, "
            f"standalones={self.standalones})"
        )
        return rep

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrozenInterDependencies_):
            return False
        return nx.utils.graphs_equal(self.graph, other.graph)

    def to_interdependencies(self) -> InterDependencies_:
        """
        Convert this FrozenInterDependencies_ back to a mutable InterDependencies_ instance.

        Returns:
            A new InterDependencies_ instance with the same data as this frozen instance.

        """
        new_graph = nx.DiGraph(self.graph)
        return InterDependencies_._from_graph(new_graph)
