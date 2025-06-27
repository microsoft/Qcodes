"""
This module holds the objects that describe the intra-run relationships
between the parameters of that run. Most importantly, the information about
which parameters depend on each other is handled here.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from itertools import chain, product
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
import numpy.typing as npt

from .param_spec import ParamSpecBase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .versioning.rundescribertypes import InterDependencies_Dict

ParamSpecTree = dict[ParamSpecBase, tuple[ParamSpecBase, ...]]
ParamNameTree = dict[str, list[str]]
ErrorTuple = tuple[type[Exception], str]


class IncompleteSubsetError(Exception):
    def __init__(self, subset_parans: set[str], missing_params: set[str], *args: Any):
        super().__init__(*args)
        self._subset_parans = subset_parans
        self._missing_params = missing_params

    def __str__(self) -> str:
        return (
            f"{self._subset_parans} is not a complete subset. The following interdependencies are "
            f"missing: {self._missing_params}"
        )


class InterDependencies_:
    def __init__(
        self,
        dependencies: ParamSpecTree | None = None,
        inferences: ParamSpecTree | None = None,
        standalones: tuple[ParamSpecBase, ...] = (),
    ):
        self._graph: nx.DiGraph = nx.DiGraph()
        self.add_dependencies(dependencies)
        self.add_inferences(inferences)
        self.add_standalones(standalones)

    def add_paramspecs(self, paramspecs: list[ParamSpecBase]) -> None:
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
        self, links: list[tuple[ParamSpecBase, ParamSpecBase]], type: str
    ) -> None:
        for link in links:
            paramspec_from, paramspec_to = link
            if self._graph.has_edge(paramspec_to.name, paramspec_from.name) or (
                self._graph.has_edge(paramspec_from.name, paramspec_to.name)
                and self.graph[paramspec_from.name][paramspec_to.name]["type"] != type
            ):
                raise ValueError(
                    f"An edge between {paramspec_from.name} and {paramspec_to.name} already exists. \n"
                    "The relationship between them is not well-defined"
                )
            self._graph.add_edge(paramspec_from.name, paramspec_to.name, type=type)

    def add_dependencies(self, dependencies: ParamSpecTree | None) -> None:
        if dependencies is None or dependencies == {}:
            return
        self.validate_paramspectree(dependencies, type="dependencies")
        self._add_interdeps(dependencies, type="depends_on")

    def add_inferences(self, inferences: ParamSpecTree | None) -> None:
        if inferences is None or inferences == {}:
            return
        self.validate_paramspectree(inferences, type="inferences")
        self._add_interdeps(inferences, type="inferred_from")

    def add_standalones(self, standalones: tuple[ParamSpecBase, ...]) -> None:
        for ps in standalones:
            if not isinstance(ps, ParamSpecBase):
                raise ValueError("Invalid standalones") from TypeError(
                    "Standalones must be a sequence of ParamSpecs"
                )
        self.add_paramspecs(list(standalones))

    def _add_interdeps(self, interdeps: ParamSpecTree, type: str) -> None:
        for spec_dep, spec_indeps in interdeps.items():
            flat_specs = list(chain.from_iterable([(spec_dep,), spec_indeps]))
            flat_deps = list(product((spec_dep,), spec_indeps))
            self.add_paramspecs(flat_specs)
            self._add_interdeps_by_type(flat_deps, type=type)
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
    def _dependency_subgraph(self) -> nx.DiGraph:
        depends_on_edges = [
            edge
            for edge in self.graph.edges
            if self.graph.edges[edge]["type"] == "depends_on"
        ]
        return cast("nx.DiGraph", self.graph.edge_subgraph(depends_on_edges))

    @property
    def _inference_subgraph(self) -> nx.DiGraph:
        inferred_from_edges = [
            edge
            for edge in self.graph.edges
            if self.graph.edges[edge]["type"] == "inferred_from"
        ]
        return cast("nx.DiGraph", self.graph.edge_subgraph(inferred_from_edges))

    def extend(
        self,
        dependencies: ParamSpecTree | None = None,
        inferences: ParamSpecTree | None = None,
        standalones: tuple[ParamSpecBase, ...] = (),
    ) -> InterDependencies_:
        new_interdependencies = InterDependencies_._from_graph(deepcopy(self.graph))

        new_interdependencies.add_dependencies(dependencies)
        new_interdependencies.add_inferences(inferences)
        new_interdependencies.add_standalones(standalones)
        return new_interdependencies

    def _paramspec_tree_by_type(self, type: str) -> ParamSpecTree:
        paramspec_tree_list: dict[ParamSpecBase, list[ParamSpecBase]] = defaultdict(
            list
        )
        for node_from, node_to, edge_data in self.graph.out_edges(data=True):
            if edge_data["type"] == type:
                paramspec_tree_list[self._node_to_paramspec(node_from)].append(
                    self._node_to_paramspec(node_to)
                )
        return {key: tuple(val) for key, val in paramspec_tree_list.items()}

    def _node_to_paramspec(self, node_id: str) -> ParamSpecBase:
        return cast("ParamSpecBase", self.graph.nodes[node_id]["value"])

    def _paramspec_predecessors_by_type(
        self, paramspec: ParamSpecBase, type: str
    ) -> tuple[ParamSpecBase, ...]:
        return tuple(
            self._node_to_paramspec(node_from)
            for node_from, _, edge_data in self.graph.in_edges(
                paramspec.name, data=True
            )
            if edge_data["type"] == type
        )

    @property
    def dependencies(self) -> ParamSpecTree:
        return self._paramspec_tree_by_type("depends_on")

    def what_depends_on(self, ps: ParamSpecBase) -> tuple[ParamSpecBase, ...]:
        return self._paramspec_predecessors_by_type(ps, type="depends_on")

    def what_is_inferred_from(self, ps: ParamSpecBase) -> tuple[ParamSpecBase, ...]:
        return self._paramspec_predecessors_by_type(ps, type="inferred_from")

    @property
    def inferences(self) -> ParamSpecTree:
        return self._paramspec_tree_by_type("inferred_from")

    @property
    def standalones(self) -> frozenset[ParamSpecBase]:
        degree_iterator = self.graph.degree
        assert not isinstance(
            degree_iterator, int
        )  # without arguments, graph.degree returns an iterable
        return frozenset(
            [
                self._node_to_paramspec(node_id)
                for node_id, degree in degree_iterator
                if degree == 0
            ]
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.graph)

    @property
    def paramspecs(self) -> tuple[ParamSpecBase, ...]:
        """
        Return the ParamSpecBase objects of this instance
        """
        return tuple(
            cast("ParamSpecBase", paramspec)
            # The type check for this does not correctly allow the `data` arg to be a string
            for _, paramspec in self.graph.nodes(data="value")  # pyright: ignore[reportArgumentType]
        )

    @property
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

    def remove(self, paramspec: ParamSpecBase) -> InterDependencies_:
        if self.graph.in_degree(paramspec.name) > 0:
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
        return nx.utils.misc.graphs_equal(self.graph, other.graph)

    def __contains__(self, ps: ParamSpecBase) -> bool:
        return ps.name in self.graph

    def __getitem__(self, name: str) -> ParamSpecBase:
        return self._node_to_paramspec(name)

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    def to_ipycytoscape_json(self) -> dict[str, list[dict[str, Any]]]:
        graph_json: dict[str, list[dict[str, Any]]] = nx.cytoscape_data(self.graph)[
            "elements"
        ]
        # TODO: Add different node types?
        for edge_dict in graph_json["edges"]:
            edge_dict["classes"] = edge_dict["data"]["type"]
        return graph_json

    @staticmethod
    def validate_paramspectree(
        paramspectree: ParamSpecTree, type: str | None = None
    ) -> None:
        """
        Validate a ParamSpecTree. Apart from adhering to the type, a
        ParamSpecTree must not have any cycles.

        Returns:
            A tuple with an exception type and an error message or None, if
            the paramtree is valid

        """
        type = type or "ParamSpecTree"
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
                raise ValueError(f"Invalid {type}") from ValueError(
                    "ParamSpecTree can not have cycles"
                )
        else:
            raise ValueError(f"Invalid {type}") from TypeError(cause)

    def validate_subset(self, paramspecs: Sequence[ParamSpecBase]) -> None:
        subset_nodes = set([paramspec.name for paramspec in paramspecs])
        for subset_node in subset_nodes:
            descendant_nodes_per_subset_node = nx.descendants(self.graph, subset_node)
            if missing_nodes := descendant_nodes_per_subset_node.difference(
                subset_nodes
            ):
                raise IncompleteSubsetError(
                    subset_parans=subset_nodes, missing_params=missing_nodes
                )

    @classmethod
    def _from_graph(cls, graph: nx.DiGraph) -> InterDependencies_:
        new_interdependencies = cls()
        new_interdependencies._graph = graph
        return new_interdependencies

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

    def _empty_data_dict(self) -> dict[str, dict[str, npt.NDArray]]:
        """
        Create an dictionary with empty numpy arrays as values
        matching the expected output of ``DataSet``'s ``get_parameter_data`` /
        ``cache.data`` so that the order of keys in the returned dictionary
        is the same as the order of parameters in the interdependencies
        in this class.
        """

        output: dict[str, dict[str, npt.NDArray]] = {}
        for dependent, independents in self.dependencies.items():
            dependent_name = dependent.name
            output[dependent_name] = {dependent_name: np.array([])}
            for independent in independents:
                output[dependent_name][independent.name] = np.array([])
        for standalone in (ps.name for ps in self.standalones):
            output[standalone] = {}
            output[standalone][standalone] = np.array([])
        return output


def paramspec_tree_to_param_name_tree(
    paramspec_tree: ParamSpecTree,
) -> ParamNameTree:
    return {
        key.name: [item.name for item in items] for key, items in paramspec_tree.items()
    }
