"""
This module holds the objects that describe the intra-run relationships
between the parameters of that run. Most importantly, the information about
which parameters depend on each other is handled here.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import chain, product
from typing import TYPE_CHECKING, Any, Self, cast

import networkx as nx
import numpy as np
import numpy.typing as npt

from .param_spec import ParamSpecBase

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
        return (
            f"{self._param_name} has the following dependencies that are "
            f"missing: {self._missing_params}"
        )


class InferenceError(Exception):
    def __init__(self, param_name: str, missing_params: set[str], *args: Any):
        super().__init__(*args)
        self._param_name = param_name
        self._missing_params = missing_params

    def __str__(self) -> str:
        return (
            f"{self._param_name} has the following inferences that are "
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
        if dependencies is not None:
            self.add_dependencies(dependencies)
        if inferences is not None:
            self.add_inferences(inferences)
        if standalones != ():
            self.add_standalones(standalones)

    def add_paramspecs(self, paramspecs: list[ParamSpecBase]) -> None:
        for paramspec in paramspecs:
            self._graph.add_node(paramspec.name, value=paramspec)

    def add_interdeps_by_type(
        self, links: list[tuple[ParamSpecBase, ParamSpecBase]], type: str
    ) -> None:
        for link in links:
            paramspec_from, paramspec_to = link
            if self._graph.has_edge(
                paramspec_from.name, paramspec_to.name
            ) or self._graph.has_edge(paramspec_to.name, paramspec_from.name):
                raise ValueError("Link already exists")
            self._graph.add_edge(paramspec_from.name, paramspec_to.name, type=type)
        if not nx.is_forest(self.graph):
            raise ValueError(
                "Adding these interdependencies caused the graph to become cyclic"
            )

    def add_dependencies(self, dependencies: ParamSpecTree) -> None:
        self._add_interdeps(dependencies, type="depends_on")

    def add_inferences(self, inferences: ParamSpecTree) -> None:
        self._add_interdeps(inferences, type="inferred_from")

    def add_standalones(self, standalones: tuple[ParamSpecBase, ...]) -> None:
        self.add_paramspecs(list(standalones))

    def _add_interdeps(self, interdeps: ParamSpecTree, type: str) -> None:
        for spec_dep, spec_indeps in interdeps.items():
            flat_specs = list(chain.from_iterable([(spec_dep,), spec_indeps]))
            flat_deps = list(product((spec_dep,), spec_indeps))
            self.add_paramspecs(flat_specs)
            self.add_interdeps_by_type(flat_deps, type=type)

    def extend(
        self,
        dependencies: ParamSpecTree | None = None,
        inferences: ParamSpecTree | None = None,
        standalones: tuple[ParamSpecBase, ...] = (),
    ) -> Self:
        if dependencies is not None:
            self.add_dependencies(dependencies)
        if inferences is not None:
            self.add_inferences(inferences)
        if standalones != ():
            self.add_standalones(standalones)
        return self

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
    def standalones(self) -> tuple[ParamSpecBase, ...]:
        return tuple(
            [
                self._node_to_paramspec(node_id)
                for node_id, degree in graph.degree
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

    def remove(self, paramspec: ParamSpecBase) -> Self:
        self._graph.remove_node(paramspec.name)
        return self

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
