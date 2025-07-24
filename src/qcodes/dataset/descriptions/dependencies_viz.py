"""
Utils for visualizing InterDependencies_. Note that these only work in
Jupyter and require you to have ipycytoscape installed. This can be
installed with ``pip install qcodes[live_plotting]``
"""

from typing import TYPE_CHECKING, Any

import ipycytoscape
import ipywidgets
import networkx as nx

if TYPE_CHECKING:
    from qcodes.dataset.descriptions.dependencies import InterDependencies_

INTERDEPENDENCIES_STYLE = [
    {
        "selector": "node",
        "css": {
            "content": "data(id)",
            "text-valign": "center",
            "font-weight": 10,
            "color": "white",
            "text-outline-width": 1,
            "text-outline-color": "#11479e",
            "background-color": "#8aa8d8",
            "border-color": "#11479e",
            "border-width": 1.5,
            "border-style": "solid",
        },
    },
    {
        "selector": "node[node_type='dependency']",
        "css": {
            "text-outline-color": "#089222",
            "background-color": "#90C79A",
            "border-color": "#089222",
        },
    },
    {
        "selector": "node[node_type='inference']",
        "css": {
            "text-outline-color": "#920808",
            "background-color": "#D37B7B",
            "border-color": "#920808",
        },
    },
    {
        "selector": "edge",
        "style": {
            "width": 4,
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
        },
    },
    {
        "selector": "edge[interdep_type = 'depends_on'],  ",
        "style": {
            "line-color": "#089222",
            "target-arrow-color": "#089222",
        },
    },
    {
        "selector": "edge[interdep_type = 'inferred_from']",
        "style": {
            "line-color": "#920808",
            "target-arrow-color": "#920808",
        },
    },
]


def draw_interdepenencies(interdeps: "InterDependencies_") -> ipywidgets.HBox:
    graphwidget = ipycytoscape.CytoscapeWidget()
    graphwidget.graph.add_graph_from_json(interdeps_to_ipycytoscape_json(interdeps))
    graphwidget.set_style(INTERDEPENDENCIES_STYLE)
    graphwidget.user_zooming_enabled = True
    graphwidget.min_zoom = 0.2
    graphwidget.max_zoom = 5
    graphwidget.set_layout(name="cola", animate=False, avoidOverlap=True)
    graphwidget.wheel_sensitivity = 0.1
    return ipywidgets.HBox([graphwidget])


def interdeps_to_ipycytoscape_json(
    interdeps: "InterDependencies_",
) -> dict[str, list[dict[str, Any]]]:
    graph_json: dict[str, list[dict[str, Any]]] = nx.cytoscape_data(interdeps.graph)[
        "elements"
    ]
    interdeps_dict = interdeps._to_dict()
    dependencies = list(interdeps_dict["dependencies"].keys())
    inferences = list(interdeps_dict["inferences"].keys())

    for node_dict in graph_json["nodes"]:
        if node_dict["data"]["id"] in dependencies:
            node_dict["data"]["node_type"] = "dependency"
        elif node_dict["data"]["id"] in inferences:
            node_dict["data"]["node_type"] = "inference"
    return graph_json
