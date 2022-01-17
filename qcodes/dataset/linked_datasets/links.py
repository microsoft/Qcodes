"""
This module defines the Link dataclass as well as two functions to read and
write an Link object to/from string, respectively
"""
from typing import List
from dataclasses import dataclass, asdict
import json
from qcodes.dataset.guids import validate_guid_format


@dataclass
class Link:
    """
    Class to represent a link between two datasets. The link is a little graph
    with two nodes (head and tail) and a directed edge.

    Attributes:
        head: a guid representing the head of the graph
        tail: a guid representing the tail of the graph
        edge_type: a name to represent the type of the edge
        description: free-form optional field add a description of the graph
    """
    head: str
    tail: str
    edge_type: str
    description: str = ""

    @staticmethod
    def validate_node(node_guid: str, node: str) -> None:
        """
        Validate that the guid given is a valid guid.

        Args:
            node_guid: the guid
            node: either "head" or "tail"
        """
        try:
            validate_guid_format(node_guid)
        except ValueError:
            raise ValueError(
                f'The guid given for {node} is not a valid guid. Received '
                f'{node_guid}.')

    def __post_init__(self) -> None:
        self.validate_node(self.head, "head")
        self.validate_node(self.tail, "tail")


def link_to_str(link: Link) -> str:
    """
    Convert a Link to a string
    """
    return json.dumps(asdict(link))


def str_to_link(string: str) -> Link:
    """
    Convert a string to a Link
    """
    ldict = json.loads(string)
    link = Link(**ldict)
    return link


def links_to_str(links: List[Link]) -> str:
    """
    Convert a list of links to string. Note that this is the output that gets
    stored in the DB file
    """
    output = json.dumps([link_to_str(link) for link in links])
    return output


def str_to_links(links_string: str) -> List[Link]:
    """
    Convert a string into a list of Links
    """
    if links_string == '[]':
        return []
    link_dicts = [json.loads(l_str) for l_str in json.loads(links_string)]
    links = [Link(**ld) for ld in link_dicts]
    return links
