import json
import random
import re
from datetime import datetime, timedelta

import hypothesis.strategies as hst
import pytest
from hypothesis import given, settings

from qcodes.dataset.guids import generate_guid
from qcodes.dataset.linked_datasets.links import (
    Link,
    link_to_str,
    links_to_str,
    str_to_link,
    str_to_links,
)


def generate_some_links(N: int) -> list[Link]:
    """
    Generate N links with the same head
    """

    def _timestamp() -> int:
        """
        return a random timestamp that is approximately
        one day in the past.
        """
        timestamp = datetime.now() - timedelta(days=1, seconds=random.randint(1, 1000))
        return int(round(timestamp.timestamp() * 1000))

    known_types = ("fit", "analysis", "step")
    known_descs = (
        "A second-order fit",
        "Manual analysis (see notebook)",
        "Step 3 in the characterisation",
    )

    head_guid = generate_guid(_timestamp())
    head_guids = [head_guid] * N
    tail_guids = [generate_guid(_timestamp()) for _ in range(N)]
    edge_types = [known_types[i % len(known_types)] for i in range(N)]
    descriptions = [known_descs[i % len(known_descs)] for i in range(N)]

    zipattrs = zip(head_guids, tail_guids, edge_types, descriptions)

    links = [Link(hg, tg, n, d) for hg, tg, n, d in zipattrs]

    return links


def test_link_construction_passes() -> None:
    head_guid = generate_guid()
    tail_guid = generate_guid()
    edge_type = "fit"
    description = "We did a second order fit with math"

    link = Link(head_guid, tail_guid, edge_type)

    assert link.head == head_guid
    assert link.tail == tail_guid
    assert link.edge_type == edge_type
    assert link.description == ""

    link = Link(head_guid, tail_guid, edge_type, description)

    assert link.description == description


@settings(max_examples=20)
@given(not_guid=hst.text())
def test_link_construction_raises(not_guid) -> None:
    head_guid = generate_guid()
    tail_guid = generate_guid()
    edge_type = "fit"

    match = re.escape(
        f"The guid given for head is not a valid guid. Received {not_guid}."
    )
    with pytest.raises(ValueError, match=match):
        Link(not_guid, tail_guid, edge_type)

    match = re.escape(
        f"The guid given for tail is not a valid guid. Received {not_guid}"
    )
    with pytest.raises(ValueError, match=match):
        Link(head_guid, not_guid, edge_type)


def test_link_to_str() -> None:
    head_guid = generate_guid()
    tail_guid = generate_guid()
    edge_type = "test"
    description = "used in test_link_to_str"

    link = Link(head_guid, tail_guid, edge_type, description)

    expected_str = json.dumps(
        {
            "head": head_guid,
            "tail": tail_guid,
            "edge_type": edge_type,
            "description": description,
        }
    )

    assert link_to_str(link) == expected_str


def test_str_to_link() -> None:
    head_guid = generate_guid()
    tail_guid = generate_guid()
    edge_type = "test"
    description = "used in test_str_to_link"

    lstr = json.dumps(
        {
            "head": head_guid,
            "tail": tail_guid,
            "edge_type": edge_type,
            "description": description,
        }
    )

    expected_link = Link(head_guid, tail_guid, edge_type, description)

    assert str_to_link(lstr) == expected_link


def test_link_to_string_and_back() -> None:
    head_guid = generate_guid()
    tail_guid = generate_guid()
    edge_type = "analysis"
    description = "hyper-spectral quantum blockchain ML"

    link = Link(head_guid, tail_guid, edge_type, description)

    lstr = link_to_str(link)
    newlink = str_to_link(lstr)

    assert newlink == link


@settings(max_examples=5, deadline=1200)
@given(N=hst.integers(min_value=0, max_value=25))
def test_links_to_str_and_back(N) -> None:
    links = generate_some_links(N)

    new_links = str_to_links(links_to_str(links))

    assert new_links == links
