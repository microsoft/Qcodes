import re
import json

import pytest
import hypothesis.strategies as hst
from hypothesis import given, settings


from qcodes.dataset.linked_datasets.links import (
    Link, link_to_str, str_to_link)
from qcodes.dataset.guids import generate_guid


def test_link_construction_passes():
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
def test_link_construction_raises(not_guid):
    head_guid = generate_guid()
    tail_guid = generate_guid()
    edge_type = "fit"
    description = "We did a second order fit with math"

    match = re.escape(
        f'The guid given for head is not a valid guid. Received '
        f'{not_guid}.')
    with pytest.raises(ValueError, match=match):
        Link(not_guid, tail_guid, edge_type)

    match = re.escape(
        f'The guid given for tail is not a valid guid. Received '
        f'{not_guid}')
    with pytest.raises(ValueError, match=match):
        Link(head_guid, not_guid, edge_type)


def test_link_to_str():
    head_guid = generate_guid()
    tail_guid = generate_guid()
    edge_type = "test"
    description = "used in test_link_to_str"

    link = Link(head_guid, tail_guid, edge_type, description)

    expected_str = json.dumps({"head": head_guid,
                               "tail": tail_guid,
                               "edge_type": edge_type,
                               "description": description})

    assert link_to_str(link) == expected_str


def test_str_to_link():
    head_guid = generate_guid()
    tail_guid = generate_guid()
    edge_type = "test"
    description = "used in test_str_to_link"


    lstr = json.dumps({"head": head_guid,
                       "tail": tail_guid,
                       "edge_type": edge_type,
                       "description": description})

    expected_link = Link(head_guid, tail_guid, edge_type, description)

    assert str_to_link(lstr) == expected_link


def test_link_to_string_and_back():
    head_guid = generate_guid()
    tail_guid = generate_guid()
    edge_type = "analysis"
    description = "hyper-spectral quantum blockchain ML"

    link = Link(head_guid, tail_guid, edge_type, description)

    lstr = link_to_str(link)
    newlink = str_to_link(lstr)

    assert newlink == link
