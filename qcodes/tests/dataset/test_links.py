import re

import pytest
import hypothesis.strategies as hst
from hypothesis import given, settings


from qcodes.dataset.linked_datasets.links import Link
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
