import json

from qcodes.parameters.named_repr import named_repr
from qcodes.utils import full_class


def test_full_class():
    j = json.JSONEncoder()
    assert full_class(j) == "json.encoder.JSONEncoder"


def test_named_repr():
    j = json.JSONEncoder()
    id_ = id(j)
    j.name = "Peppa"
    assert named_repr(j) == f"<json.encoder.JSONEncoder: Peppa at {id_}>"
