import json

from qcodes.utils.helpers import full_class, named_repr


def test_full_class():
    j = json.JSONEncoder()
    assert full_class(j) == 'json.encoder.JSONEncoder'


def test_named_repr():
    j = json.JSONEncoder()
    id_ = id(j)
    j.name = 'Peppa'
    assert named_repr(j) == f'<json.encoder.JSONEncoder: Peppa at {id_}>'
