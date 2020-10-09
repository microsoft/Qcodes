import pytest

from qcodes.utils.metadata import Metadatable, diff_param_values


class HasSnapshotBase(Metadatable):
    def snapshot_base(self, update=False,
                      params_to_skip_update=None):
        return {'cheese': 'gruyere'}


class HasSnapshot(Metadatable):
    # Users shouldn't do this... but we'll test its behavior
    # for completeness
    def snapshot(self, update=False):
        return {'fruit': 'kiwi'}


DATASETLEFT = {
    "station": {
        "parameters": {
            "apple": {
                "value": "orange"
            }
        },
        "instruments": {
            "correct": {
                "parameters": {
                    "horse": {
                        "value": "battery"
                    },
                    "left": {
                        "value": "only"
                    }
                }
            },
            "another": {
                "parameters": {}
            }
        }
    }
}
DATASETRIGHT = {
    "station": {
        "parameters": {
            "apple": {
                "value": "grape"
            }
        },
        "instruments": {
            "correct": {
                "parameters": {
                    "horse": {
                        "value": "staple"
                    },
                    "right": {
                        "value": "only"
                    }
                }
            },
            "another": {
                "parameters": {
                    "pi": {
                        "value": 3.1
                    }
                }
            }
        }
    }
}


def test_load():
    m = Metadatable()
    assert m.metadata == {}
    m.load_metadata({1: 2, 3: 4})
    assert m.metadata == {1: 2, 3: 4}
    m.load_metadata({1: 5})
    assert m.metadata == {1: 5, 3: 4}


def test_init():
    with pytest.raises(TypeError):
        Metadatable(metadata={2: 3}, not_metadata={4: 5})

    m = Metadatable(metadata={2: 3})
    assert m.metadata == {2: 3}


def test_snapshot():
    m = Metadatable(metadata={6: 7})
    assert m.snapshot_base() == {}
    assert m.snapshot() == {'metadata': {6: 7}}
    del m.metadata[6]
    assert m.snapshot() == {}

    sb = HasSnapshotBase(metadata={7: 8})
    assert sb.snapshot_base() == {'cheese': 'gruyere'}
    assert sb.snapshot() == \
           {'cheese': 'gruyere', 'metadata': {7: 8}}
    del sb.metadata[7]
    assert sb.snapshot() == sb.snapshot_base()

    s = HasSnapshot(metadata={8: 9})
    assert s.snapshot() == {'fruit': 'kiwi'}
    assert s.snapshot_base() == {}
    assert s.metadata == {8: 9}


def test_dataset_diff():
    diff = diff_param_values(DATASETLEFT, DATASETRIGHT)
    assert diff.changed == {
            "apple": ("orange", "grape"),
            ("correct", "horse"): ("battery", "staple"),
        }
    assert diff.left_only == {
            ("correct", "left"): "only"
        }
    assert diff.right_only == {
            ("correct", "right"): "only",
            ("another", "pi"): 3.1,
        }


def test_station_diff():
    left = DATASETLEFT["station"]
    right = DATASETRIGHT["station"]

    diff = diff_param_values(left, right)
    assert diff.changed == {
            "apple": ("orange", "grape"),
            ("correct", "horse"): ("battery", "staple")
        }
    assert diff.left_only == {
            ("correct", "left"): "only"
        }
    assert diff.right_only == {
            ("correct", "right"): "only",
            ("another", "pi"): 3.1,
        }


def test_instrument_diff():
    left = DATASETLEFT["station"]["instruments"]["another"]
    right = DATASETRIGHT["station"]["instruments"]["another"]

    diff = diff_param_values(left, right)

    assert diff.changed == {}
    assert diff.left_only == {}
    assert diff.right_only == {
             "pi": 3.1
        }
