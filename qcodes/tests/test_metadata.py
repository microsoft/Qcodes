from unittest import TestCase

from qcodes.utils.metadata import Metadatable, diff_param_values


class TestMetadatable(TestCase):
    def test_load(self):
        m = Metadatable()
        self.assertEqual(m.metadata, {})
        m.load_metadata({1: 2, 3: 4})
        self.assertEqual(m.metadata, {1: 2, 3: 4})
        m.load_metadata({1: 5})
        self.assertEqual(m.metadata, {1: 5, 3: 4})

    def test_init(self):
        with self.assertRaises(TypeError):
            Metadatable(metadata={2: 3}, not_metadata={4: 5})

        m = Metadatable(metadata={2: 3})
        self.assertEqual(m.metadata, {2: 3})

    class HasSnapshotBase(Metadatable):
        def snapshot_base(self, update=False):
            return {'cheese': 'gruyere'}

    class HasSnapshot(Metadatable):
        # Users shouldn't do this... but we'll test its behavior
        # for completeness
        def snapshot(self, update=False):
            return {'fruit': 'kiwi'}

    def test_snapshot(self):
        m = Metadatable(metadata={6: 7})
        self.assertEqual(m.snapshot_base(), {})
        self.assertEqual(m.snapshot(), {'metadata': {6: 7}})
        del m.metadata[6]
        self.assertEqual(m.snapshot(), {})

        sb = self.HasSnapshotBase(metadata={7: 8})
        self.assertEqual(sb.snapshot_base(), {'cheese': 'gruyere'})
        self.assertEqual(sb.snapshot(),
                         {'cheese': 'gruyere', 'metadata': {7: 8}})
        del sb.metadata[7]
        self.assertEqual(sb.snapshot(), sb.snapshot_base())

        s = self.HasSnapshot(metadata={8: 9})
        self.assertEqual(s.snapshot(), {'fruit': 'kiwi'})
        self.assertEqual(s.snapshot_base(), {})
        self.assertEqual(s.metadata, {8: 9})

    def test_diff(self):
        left = {
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
                    }
                }
            }
        }
        right = {
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
                    }
                }
            }
        }

        diff = diff_param_values(left, right)
        self.assertEqual(
            diff.changed, {
                "apple": ("orange", "grape"),
                ("correct", "horse"): ("battery", "staple")
            }
        )
        self.assertEqual(
            diff.left_only, {
                ("correct", "left"): "only"
            }
        )
        self.assertEqual(
            diff.right_only, {
                ("correct", "right"): "only"
            }
        )
