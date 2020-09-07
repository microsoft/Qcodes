from unittest import TestCase

from qcodes.tests.instrument_mocks import DummyInstrument
from qcodes.instrument.parameter import ScaledParameter, ManualParameter


class TestScaledParameter(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent_instrument = DummyInstrument('dummy')

    def setUp(self):
        self.target_name = 'target_parameter'
        self.target_label = 'Target Parameter'
        self.target_unit = 'V'

        self.target = ManualParameter(name=self.target_name,
                                      label=self.target_label,
                                      unit=self.target_unit,
                                      initial_value=1.0,
                                      instrument=self.parent_instrument)
        self.parent_instrument.add_parameter(self.target)
        self.scaler = ScaledParameter(self.target, division=1)

    @classmethod
    def tearDownClass(cls):
        cls.parent_instrument.close()
        del cls.parent_instrument

    def test_constructor(self):
        # Test the behaviour of the constructor

        # Require a wrapped parameter
        with self.assertRaises(TypeError):
            ScaledParameter()

        # Require a scaling factor
        with self.assertRaises(ValueError):
            ScaledParameter(self.target)

        # Require only one scaling factor
        with self.assertRaises(ValueError):
            ScaledParameter(self.target, division=1, gain=1)

    def test_namelabel(self):
        # Test handling of name and label

        # Test correct inheritance
        assert self.scaler.name == self.target_name + '_scaled'
        assert self.scaler.label == self.target_label + '_scaled'

        # Test correct name/label handling by the constructor
        scaled_name = 'scaled'
        scaled_label = "Scaled parameter"
        scaler2 = ScaledParameter(self.target, division=1,
                                  name=scaled_name, label=scaled_label)
        assert scaler2.name == scaled_name
        assert scaler2.label == scaled_label

    def test_unit(self):
        # Test handling of the units

        # Check if the unit is correctly inherited
        assert self.scaler.unit == 'V'

        # Check if we can change successfully the unit
        self.scaler.unit = 'A'
        assert self.scaler.unit == 'A'

        # Check if unit is correctly set in the constructor
        scaler2 = ScaledParameter(self.target, name='scaled_value',
                                  division=1, unit='K')
        assert scaler2.unit == 'K'

    def test_metadata(self):
        # Test the metadata

        test_gain = 3
        test_unit = 'V'
        self.scaler.gain = test_gain
        self.scaler.unit = test_unit

        # Check if relevant fields are present in the snapshot
        snap = self.scaler.snapshot()
        snap_keys = snap.keys()
        metadata_keys = snap['metadata'].keys()
        assert 'division' in snap_keys
        assert 'gain' in snap_keys
        assert 'role' in snap_keys
        assert 'unit' in snap_keys
        assert 'variable_multiplier' in metadata_keys
        assert 'wrapped_parameter' in metadata_keys
        assert 'wrapped_instrument' in metadata_keys

        # Check if the fields are correct
        assert snap['gain'] == test_gain
        assert snap['division'] == 1/test_gain
        assert snap['role'] == ScaledParameter.Role.GAIN
        assert snap['unit'] == test_unit
        assert snap['metadata']['variable_multiplier'] is False
        assert snap['metadata']['wrapped_parameter'] == self.target.name

    def test_wrapped_parameter(self):
        # Test if the target parameter is correctly inherited
        assert self.scaler.wrapped_parameter == self.target

    def test_divider(self):
        test_division = 10
        test_value = 5

        self.scaler.division = test_division
        self.scaler(test_value)
        assert self.scaler() == test_value
        assert self.target() == test_division * test_value
        assert self.scaler.gain == 1/test_division
        assert self.scaler.role == ScaledParameter.Role.DIVISION

    def test_multiplier(self):
        test_multiplier = 10
        test_value = 5

        self.scaler.gain = test_multiplier
        self.scaler(test_value)
        assert self.scaler() == test_value
        assert self.target() == test_value / test_multiplier
        assert self.scaler.division == 1/test_multiplier
        assert self.scaler.role == ScaledParameter.Role.GAIN

    def test_variable_gain(self):
        test_value = 5

        initial_gain = 2
        variable_gain_name = 'gain'
        gain = ManualParameter(name=variable_gain_name,
                               initial_value=initial_gain)
        self.scaler.gain = gain
        self.scaler(test_value)

        assert self.scaler() == test_value
        assert self.target() == test_value / initial_gain
        assert self.scaler.division == 1/initial_gain

        second_gain = 7
        gain(second_gain)
        # target value must change on scaler value change, not on gain/division
        assert self.target() == test_value / initial_gain
        self.scaler(test_value)
        assert self.target() == test_value / second_gain
        assert self.scaler.division == 1 / second_gain

        assert self.scaler.metadata['variable_multiplier'] == variable_gain_name
