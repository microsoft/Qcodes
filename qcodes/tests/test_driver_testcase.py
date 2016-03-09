import unittest

from qcodes.instrument_drivers.test import DriverTestCase
from qcodes.instrument.mock import MockInstrument, MockModel


class EmptyModel(MockModel):
    pass


class MockMock(MockInstrument):
    def __init__(self, name, model):
        super().__init__(name, model=model)


@unittest.skip('just need this definition')
class HasNoDriver(DriverTestCase):
    noskip = True


class MockMock2(MockInstrument):
    def __init__(self, name, model):
        super().__init__(name, model=model)


@unittest.skip('just need this definition')
class HasNoInstances(DriverTestCase):
    noskip = True
    driver = MockMock2


class TestDriverTestCase(DriverTestCase):
    driver = MockMock
    noskip = True

    @classmethod
    def setUpClass(cls):
        cls.an_empty_model = EmptyModel()
        cls.an_instrument = MockMock('a', cls.an_empty_model)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.an_empty_model.halt()
        cls.an_instrument.connection.manager.halt()

    def test_instance_found(self):
        self.assertEqual(self.instrument, self.an_instrument)

    def test_no_driver(self):
        with self.assertRaises(TypeError):
            HasNoDriver.setUpClass()

    def test_no_instances(self):
        baseMock = MockInstrument('not the same class',
                                  model=self.an_empty_model)
        self.assertIn(baseMock, MockInstrument.instances())

        with self.assertRaises(ValueError):
            HasNoInstances.setUpClass()
