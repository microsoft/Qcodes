import unittest

from qcodes.instrument_drivers.test import DriverTestCase
from qcodes.instrument.mock import MockInstrument, MockModel


class EmptyModel(MockModel):
    pass


class MockMock(MockInstrument):
    pass


@unittest.skip('just need this definition')
class HasNoDriver(DriverTestCase):
    noskip = True


class MockMock2(MockInstrument):
    pass


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
        cls.an_instrument = MockMock('a', model=cls.an_empty_model, server_name='')
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.an_empty_model.close()
        cls.an_instrument._manager.close()

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
