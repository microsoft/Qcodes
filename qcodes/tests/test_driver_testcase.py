import unittest

from qcodes.instrument_drivers.test import DriverTestCase
from qcodes.instrument.mock import MockInstrument


class EmptyModel:
    '''
    just enough to let us instantiate a mock instrument
    '''
    write = 'don\'t even try'
    ask = 'no questions!'


class MockMock(MockInstrument):
    def __init__(self, name):
        super().__init__(name, model=EmptyModel())


instrument = MockMock('a')


@unittest.skip('just need this definition')
class HasNoDriver(DriverTestCase):
    noskip = True


class MockMock2(MockInstrument):
    def __init__(self, name):
        super().__init__(name, model=EmptyModel())


@unittest.skip('just need this definition')
class HasNoInstances(DriverTestCase):
    noskip = True
    driver = MockMock2


class TestDriverTestCase(DriverTestCase):
    driver = MockMock
    noskip = True

    def test_instance_found(self):
        self.assertEqual(self.instrument, instrument)

    def test_no_driver(self):
        with self.assertRaises(TypeError):
            HasNoDriver.setUpClass()

    def test_no_instances(self):
        baseMock = MockInstrument('not the same class', model=EmptyModel())
        self.assertIn(baseMock, MockInstrument.instances())

        with self.assertRaises(ValueError):
            HasNoInstances.setUpClass()
