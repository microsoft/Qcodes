from unittest import TestCase
from unittest.mock import patch
import visa
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils.validators import Numbers
import warnings


class MockVisa(VisaInstrument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_parameter('state',
                           get_cmd='STAT?', get_parser=float,
                           set_cmd='STAT:{:.3f}',
                           vals=Numbers(-20, 20))

    def set_address(self, address):
        self.visa_handle = MockVisaHandle()
        self.visabackend = self.visalib


class MockVisaHandle:
    """
    mock the API needed for a visa handle that throws lots of errors:

    - any write command sets a single "state" variable to a float
      after the last : in the command
    - a negative number results in an error raised here
    - 0 results in a return code for visa timeout
    - any ask command returns the state
    - a state > 10 throws an error
    """
    def __init__(self):
        self.state = 0
        self.closed = False

    def clear(self):
        self.state = 0

    def close(self):
        # make it an error to ask or write after close
        self.closed = True

    def write(self, cmd):
        if self.closed:
            raise RuntimeError("Trying to write to a closed instrument")
        num = float(cmd.split(':')[-1])
        self.state = num

        if num < 0:
            raise ValueError('be more positive!')

        if num == 0:
            ret_code = visa.constants.VI_ERROR_TMO
        else:
            ret_code = 0

        return len(cmd), ret_code

    def ask(self, cmd):
        if self.closed:
            raise RuntimeError("Trying to ask a closed instrument")
        if self.state > 10:
            raise ValueError("I'm out of fingers")
        return self.state

    def query(self, cmd):
        if self.state > 10:
            raise ValueError("I'm out of fingers")
        return self.state


class TestVisaInstrument(TestCase):
    # error args for set(-10)
    args1 = [
        'be more positive!',
        "writing 'STAT:-10.000' to <MockVisa: Joe>",
        'setting Joe_state to -10'
    ]

    # error args for set(0)
    args2 = [
        "writing 'STAT:0.000' to <MockVisa: Joe>",
        'setting Joe_state to 0'
    ]

    # error args for get -> 15
    args3 = [
        "I'm out of fingers",
        "asking 'STAT?' to <MockVisa: Joe>",
        'getting Joe_state'
    ]

    def test_ask_write_local(self):
        mv = MockVisa('Joe', 'none_address')
        self.addCleanup(mv.close)

        # test normal ask and write behavior
        mv.state.set(2)
        self.assertEqual(mv.state.get(), 2)
        mv.state.set(3.4567)
        self.assertEqual(mv.state.get(), 3.457)  # driver rounds to 3 digits

        # test ask and write errors
        with self.assertRaises(ValueError) as e:
            mv.state.set(-10)
        for arg in self.args1:
            self.assertIn(arg, e.exception.args)
        self.assertEqual(mv.state.get(), -10)  # set still happened

        with self.assertRaises(visa.VisaIOError) as e:
            mv.state.set(0)
        for arg in self.args2:
            self.assertIn(arg, e.exception.args)
        self.assertEqual(mv.state.get(), 0)

        mv.state.set(15)
        with self.assertRaises(ValueError) as e:
            mv.state.get()
        for arg in self.args3:
            self.assertIn(arg, e.exception.args)

    @patch('qcodes.instrument.visa.visa.ResourceManager')
    def test_visa_backend(self, rm_mock):
        address_opened = [None]

        class MockBackendVisaInstrument(VisaInstrument):
            visa_handle = MockVisaHandle()

        class MockRM:
            def open_resource(self, address):
                address_opened[0] = address
                return MockVisaHandle()

        rm_mock.return_value = MockRM()

        inst1 = MockBackendVisaInstrument('name', address='None')
        self.addCleanup(inst1.close)
        self.assertEqual(rm_mock.call_count, 1)
        self.assertEqual(rm_mock.call_args, ((),))
        self.assertEqual(address_opened[0], 'None')
        inst1.close()

        inst2 = MockBackendVisaInstrument('name2', address='ASRL2')
        self.addCleanup(inst2.close)
        self.assertEqual(rm_mock.call_count, 2)
        self.assertEqual(rm_mock.call_args, ((),))
        self.assertEqual(address_opened[0], 'ASRL2')
        inst2.close()

        # this one raises a warning
        with warnings.catch_warnings(record=True) as w:
            inst3 = MockBackendVisaInstrument('name3', address='ASRL3@py')
            self.addCleanup(inst3.close)
            self.assertTrue(len(w) == 1)
            self.assertTrue('use the visalib' in str(w[-1].message))

        self.assertEqual(rm_mock.call_count, 3)
        self.assertEqual(rm_mock.call_args, (('@py',),))
        self.assertEqual(address_opened[0], 'ASRL3')
        inst3.close()

        # this one doesn't
        inst4 = MockBackendVisaInstrument('name4',
                                          address='ASRL4', visalib='@py')
        self.addCleanup(inst4.close)
        self.assertEqual(rm_mock.call_count, 4)
        self.assertEqual(rm_mock.call_args, (('@py',),))
        self.assertEqual(address_opened[0], 'ASRL4')
        inst4.close()


def test_visa_instr_metadata(request):
    metadatadict = {'foo': 'bar'}
    mv = MockVisa('Joe', 'none_adress', metadata=metadatadict)
    request.addfinalizer(mv.close)
    assert mv.metadata == metadatadict
