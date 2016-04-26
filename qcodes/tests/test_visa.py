from unittest import TestCase
import visa
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils.validators import Numbers


class MockVisa(VisaInstrument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_parameter('state',
                           get_cmd='STAT?', get_parser=float,
                           set_cmd='STAT:{:.3f}',
                           vals=Numbers(-20, 20))

    def set_address(self, address):
        self.visa_handle = MockVisaHandle()


class MockVisaHandle:
    '''
    mock the API needed for a visa handle that throws lots of errors:
    - any write command sets a single "state" variable to a float
      after the last : in the command
      - a negative number results in an error raised here
      - 0 results in a return code for visa timeout

    - any ask command returns the state
      - a state > 10 throws an error
    '''
    def __init__(self):
        self.state = 0

    def clear(self):
        self.state = 0

    def close(self):
        # make it an error to ask or write after close
        self.write = None
        self.ask = None

    def write(self, cmd):
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
        if self.state > 10:
            raise ValueError("I'm out of fingers")
        return self.state


class TestVisaInstrument(TestCase):
    def test_default_server_name(self):
        dsn = VisaInstrument.default_server_name
        self.assertEqual(dsn(), 'VisaServer')
        self.assertEqual(dsn(address='Gpib::10'), 'GPIBServer')
        self.assertEqual(dsn(address='aSRL4'), 'SerialServer')

    def test_ask_write_local(self):
        mv = MockVisa('Joe', server_name=None)

        # test normal ask and write behavior
        mv.state.set(2)
        self.assertEqual(mv.state.get(), 2)
        mv.state.set(3.4567)
        self.assertEqual(mv.state.get(), 3.457)  # driver rounds to 3 digits

        # test ask and write errors
        with self.assertRaises(ValueError):
            mv.state.set(-10)
        self.assertEqual(mv.state.get(), -10)  # set still happened

        with self.assertRaises(visa.VisaIOError):
            mv.state.set(0)
        self.assertEqual(mv.state.get(), 0)

        mv.state.set(15)
        with self.assertRaises(ValueError):
            mv.state.get()

    def test_ask_write_server(self):
        # same thing as above but Joe is on a server now...
        mv = MockVisa('Joe')

        # test normal ask and write behavior
        mv.state.set(2)
        self.assertEqual(mv.state.get(), 2)
        mv.state.set(3.4567)
        self.assertEqual(mv.state.get(), 3.457)  # driver rounds to 3 digits

        # test ask and write errors
        with self.assertRaises(ValueError):
            mv.state.set(-10)
        self.assertEqual(mv.state.get(), -10)  # set still happened

        # only built-in errors get propagated to the main process as the
        # same type. Perhaps we could include some more common ones like
        # this (visa.VisaIOError) in the future...
        with self.assertRaises(RuntimeError):
            mv.state.set(0)
        self.assertEqual(mv.state.get(), 0)

        mv.state.set(15)
        with self.assertRaises(ValueError):
            mv.state.get()
