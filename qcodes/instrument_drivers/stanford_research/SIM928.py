from functools import partial
import logging
import numpy as np
import time

from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals
from qcodes.utils.deprecate import deprecate_moved_to_qcd

log = logging.getLogger(__name__)


@deprecate_moved_to_qcd(alternative="qcodes_contrib_drivers.drivers.StanfordResearchSystems.SIM928.SIM928")
class SIM928(VisaInstrument):
    """
    A driver for Stanford Research Systems SIM 928 DC source modules installed
    in a SIM900 mainframe.

    Args:
        name (str): An identifier for this instrument, particularly for
            attaching it to a ``Station``.
        address (str): The visa resource name to use to connect.
        slot_names (Dict[int]): An dictionary that optionally maps slot numbers
            to user-defined module names. Default ``{}``.
        timeout (int, float): Seconds to allow for responses. Default ``5``.
        metadata (Optional[Dict]): Additional static metadata to add to this
            instrument's JSON snapshot.
    """

    def __init__(self, name, address, slot_names=None, **kw):
        super().__init__(name, address=address, terminator='\n', **kw)

        if slot_names is None:
            self.slot_names = {}
        else:
            self.slot_names = slot_names
        self.module_nr = {}
        for i in self.slot_names:
            if self.slot_names[i] in self.module_nr:
                raise ValueError('Duplicate names in slot_names')
            self.module_nr[self.slot_names[i]] = i

        self.write('*DCL')  # device clear
        self.write('FLSH')  # flush port buffers
        self.write('SRST')  # SIM reset (causes 100 ms delay)
        time.sleep(0.5)

        self.modules = self.find_modules()
        for i in self.modules:
            self.write_module(i, 'TERM LF')
            module_name = self.slot_names.get(i, i)
            self.add_parameter('IDN_{}'.format(module_name),
                               label="IDN of module {}".format(module_name),
                               get_cmd=partial(self.get_module_idn, i))
            self.add_parameter('volt_{}'.format(module_name), unit='V',
                               label="Output voltage of module "
                                     "{}".format(module_name),
                               vals=vals.Numbers(-20, 20),
                               get_cmd=partial(self.get_voltage, i),
                               set_cmd=partial(self.set_voltage, i))
            self.add_parameter('volt_{}_step'.format(module_name), unit='V',
                               label="Step size when changing the voltage "
                                     "smoothly on module "
                                     "{}".format(module_name),
                               get_cmd=None, set_cmd=None,
                               vals=vals.Numbers(0, 20), initial_value=0.005)
        self.add_parameter('smooth_timestep', unit='s',
                           label="Delay between sending the write commands"
                                 "when changing the voltage smoothly",
                           get_cmd=None, set_cmd=None,
                           vals=vals.Numbers(0, 1), initial_value=0.05)

        super().connect_message()

    def get_module_idn(self, i):
        """
        Get the vendor, model, serial number and firmware version of a module.

        Args:
            i (int, str): Slot number or module name (as in ``slot_names``)
                of the module whose id is returned.

        Returns:
            A dict containing vendor, model, serial, and firmware.
        """
        if not isinstance(i, int):
            i = self.module_nr[i]
        idstr = self.ask_module(i, '*IDN?')
        idparts = [p.strip() for p in idstr.split(',', 3)]
        if len(idparts) < 4:
            idparts += [None] * (4 - len(idparts))
        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def find_modules(self):
        """
        Query the SIM900 mainframe for which slots have a SIM928 module present.

        Returns:
             A list of slot numbers where a SIM928 module is present (starting
                 from 1)
        """
        CTCR = self.ask('CTCR?')
        CTCR = int(CTCR) >> 1
        modules = []
        for i in range(1, 10):
            if CTCR & 1 != 0 and self.get_module_idn(i)['model'] == 'SIM928':
                modules.append(i)
            CTCR >>= 1
        return modules

    def ask_module(self, i, cmd):
        """
        Write a command string to a module and return a response.

        Args:
            i (int, str): Slot number or module name (as in ``slot_names``)
                of the module to ask from.
            cmd (str): The VISA query string.

        Returns:
            The response string from the module.
        """
        if not isinstance(i, int):
            i = self.module_nr[i]
        msg = 'SNDT {},"{}"'.format(i, cmd)
        self.write(msg)
        time.sleep(100e-3)
        msg = 'GETN? {},128'.format(i)
        msg = self.ask(msg)
        # first read consumes the terminator of the message from the submodule,
        # so we have a terminator from the message to us still in the input
        # buffer.
        self.visa_handle.read()

        if msg[:2] != '#3':
            raise RuntimeError('Unexpected format of answer: {}'.format(msg))
        return msg[5:]

    def write_module(self, i, cmd):
        """
        Write a command string to a module with NO response expected.

        Args:
            i (int, str): Slot number or module name (as in ``slot_names``)
                of the module to write to.
            cmd (str): The VISA command string.
        """
        if not isinstance(i, int):
            i = self.module_nr[i]
        self.write('SNDT {},"{}"'.format(i, cmd))

    def set_voltage(self, i, voltage):
        """
        Set the output voltage of a module.

        Args:
            i (int, str): Slot number or module name (as in ``slot_names``)
                of the module to set the voltage of.
            voltage (float): The value to set the voltage to.
        """
        if not isinstance(i, int):
            name = i
            i = self.module_nr[i]
        else:
            name = self.slot_names.get(i, i)
        self.write_module(i, 'VOLT {:.3f}'.format(voltage))
        self.parameters['volt_{}'.format(name)]._save_val(voltage)

    def get_voltage(self, i):
        """
        Get the output voltage of a module.

        Args:
           i (int, str): Slot number or module name (as in ``slot_names``)
               of the module to get the voltage of.

        Returns:
            The current voltage of module ``i`` as a ``float``.
        """
        if not isinstance(i, int):
            i = self.module_nr[i]
        return float(self.ask_module(i, 'VOLT?'))

    def set_smooth(self, voltagedict, equitime=False):
        """
        Set the voltages as specified in ``voltagedict` smoothly,
        by changing the output on each module at a rate
        ``volt_#_step/smooth_timestep``.

        Args:
            voltagedict (Dict[float]): A dictionary where keys are module slot
                numbers or names and values are the desired output voltages.
            equitime (bool): If ``True``, uses smaller step sizes for some of
                the modules so that all modules reach the desired value at the
                same time.
        """

        # convert voltagedict to contain module names only and validate inputs
        vdict = {}
        for i in voltagedict:
            if not isinstance(i, int):
                if self.module_nr[i] not in self.modules:
                    raise KeyError('There is no module named {}'.format(i))
                name = i
            else:
                if i not in self.modules:
                    raise KeyError('There is no module in slot {}'.format(i))
                name = self.slot_names.get(i, i)
            vdict[name] = voltagedict[i]
            self.parameters['volt_{}'.format(name)].validate(vdict[name])

        intermediate = []
        if equitime:
            maxsteps = 0
            deltav = {}
            for i in vdict:
                deltav[i] = vdict[i]-self.get('volt_{}'.format(i))
                stepsize = self.get('volt_{}_step'.format(i))
                steps = abs(int(np.ceil(deltav[i]/stepsize)))
                if steps > maxsteps:
                    maxsteps = steps
            for s in range(maxsteps):
                intermediate.append({})
                for i in vdict:
                    intermediate[-1][i] = vdict[i] - \
                                          deltav[i]*(maxsteps-s-1)/maxsteps
        else:
            done = []
            prevvals = {}
            for i in vdict:
                prevvals[i] = self.get('volt_{}'.format(i))
            while len(done) != len(vdict):
                intermediate.append({})
                for i in vdict:
                    if i in done:
                        continue
                    stepsize = self.get('volt_{}_step'.format(i))
                    deltav = vdict[i]-prevvals[i]
                    if abs(deltav) <= stepsize:
                        intermediate[-1][i] = vdict[i]
                        done.append(i)
                    elif deltav > 0:
                        intermediate[-1][i] = prevvals[i] + stepsize
                    else:
                        intermediate[-1][i] = prevvals[i] - stepsize
                    prevvals[i] = intermediate[-1][i]

        for voltages in intermediate:
            for i in voltages:
                self.set_voltage(i, voltages[i])
            time.sleep(self.smooth_timestep())

    def get_module_status(self, i):
        """
        Gets and clears the status bytes corresponding to the registers ESR,
        CESR and OVSR of module ``i``.

        Args:
            i (int, str): Slot number or module name (as in ``slot_names``)
                of the module to get the status of.

        Returns:
            int, int, int: The bytes corresponding to standard event,
            communication error and overload statuses of module ``i``
        """
        stdevent = self.ask_module(i, '*ESR?')
        commerr = self.ask_module(i, 'CESR?')
        overload = self.ask_module(i, 'OVSR?')
        return stdevent, commerr, overload

    def reset_module(self, i):
        """
        Sends the SIM Reset signal to module i.

        Causes a break signal (MARK level) to be asserted for 100 milliseconds
        to module i. Upon receiving the break signal the modul will flush its
        internal input buffer, reset its command parser, and default to 9600
        baud communications.

        Args:
            i (int, str): Slot number or module name (as in ``slot_names``)
                of the module to reset.
        """
        if not isinstance(i, int):
            i = self.module_nr[i]
        self.write('SRST {}'.format(i))


    def check_module_errors(self, i, raiseexc=True):
        """
        Check if any errors have occurred on module ``i`` and clear the status
        registers.

        Args:
            i (int, str): Slot number or module name (as in ``slot_names``)
                of the module to check the error of.
            raiseexc (bool): If true, raises an exception if any errors have
                occurred. Default ``True``.

        Returns:
            list[str]: A list of strings with the error messages that have
            occurred.
        """
        stdevent, commerr, overload = self.get_module_status(i)
        OPC, INP, QYE, DDE, EXE, CME, URQ, PON \
            = self.byte_to_bits(int(stdevent))
        PARITY, FRAME, NOISE, HWOVRN, OVR, RTSH, CTSH, DCAS \
            = self.byte_to_bits(int(commerr))
        Overload, Overvoltage, BatSwitch, BatFault, _, _, _, _ \
            = self.byte_to_bits(int(overload))

        errors = []
        warnings = []
        if INP:
            errors.append('Input Buffer Error.')
        if QYE:
            errors.append('Query Error.')
        if DDE:
            code = self.ask_module(i, 'LDDE?')
            errors.append('Device Dependant Error: {}.'.format(code))
        if EXE:
            code = self.ask_module(i, 'LEXE?')
            msg = {0: 'No error',
                   1: 'Illegal value',
                   2: 'Wrong token',
                   3: 'Invalid bit'}.get(int(code), 'Unknown')
            if int(code) > 3 or int(code) == 0:
                warnings.append('Execution Error: {} ({}).'.format(msg, code))
            else:
                errors.append('Execution Error: {} ({}).'.format(msg, code))
        if CME:
            code = self.ask_module(i, 'LCME?')
            msg = {0: 'No error',
                   1: 'Illegal command',
                   2: 'Undefined command',
                   3: 'Illegal query',
                   4: 'Illegal set',
                   5: 'Missing parameter(s)',
                   6: 'Extra parameter(s)',
                   7: 'Null parameter(s)',
                   8: 'Parameter buffer overflow',
                   9: 'Bad floating-point',
                   10: 'Bad integer',
                   11: 'Bad integer token',
                   12: 'Bad token value',
                   13: 'Bad hex block',
                   14: 'Unknown token'}.get(int(code), 'Unknown')
            if int(code) > 14 or int(code) == 0:
                warnings.append('Command Error: {} ({}).'.format(msg, code))
            else:
                errors.append('Command Error: {} ({}).'.format(msg, code))
        if PARITY:
            errors.append('Parity Error.')
        if FRAME:
            errors.append('Framing Error.')
        if NOISE:
            errors.append('Noise Error.')
        if HWOVRN:
            errors.append('Hardware Overrun.')
        if OVR:
            errors.append('Input Buffer Overrun.')
        if RTSH:
            errors.append('Undefined Error (RTSH).')
        if CTSH:
            errors.append('Undefined Error (CTSH).')
        if Overload:
            errors.append('Current Overload.')
        if Overvoltage:
            errors.append('Voltage Overload.')
        if BatFault:
            errors.append('Battery Fault.')

        if raiseexc:
            if len(errors) != 0:
                raise Exception(' '.join(errors + warnings))
        return errors + warnings

    @staticmethod
    def byte_to_bits(x):
        """
        Convert an integer to a list of bits

        Args:
            x (int): The number to convert.

        Returns:
            list[bool]: A list of the lowest 8 bits of ``x`` where ``True``
            represents 1 and ``False`` 0.
        """
        bits = []
        for _ in range(8):
            if x & 1 != 0:
                bits.append(True)
            else:
                bits.append(False)
            x >>= 1
        return bits
