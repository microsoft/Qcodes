import logging
import numpy as np
from functools import partial
import time
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ManualParameter


class SIM928(VisaInstrument):
    """
    A driver for Stanford Research Systems SIM 928 DC source modules installed
    in a SIM900 mainframe.
    """

    def __init__(self, name, address, **kw):
        super().__init__(name, address=address, terminator='\n', **kw)

        self.write('*DCL')  # device clear
        self.write('FLSH')  # flush port buffers
        self.write('SRST')  # SIM reset (causes 100 ms delay)
        time.sleep(0.5)

        self.modules = self.find_modules()
        for i in self.modules:
            self.write_module(i, 'TERM LF')
            self.add_parameter('IDN{}'.format(i),
                               label="IDN of module {}".format(i),
                               get_cmd=partial(self.get_idn, i))
            self.add_parameter('voltage{}'.format(i), unit='V',
                               label="Output voltage of module {}".format(i),
                               vals=vals.Numbers(-20, 20),
                               get_cmd=partial(self.get_voltage, i),
                               set_cmd=partial(self.set_voltage, i))
            self.add_parameter('voltage{}_step'.format(i), unit='V',
                               label="The step size when changing the voltage "
                                     "smoothly on module {}".format(i),
                               parameter_class=ManualParameter,
                               vals=vals.Numbers(0, 20), initial_value=0.005)
        self.add_parameter('smooth_timestep', unit='s',
                           label="The delay between sending the write commands"
                                 "when changing the voltage smoothly",
                           parameter_class=ManualParameter,
                           vals=vals.Numbers(0,1), initial_value=0.05)

        super().connect_message()

    def get_idn(self, i=None):
        """
        i:
            Slot of the module whose id is returned
        Returns:
            A dict containing vendor, model, serial, and firmware.
        """
        try:
            idstr = ''  # in case self.ask fails
            if i is None:
                idstr = self.ask('*IDN?')
            else:
                idstr = self.ask_module(i, '*IDN?')
            # form is supposed to be comma-separated, but we've seen
            # other separators occasionally
            for separator in ',;:':
                # split into no more than 4 parts, so we don't lose info
                idparts = [p.strip() for p in idstr.split(separator, 3)]
                if len(idparts) > 1:
                    break
            # in case parts at the end are missing, fill in None
            if len(idparts) < 4:
                idparts += [None] * (4 - len(idparts))
        except:
            logging.warning('Error getting or interpreting *IDN?: ' +
                            repr(idstr))
            idparts = [None, None, None, None]

        # some strings include the word 'model' at the front of model
        if str(idparts[1]).lower().startswith('model'):
            idparts[1] = str(idparts[1])[5:].strip()

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def find_modules(self):
        """
        Queries the SIM900 mainframe, in which slots there is a module present.
        :return: a list of slot numbers where a module is present (starting
                 from 1)
        """
        CTCR = self.ask('CTCR?')
        CTCR = int(CTCR) >> 1
        modules = []
        for i in range(1,10):
            if CTCR & 1 != 0:
                modules.append(i)
            CTCR >>= 1
        return modules

    def ask_module(self, i, cmd):
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
        self.write('SNDT {},"{}"'.format(i, cmd))

    def set_voltage(self, i, voltage):
        self.write_module(i, 'VOLT {:.3f}'.format(voltage))
        self.parameters['voltage{}'.format(i)]._save_val(voltage)

    def get_voltage(self, i):
        return float(self.ask_module(i, 'VOLT?'))

    def set_smooth(self, voltagedict, equitime=False, verbose=False):
        """
        Sets the voltages as specified in voltagedict, by changing the output
        on each module at a rate voltage#_step/smooth_timestep.

        voltagedict:
            a dictionary where keys are module slot numbers and values are the
            desired output voltages.
        equitime:
            if True, uses smaller step sizes for some of the modules so that all
            modules reach the desired value at the same time
        """

        for i in voltagedict:
            if i not in self.modules:
                raise KeyError('There is no module in slot {}'.format(i))
            self.parameters['voltage{}'.format(i)].validate(voltagedict[i])

        intermediate = []
        if equitime:
            maxsteps = 0
            deltav = {}
            for i in voltagedict:
                deltav[i] = voltagedict[i]-self.get('voltage{}'.format(i))
                stepsize = self.get('voltage{}_step'.format(i))
                steps = abs(int(np.ceil(deltav[i]/stepsize)))
                if steps > maxsteps:
                    maxsteps = steps
            for s in range(maxsteps):
                intermediate.append({})
                for i in voltagedict:
                    intermediate[-1][i] = voltagedict[i] - \
                                          deltav[i]*(maxsteps-s-1)/maxsteps
        else:
            done = []
            prevvals = {}
            for i in voltagedict:
                prevvals[i] = self.get('voltage{}'.format(i))
            while len(done) != len(voltagedict):
                intermediate.append({})
                for i in voltagedict:
                    if i in done:
                        continue
                    stepsize = self.get('voltage{}_step'.format(i))
                    deltav = voltagedict[i]-prevvals[i]
                    if abs(deltav) <= stepsize:
                        intermediate[-1][i] = voltagedict[i]
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
        stdevent = self.ask_module(i, '*ESR?')
        commerr = self.ask_module(i, 'CESR?')
        overload = self.ask_module(i, 'OVSR?')
        return stdevent, commerr, overload

    def check_module_errors(self, i, raiseexc=True):
        stdevent, commerr, overload = self.get_module_status(i)
        OPC, INP, QYE, DDE, EXE, CME, URQ, PON \
            = self.byte_to_bits(int(stdevent))
        PARITY, FRAME, NOISE, HWOVRN, OVR, RTSH, CTSH, DCAS \
            = self.byte_to_bits(int(commerr))
        Overload, Overvoltage, BatSwitch, BatFault, _, _, _, _ \
            = self.byte_to_bits(int(overload))

        errors = []
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
                   3: 'Invalid bit'}.get(code, 'Unknown')
            errors.append('Execution Error: {} ({})'.format(msg, code))
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
                   14: 'Unknown token'}.get(code, 'Unknown')
            errors.append('Command Error: {} ({})'.format(msg, code))
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
                raise Exception(' '.join(errors))
        return errors

    @classmethod
    def byte_to_bits(cls, x):
        bits = []
        for i in range(8):
            if x & 1 != 0:
                bits.append(True)
            else:
                bits.append(False)
            x >>= 1
        return bits