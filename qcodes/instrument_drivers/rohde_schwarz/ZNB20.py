import time
import logging
import numpy as np

from qcodes import VisaInstrument
from qcodes.utils import validators as vals
from cmath import phase
from qcodes import MultiParameter


class FrequencySweep(MultiParameter):
    """
    Hardware controlled parameter class for Rohde Schwarz RSZNB20/8 trace.

    Instrument returns an list of transmission data in the form of a list of
    complex numbers taken from a frequency sweep.

    Args:
      name: parameter name
      instrument: instrument the parameter belongs to
      start: starting frequency of sweep
      stop: ending frequency of sweep
      npts: numper of points in frequency sweep

     Methods:
        set_sweep(start, stop, npts): sets the shapes and
            setpoint arrays of the parameter to correspond with the sweep
        get(): executes a sweep and returns magnitude and phase arrays

        get_ramping: Queries the value of self.ramp_state and
            self.ramp_time. Returns a string.

    TODO(nataliejpg): ability to choose for abs or db in magnitude return
    """

    def __init__(self, name, instrument, start, stop, npts):
        super().__init__(name, names=("", ""), shapes=((), ()))
        self._instrument = instrument
        self.set_sweep(start, stop, npts)
        self.names = ('magnitude', 'phase')
        self.units = ('dBm', 'rad')
        self.setpoint_names = (('frequency',), ('frequency',))

    def set_sweep(self, start, stop, npts):
        #  needed to update config of the software parameter on sweep chage
        # freq setpoints tuple as needs to be hashable for look up
        f = tuple(np.linspace(int(start), int(stop), num=npts))
        self.setpoints = ((f,), (f,))
        self.shapes = ((npts,), (npts,))

    def get(self):
        self._instrument.write('SENS1:AVER:STAT ON')
        self._instrument.write('AVER:CLE')
        self._instrument.cont_meas_off()

        # instrument averages over its last 'avg' number of sweeps
        # need to ensure averaged result is returned
        for avgcount in range(self._instrument.avg()):
            self._instrument.write('INIT:IMM; *WAI')
        data_str = self._instrument.ask('CALC:DATA? SDAT').split(',')
        data_list = [float(v) for v in data_str]

        # data_list of complex numbers [re1,im1,re2,im2...]
        data_arr = np.array(data_list).reshape(int(len(data_list) / 2), 2)
        mag_array, phase_array = [], []
        for comp in data_arr:
            complex_num = complex(comp[0], comp[1])
            mag_array.append(abs(complex_num))
            phase_array.append(phase(complex_num))
        self._instrument.cont_meas_on()
        return mag_array, phase_array


class ZNB20(VisaInstrument):
    """
    This is the qcodes driver for the Rohde & Schwarz ZNB20 and ZNB8
    virtual network analysers

    Spectroscopy mode is available for the 4 port ZNB8 where the drive
    frequency from port 1 is fixed and instead frequency sweep on port 3
    is carried out. The same quentity is measured (ie S21)

    Requires FrequencySweep parameter for taking a trace

    """

    def __init__(self, name, address, port_num, **kwargs):
        self.ports = port_num

        super().__init__(name=name, address=address, **kwargs)

        self.add_parameter(name='power',
                           label='Power',
                           unit='dBm',
                           get_cmd='SOUR:POW?',
                           set_cmd='SOUR:POW {:.4f}',
                           get_parser=int,
                           vals=vals.Numbers(-150, 25))

        self.add_parameter(name='bandwidth',
                           label='Bandwidth',
                           unit='Hz',
                           get_cmd='SENS:BAND?',
                           set_cmd='SENS:BAND {:.4f}',
                           get_parser=int,
                           vals=vals.Numbers(1, 1e6))

        self.add_parameter(name='avg',
                           label='Averages',
                           unit='',
                           get_cmd='AVER:COUN?',
                           set_cmd='AVER:COUN {:.4f}',
                           get_parser=int,
                           vals=vals.Numbers(1, 5000))

        self.add_parameter(name='start',
                           get_cmd='SENS:FREQ:START?',
                           set_cmd=self._set_start,
                           get_parser=int)

        self.add_parameter(name='stop',
                           get_cmd='SENS:FREQ:STOP?',
                           set_cmd=self._set_stop,
                           get_parser=int)

        self.add_parameter(name='center',
                           get_cmd='SENS:FREQ:CENT?',
                           get_parser=int)

        self.add_parameter(name='npts',
                           get_cmd='SENS:SWE:POIN?',
                           set_cmd=self._set_npts,
                           get_parser=int)

        self.add_parameter(name='trace',
                           start=self.start(),
                           stop=self.stop(),
                           npts=self.npts(),
                           parameter_class=FrequencySweep)

        # Spectroscopy mode parameters

        # query spec mode via power state: PERManent for spec mode on
        self.add_parameter(name='spec_mode',
                           get_cmd='SOUR:POW1:PERM?',
                           set_cmd=self._set_spec_mode,
                           get_parser=int,
                           vals=vals.OnOff(),
                           val_mapping={'on': 1,
                                        'off': 0})

        self.add_parameter(name='fixed_freq',
                           get_cmd=self._get_fixed_freq,
                           set_cmd=self._set_fixed_freq)

        self.add_parameter(name='fixed_pow',
                           get_cmd=self._get_fixed_pow,
                           set_cmd=self._set_fixed_pow,
                           vals=vals.Numbers(-150, 25))

        self.add_function('reset', call_cmd='*RST')
        self.add_function('tooltip_on', call_cmd='SYST:ERR:DISP ON')
        self.add_function('tooltip_off', call_cmd='SYST:ERR:DISP OFF')
        self.add_function('cont_meas_on', call_cmd='INIT:CONT:ALL ON')
        self.add_function('cont_meas_off', call_cmd='INIT:CONT:ALL OFF')
        self.add_function('update_display_once', call_cmd='SYST:DISP:UPD ONCE')
        self.add_function('update_display_on', call_cmd='SYST:DISP:UPD ON')
        self.add_function('update_display_off', call_cmd='SYST:DISP:UPD OFF')
        self.add_function('rf_off', call_cmd='OUTP OFF')
        self.add_function('rf_on', call_cmd='OUTP ON')

        self.initialise()
        self.connect_message()

    def initialise(self):
        """
        Initialise instrument to sweep linearly, automatically calculate and
        sweep in minimum time, not require a trigger, do averaging if avg set.

        Spectroscopy mode initialised off
        """
        self.write('SENS1:SWE:TYPE LIN')
        self.write('SENS1:SWE:TIME:AUTO ON')
        self.write('TRIG1:SEQ:SOUR IMM')
        self.write('SENS1:AVER:STAT ON')
        self.update_display_on()
        self.trace.set_sweep(self.start(), self.stop(), self.npts())

    def _set_start(self, val):
        self.write('SENS:FREQ:START {:.4f}'.format(val))
        # update setpoints for FrequencySweep param
        self.trace.set_sweep(val, self.stop(), self.npts())

    def _set_stop(self, val):
        self.write('SENS:FREQ:STOP {:.4f}'.format(val))
        # update setpoints for FrequencySweep param
        self.trace.set_sweep(self.start(), val, self.npts())

    def _set_npts(self, val):
        self.write('SENS:SWE:POIN {:.4f}'.format(val))
        # update setpoints for FrequencySweep param
        self.trace.set_sweep(self.start(), self.stop(), val)

    # external generator settings

    def count_external_generators(self):
        num = self.ask('SYST:COMM:RDEV:GEN:COUN?').strip()
        return int(num)

    def get_external_generator_setup(self, num=1):
        setup = self.ask(
            'SYSTem:COMMunicate:RDEVice:GEN{:.0f}:DEF?'.format(num)).strip()
        return setup

    def set_external_generator(self, address, gen=1, gen_name="ext gen 1",
                               driver="SGS100A", interface="VXI-11"):
        self.write('SYST:COMM:RDEV:GEN{:.0f}:DEF "{}", "{}", "{}",  "{}"'.format(
            gen, gen_name, driver, interface, address))

    def clear_external_generator(self, num=1):
        self.write('SYST:COMM:RDEV:GEN{:.0f}:DEL'.format(num))

    def get_external_generator_numbers(self):
        cat = self.ask('SYST:COMM:RDEV:GEN1:CAT?').strip()
        return cat

    # Spectrocopy mode settings

    def _set_spec_mode(self, val):
        """
        Set the spectroscopy mode to off or on:
          on:
            4 port:
            - turns on port 3 and sets ports 1 and 3 to permanent drive
            - sets  port 1 to drive at the previous central frequency
              and port 2 to measure at the same frequency
            - sets port 1 power to be same as previous power (by way of offset)
            - measure b2/a1 with source port 1
            2 port:
            on:
            - turns on generator and sets ports 1 and generator to permanent
            - sets port 1 to drive at the previous central frequency
              and port 2 to measure at the same frequency
            - turns off port 2 drive
            - measure b2 with source generator
          off (both):
            - resets port 1 and 2 to be sweep ports
            - sets power offset of ports 1 and 2 back to 0
            - sets permanent drives off and turns off port 3 (/generator) drive
            - measure b2/a1 with source port 1

        args:
            val (0 or 1 for mode off or on)
        """

        if val == 0 and self.ports == 4:
            self.write('CALC1:PAR:MEAS "Trc1", "B2D1/A1D1SAM"')
            self.write('SOUR:FREQ1:CONV:ARB:IFR 1, 1, 0, SWE')
            self.write('SOUR:FREQ2:CONV:ARB:IFR 1, 1, 0, SWE')
            self.write('SOUR:POW1:OFFS 0, CPAD')
            self.write('SOUR:POW2:OFFS 0, CPAD')
            self.write('SOUR:POW1:PERM 0')
            self.write('SOUR:POW3:PERM 0')
            self.write('SOUR:POW3:STAT 0')
        elif val == 1 and self.ports == 4:
            self.write('CALC1:PAR:MEAS "Trc1", "B2D1/A1D1SAM"')
            freq = self.center()
            pow = self.power()
            self.write('SOUR:POW3:STAT 1')
            time.sleep(0.2)
            self.write('SOUR:POW1:PERM 1')
            time.sleep(0.2)
            self.write('SOUR:POW3:PERM 1')
            time.sleep(0.2)
            self.fixed_freq(freq)
            self.fixed_pow(pow)
        elif val == 0 and self.ports == 2:
            self.write('CALC1:PAR:MEAS "Trc1", "B2D1/A1D1SAM"')
            self.write('SOUR:FREQ1:CONV:ARB:IFR 1, 1, 0, SWE')
            self.write('SOUR:FREQ2:CONV:ARB:IFR 1, 1, 0, SWE')
            self.write('SOUR:POW1:OFFS 0, CPAD')
            self.write('SOUR:POW2:OFFS 0, CPAD')
            self.write('SOUR:POW1:PERM OFF')
            self.write('SOUR:POW:GEN1:PERM OFF')
            self.write('SOUR:POW:GEN1:STAT OFF')
            self.write('SOUR:POW2:STAT ON')
        elif val == 1 and self.ports == 2:
            self.write('CALC1:PAR:MEAS "Trc1", "B2G1SAM"')
            freq = self.center()
            pow = self.power()
            self.write('SOUR:POW2:STAT OFF')
            time.sleep(0.2)
            self.write('SOUR:POW:GEN1:STAT ON')
            time.sleep(0.2)
            self.write('SOUR:POW:GEN1:PERM ON')
            time.sleep(0.2)
            self.write('SOUR:POW1:PERM ON')
            time.sleep(0.2)
            self.fixed_freq(freq)
            self.fixed_pow(pow)
        else:
            logging.error('cannot set mode %s' % val)

    def _get_fixed_freq(self):
        if self.spec_mode() != 'on':
            logging.warning(
                'cannot get fixed frequency if spec mode is not on')
        if self.ports == 4:
            ret = self.ask('SOUR:FREQ1:CONV:ARB:IFR?').split(',')[2]
        elif self.ports == 2:
            ret = self.ask('SOUR:FREQ:CONV:ARB:EFR1?').split(',')[3]
        return int(ret)

    def _set_fixed_freq(self, freq):
        if self.spec_mode() != 'on':
            logging.warning(
                'cannot set fixed frequency if spec mode is not on')
        if self.ports == 4:
            self.write('SOUR:FREQ1:CONV:ARB:IFR 0, 1, {:.6f}, CW'.format(freq))
            self.write('SOUR:FREQ2:CONV:ARB:IFR 0, 1, {:.6f}, CW'.format(freq))
        elif self.ports == 2:
            self.write(
                'SOUR:FREQ:CONV:ARB:EFR1 ON, 0, 1, {:.6f}, CW'.format(freq))
            self.write('SOUR:FREQ2:CONV:ARB:IFR 0, 1, {:.6f}, CW'.format(freq))

    def _get_fixed_pow(self):
        if self.spec_mode() != 'on':
            logging.warning('cannot get fixed power if spec mode is not on')
        if self.ports == 4:
            ret = self.ask('SOUR:POW1:OFFS?').split(',')[0]
        if self.ports == 2:
            ret = self.ask('SOUR:POW:GEN1:OFFS?').split(',')[0]
        return int(ret)

    def _set_fixed_pow(self, pow):
        if self.spec_mode() != 'on':
            logging.error('cannot set fixed power if spec mode is not on')
        if self.ports == 4:
            self.write('SOUR:POW1:OFFS {:.3f}, ONLY'.format(pow))
            self.write('SOUR:POW2:OFFS {:.3f}, ONLY'.format(pow))
        if self.ports == 2:
            self.write('SOUR:POW:GEN1:OFFS {:.3f}, ONLY'.format(pow))
            self.write('SOUR:POW2:OFFS {:.3f}, ONLY'.format(pow))
