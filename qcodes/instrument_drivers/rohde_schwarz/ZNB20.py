from qcodes import VisaInstrument
from qcodes.utils import validators as vals
from cmath import phase
import numpy as np
from qcodes import MultiParameter, Parameter


class FrequencySweep(MultiParameter):
    """
    Hardware controlled parameter class for Rohde Schwarz RSZNB20 trace.

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

    TODO:
      - ability to choose for abs or db in magnitude return
    """
    def __init__(self, name, instrument, start, stop, npts):
        super().__init__(name, names=("", ""), shapes=((), ()))
        self._instrument = instrument
        self.set_sweep(start, stop, npts)
        self.names = ('magnitude', 'phase')
        self.units = ('dBm', 'rad')
        self.setpoint_units = (('Hz',), ('Hz',))
        self.setpoint_names = (('frequency',), ('frequency',))

    def set_sweep(self, start, stop, npts):
        #  needed to update config of the software parameter on sweep change
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
    qcodes driver for the Rohde & Schwarz ZNB20 virtual network analyser

    Requires FrequencySweep parameter for taking a trace

    TODO:
    - centre/span settable for frequwncy sweep
    - check initialisation settings and test functions
    """
    def __init__(self, name, address, **kwargs):

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
                           get_parser=float)

        self.add_parameter(name='stop',
                           get_cmd='SENS:FREQ:STOP?',
                           set_cmd=self._set_stop,
                           get_parser=float)

        self.add_parameter(name='center',
                           get_cmd = 'SENS:FREQ:CENT?',
                           set_cmd = self._set_center,
                           get_parser=float)

        self.add_parameter(name='span',
                           get_cmd = 'SENS:FREQ:SPAN?',
                           set_cmd=self._set_span,
                           get_parser=float)

        self.add_parameter(name='npts',
                           get_cmd='SENS:SWE:POIN?',
                           set_cmd=self._set_npts,
                           get_parser=int)

        self.add_parameter(name='trace',
                           start=self.start(),
                           stop=self.stop(),
                           npts=self.npts(),
                           parameter_class=FrequencySweep)

        self.add_function('reset', call_cmd='*RST')
        self.add_function('tooltip_on', call_cmd='SYST:ERR:DISP ON')
        self.add_function('tooltip_off', call_cmd='SYST:ERR:DISP OFF')
        self.add_function('cont_meas_on', call_cmd='INIT:CONT:ALL ON')
        self.add_function('cont_meas_off', call_cmd='INIT:CONT:ALL OFF')
        self.add_function('update_display_once', call_cmd='SYST:DISP:UPD ONCE')
        self.add_function('update_display_on', call_cmd='SYST:DISP:UPD ON')
        self.add_function('update_display_off', call_cmd='SYST:DISP:UPD OFF')
        self.add_function('rf_off', call_cmd='OUTP1 OFF')
        self.add_function('rf_on', call_cmd='OUTP1 ON')

        self.initialise()
        self.connect_message()

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

    def _set_span(self, val):
        self.write('SENS:FREQ:SPAN {:.4f}'.format(val))
        self.trace.set_sweep(self.start(), self.stop(), self.npts())

    def _set_center(self, val):
        self.write('SENS:FREQ:CENT {:.4f}'.format(val))
        self.trace.set_sweep(self.start(), self.stop(), self.npts())

    def initialise(self):
        self.write('*RST')
        self.write('SENS1:SWE:TYPE LIN')
        self.write('SENS1:SWE:TIME:AUTO ON')
        self.write('TRIG1:SEQ:SOUR IMM')
        self.write('SENS1:AVER:STAT ON')
        self.update_display_on()
        self.start(1e6)
        self.stop(2e6)
        self.npts(10)
        self.power(-50)
