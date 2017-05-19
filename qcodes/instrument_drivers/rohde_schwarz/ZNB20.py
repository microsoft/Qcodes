from functools import partial

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
    def __init__(self, name, instrument, start, stop, npts, channel, sindex):
        super().__init__(name, names=("", ""), shapes=((), ()))
        self._instrument = instrument
        self.set_sweep(start, stop, npts)
        self._channel = channel
        self._sindex = sindex
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
        self._instrument.write('SENS{}:AVER:STAT ON'.format(self._channel))
        self._instrument.write('SENS{}:AVER:CLE'.format(self._channel))
        self._instrument.cont_meas_off()

        # instrument averages over its last 'avg' number of sweeps
        # need to ensure averaged result is returned
        for avgcount in range(getattr(self._instrument, 'avg{}{}'.format(*self._sindex))()):
            self._instrument.write('INIT:IMM; *WAI')
        data_str = self._instrument.ask('CALC{}:DATA? SDAT'.format(self._channel)).split(',')
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
    - centre/span settable for frequency sweep
    - check initialisation settings and test functions
    """
    def __init__(self, name, address, **kwargs):

        super().__init__(name=name, address=address, **kwargs)
        n = 1
        self._sindex_to_channel = {}
        self._channel_to_sindex = {}
        for i in range(1,3):
            self._sindex_to_channel[i] = {}
            for j in range(1,3):
                self.add_parameter(name='power{}{}'.format(i, j),
                                   label='Power{}{}'.format(i, j),
                                   unit='dBm',
                                   get_cmd='SOUR{}:POW?'.format(n),
                                   set_cmd='SOUR{}'.format(n)+':POW {:.4f}',
                                   get_parser=int,
                                   vals=vals.Numbers(-150, 25))
                self.add_parameter(name='bandwidth{}{}'.format(i, j),
                                   label='Bandwidth{}{}'.format(i, j),
                                   unit='Hz',
                                   get_cmd='SENS{}:BAND?'.format(n),
                                   set_cmd='SENS{}'.format(n)+':BAND {:.4f}',
                                   get_parser=int,
                                   vals=vals.Numbers(1, 1e6))
                self.add_parameter(name='avg{}{}'.format(i ,j),
                                   label='Averages{}{}'.format(i ,j),
                                   unit='',
                                   get_cmd='SENS{}:AVER:COUN?'.format(n),
                                   set_cmd='SENS{}'.format(n)+':AVER:COUN {:.4f}',
                                   get_parser=int,
                                   vals=vals.Numbers(1, 5000))
                self.add_parameter(name='start{}{}'.format(i ,j),
                                   get_cmd='SENS{}:FREQ:START?'.format(n),
                                   set_cmd=partial(self._set_start, channel=n),
                                   get_parser=float)
                self.add_parameter(name='stop{}{}'.format(i ,j),
                                   get_cmd='SENS{}:FREQ:STOP?'.format(n),
                                   set_cmd=partial(self._set_stop, channel=n),
                                   get_parser=float)
                self.add_parameter(name='center{}{}'.format(i ,j),
                                   get_cmd='SENS{}:FREQ:CENT?'.format(n),
                                   set_cmd=partial(self._set_center, channel=n),
                                   get_parser=float)
                self.add_parameter(name='span{}{}'.format(i ,j),
                                   get_cmd = 'SENS{}:FREQ:SPAN?'.format(n),
                                   set_cmd=partial(self._set_span, channel=n),
                                   get_parser=float)
                self.add_parameter(name='npts{}{}'.format(i ,j),
                                   get_cmd='SENS:SWE:POIN?',
                                   set_cmd=partial(self._set_npts, channel=n),
                                   get_parser=int)
                self.add_parameter(name='trace{}{}'.format(i, j),
                                   start=getattr(self, 'start{}{}'.format(i, j))(),
                                   stop=getattr(self, 'stop{}{}'.format(i, j))(),
                                   npts=getattr(self, 'npts{}{}'.format(i, j))(),
                                   channel=n,
                                   sindex=(i, j),
                                   parameter_class=FrequencySweep)
                self._sindex_to_channel[i][j] = n
                self._channel_to_sindex[n] = (i, j)
                n += 1

        self.add_function('reset', call_cmd='*RST')
        self.add_function('tooltip_on', call_cmd='SYST:ERR:DISP ON')
        self.add_function('tooltip_off', call_cmd='SYST:ERR:DISP OFF')
        self.add_function('cont_meas_on', call_cmd='INIT:CONT:ALL ON')
        self.add_function('cont_meas_off', call_cmd='INIT:CONT:ALL OFF')
        self.add_function('update_display_once', call_cmd='SYST:DISP:UPD ONCE')
        self.add_function('update_display_on', call_cmd='SYST:DISP:UPD ON')
        self.add_function('update_display_off', call_cmd='SYST:DISP:UPD OFF')
        self.add_function('display_sij_split', call_cmd='DISP:LAY GRID;:DISP:LAY:GRID 2,2')
        self.add_function('rf_off', call_cmd='OUTP1 OFF')
        self.add_function('rf_on', call_cmd='OUTP1 ON')

        self.initialise()
        self._setup_s_channels()
        self.autoscale_all()
        self.connect_message()


    def autoscale_all(self):
        for i in range(1,5):
            self.write('DISPlay:TRACe1:Y:SCALe:AUTO ONCE, "Trc{}"'.format(i))

    def _setup_s_channels(self):
        """
        Sets up 4 channels with a single trace in each.
        Each channel will contain one trace.
        """
        for i in range(1,3):
            for j in range(1,3):
                n = self._sindex_to_channel[i][j]
                self.write("CALC{}:PAR:SDEF 'Trc{}', 'S{}{}'".format(n, n, i, j))

    def _set_start(self, val, channel):
        self.write('SENS{}:FREQ:START {:.4f}'.format(channel, val))
        i, j = self._channel_to_sindex[channel]
        stop = getattr(self, 'stop{}{}'.format(i, j))()
        npts = getattr(self, 'npts{}{}'.format(i, j))()
        trace = getattr(self, 'trace{}{}'.format(i, j))
        # update setpoints for FrequencySweep param
        trace.set_sweep(val, stop, npts)

    def _set_stop(self, val, channel):
        self.write('SENS{}:FREQ:STOP {:.4f}'.format(channel, val))
        i, j = self._channel_to_sindex[channel]
        start = getattr(self, 'start{}{}'.format(i, j))()
        npts = getattr(self, 'npts{}{}'.format(i, j))()
        trace = getattr(self, 'trace{}{}'.format(i, j))
        # update setpoints for FrequencySweep param
        trace.set_sweep(start, val, npts)

    def _set_npts(self, val, channel):
        self.write('SENS{}:SWE:POIN {:.4f}'.format(channel, val))
        i, j = self._channel_to_sindex[channel]
        start = getattr(self, 'start{}{}'.format(i, j))()
        stop = getattr(self, 'stop{}{}'.format(i, j))()
        trace = getattr(self, 'trace{}{}'.format(i, j))
        # update setpoints for FrequencySweep param
        trace.set_sweep(start, stop, val)

    def _set_span(self, val, channel):
        self.write('SENS{}:FREQ:SPAN {:.4f}'.format(channel, val))
        i, j = self._channel_to_sindex[channel]
        start = getattr(self, 'start{}{}'.format(i, j))()
        stop = getattr(self, 'stop{}{}'.format(i, j))()
        npts = getattr(self, 'npts{}{}'.format(i, j))()
        trace = getattr(self, 'trace{}{}'.format(i, j))
        trace.set_sweep(start, stop, npts)

    def _set_center(self, val, channel):
        self.write('SENS{}:FREQ:CENT {:.4f}'.format(channel, val))
        i, j = self._channel_to_sindex[channel]
        start = getattr(self, 'start{}{}'.format(i, j))()
        stop = getattr(self, 'stop{}{}'.format(i, j))()
        npts = getattr(self, 'npts{}{}'.format(i, j))()
        trace = getattr(self, 'trace{}{}'.format(i, j))
        trace.set_sweep(start, stop, npts)

    def _set_default_values(self):
        for i in range(1,3):
            for j in range(1,3):
                getattr(self, 'start{}{}'.format(i,j))(1e6)
                getattr(self, 'stop{}{}'.format(i, j))(2e6)
                getattr(self, 'npts{}{}'.format(i, j))(10)
                getattr(self, 'power{}{}'.format(i, j))(-50)

    def initialise(self):
        self.write('*RST')
        self.write('SENS1:SWE:TYPE LIN')
        self.write('SENS1:SWE:TIME:AUTO ON')
        self.write('TRIG1:SEQ:SOUR IMM')
        self.write('SENS1:AVER:STAT ON')
        self.update_display_on()
        self._set_default_values()