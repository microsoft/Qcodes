import numpy as np

from qcodes import VisaInstrument, ArrayParameter
from qcodes.utils import validators as vals


class Spectrum(ArrayParameter):
    """
    Hardware controlled parameter class for Rohde Schwarz FSV trace.

    Args:
        name: parameter name
        instrument: instrument the parameter belongs to
        start: starting frequency of sweep
        stop: ending frequency of sweep
        npts: number of points in frequency sweep

    Methods:
          set_sweep(start, stop, npts): sets the shapes and
              setpoint arrays of the parameter to correspond with the sweep
          get(): executes a sweep and returns magnitude and phase arrays

    """
    def __init__(self, name, instrument, start, stop, npts):
        super().__init__(name, shape=(npts,),
                         instrument=instrument,
                         unit='dB',
                         label='magnitude',
                         setpoint_units=('Hz',),
                         setpoint_labels=('frequency',),
                         setpoint_names=('frequency',))
        self.set_sweep(start, stop, npts)

    def set_sweep(self, start, stop, npts):
        #  needed to update config of the software parameter on sweep change
        # freq setpoints tuple as needs to be hashable for look up
        f = tuple(np.linspace(int(start), int(stop), num=npts))
        self.setpoints = (f,)
        self.shape = (npts,)

    def get_raw(self):
        data = self._instrument._get_data()
        return data


class FSV_Exception(Exception):
    pass


class FSV(VisaInstrument):
    """
    qcodes driver for the Rohde & Schwarz FSV Signal Analyzer.

    Args:
        name: instrument name
        address: VISA ressource name of instrument in format
            'TCPIP0::192.168.15.100::inst0::INSTR'
        **kwargs: passed to base class
    """

    def __init__(self, name: str, address: str, **kwargs) -> None:

        super().__init__(name=name, address=address, **kwargs)

        self.add_parameter(name='start', unit='Hz',
                           get_cmd='FREQ:START?',
                           set_cmd=self._set_start,
                           get_parser=float,
                           vals=vals.Numbers(10, 13.6e9))

        self.add_parameter(name='stop', unit='Hz',
                           get_cmd='FREQ:STOP?',
                           set_cmd=self._set_stop,
                           get_parser=float,
                           vals=vals.Numbers(10, 13.6e9))

        self.add_parameter(name='center', unit='Hz',
                           get_cmd='FREQ:CENT?',
                           set_cmd=self._set_center,
                           get_parser=float,
                           vals=vals.Numbers(5, 13.6e9))

        self.add_parameter(name='span', unit='Hz',
                           get_cmd='FREQ:SPAN?',
                           set_cmd=self._set_span,
                           get_parser=float,
                           vals=vals.Numbers(10, 13.6e9))

        self.add_parameter(name='npts',
                           get_cmd='SWE:POIN?',
                           set_cmd=self._set_npts,
                           get_parser=int)

        self.add_parameter(name='trace',
                           start=self.start(),
                           stop=self.stop(),
                           npts=self.npts(),
                           parameter_class=Spectrum)

        self.add_parameter('referencelevel', unit='dBm',
                           get_cmd='DISP:TRAC:Y:RLEV?',
                           set_cmd='DISP:TRAC:Y:RLEV {:e}',
                           get_parser=float,
                           vals=vals.Numbers(-130, 0))

        self.add_parameter(name='bandwidth', unit='Hz',
                           get_cmd='BAND?',
                           set_cmd='BAND {}',
                           get_parser=float,
                           vals=vals.Numbers(1, 10e6))

        self.add_parameter('mode',
                           get_cmd='INST?',
                           set_cmd='INST {:s}',
                           get_parser=self._parse_str,
                           val_mapping={"Spectrum": "SAN",
                                        "IQ Analyzer": "IQ",
                                        "Phase Noise": "PNO"})

        self.add_parameter('continuous',
                           get_cmd='INIT:CONT?',
                           set_cmd='INIT:CONT {:s}',
                           get_parser=self._parse_on_off,
                           vals=vals.OnOff())

    @staticmethod
    def _parse_on_off(stat):
        if stat.startswith('0'):
            stat = 'off'
        elif stat.startswith('1'):
            stat = 'on'
        return stat

    @staticmethod
    def _parse_str(string):
        return string.strip().upper()

    def _set_start(self, val):
        self.write('FREQ:STAR {:.9f}'.format(val))
        start = self.start()
        if val != start:
            log.warning(
                "Could not set start to {}.\
                 setting it to {}".format(val, start))
        self._update_traces()

    def _set_stop(self, val):
        self.write('FREQ:STOP {:.9f}'.format(val))
        stop = self.stop()
        if val != stop:
            log.warning(
                "Could not set stop to {}.\
                 setting it to {}".format(val, stop))
        self._update_traces()

    def _set_center(self, val):
        self.write('FREQ:CENT {:.9f}'.format(val))
        center = self.center()
        if val != center:
            log.warning(
                "Could not set center to {}.\
                 setting it to {}".format(val, center))
        self._update_traces()

    def _set_span(self, val):
        self.write('FREQ:SPAN {:.9f}'.format(val))
        span = self.span()
        if val != span:
            log.warning(
                "Could not set span to {}.\
                 setting it to {}".format(val, span))
        self._update_traces()

    def _set_npts(self, val):
        self.write('SWE:POIN {:d}'.format(val))
        npts = self.npts()
        if val != npts:
            log.warning(
                "Could not set npts to {}.\
                 setting it to {}".format(val, npts))
        self._update_traces()

    def _update_traces(self):
        """ updates start, stop and npts of all trace parameters"""
        start = self.start()
        stop = self.stop()
        npts = self.npts()
        span = self.span()
        center = self.center()
        for _, parameter in self.parameters.items():
            if isinstance(parameter, (ArrayParameter)):
                try:
                    parameter.set_sweep(start, stop, npts)
                except AttributeError:
                    pass

    def _get_data(self, tno=1):
        yvals = np.fromstring(self.ask('TRAC? TRACE{}'.format(tno)),
                              dtype=float, sep=',')
        return yvals

    def markers_to_peaks(self, no_of_peaks=3):
        '''
        Moves n markers to n highest peaks

        WARNING: Turns off all other markers
        '''
        for i in range(8):
            self.write('CALC:MARK%d OFF' % (i+1))
        for i in range(no_of_peaks):
            self.write('CALC:MARK%d ON' % (i+1))

    def marker_to_max(self):
        '''
        Moves marker 1 to the highest peak

        WARNING: Turns off all other markers
        '''
        self.markers_to_peaks(1)

    def marker_next(self, marker=1):
        '''
        Moves Marker number <marker>
        to the next highest peak
        '''
        if not int(self.ask('CALC:MARK%d?' % (marker)).strip()):
                raise FSV_Exception('Marker %d is not on' % (marker))
        self.write('CALC:MARK%d:MAX:NEXT' % marker)

    def get_max(self, no_of_peaks=3):
        '''
        Returns frequencies and powers of the
        first <no_of_peaks> maximas in the trace.
        '''
        xvals = []
        yvals = []
        for i in range(no_of_peaks):
            if not int(self.ask('CALC:MARK%d?' % (i+1)).strip()):
                raise FSV_Exception('Marker %d is not on' % (i+1))
            xvals.append(float(self.ask('CALC:MARK%d:X?' % (i+1)).strip()))
            yvals.append(float(self.ask('CALC:MARK%d:Y?' % (i+1)).strip()))
        return xvals, yvals
