import time
import logging
import zhinst.utils

import numpy as np
import zhinst.ziPython as zipy

from functools import partial
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals

log = logging.getLogger(__name__)


class Sweep(Parameter):
    """
    Parameter class for the ZIUHFLI instrument class for the sweeper.

    The get method returns a tuple of arrays, where each array contains the
    values of a signal added to the sweep (e.g. demodulator 4 phase).
    """
    pass


class ZIUHFLI(Instrument):
    """
    QCoDeS driver for ZI UHF LI
    """

    def __init__(self, name, device_ID, api_level=5):

        super().__init__(name)

        (self.daq, self.device, self.props) = zhinst.utils.create_api_session(device_ID,
                                                                              api_level)

        # @William:
        # the sweeper has to be declared here, otherwise I do not know how to bind to the
        # right getters and setters

        self.sweeper = self.daq.sweep()
        self.sweeper.set('sweep/device', self.device)

        ########################################
        # INSTRUMENT PARAMETERS

        # Oscillators
        self.add_parameter('oscillator1_freq',
                           label='Frequency of oscillator 1',
                           unit='Hz',
                           set_cmd=partial(self.daq.setDouble,
                                           '/' + device_ID + '/oscs/0/freq'),
                           get_cmd=partial(self.daq.getDouble,
                                           '/' + device_ID + '/oscs/0/freq'),
                           vals=vals.Numbers(0, 600e6))

        self.add_parameter('oscillator2_freq',
                           label='Frequency of oscillator 2',
                           unit='Hz',
                           set_cmd=partial(self.daq.setDouble,
                                           '/' + device_ID + '/oscs/1/freq'),
                           get_cmd=partial(self.daq.getDouble,
                                           '/' + device_ID + '/oscs/1/freq'),
                           vals=vals.Numbers(0, 600e6))

        ########################################
        # DEMODULATOR PARAMETERS

        for demod in range(1, 9):
            self.add_parameter('demod{}_order'.format(demod),
                               label='Filter order',
                               get_cmd=partial(self._demod_getter,
                                               demod-1, 'order'),
                               set_cmd=partial(self._demod_setter,
                                               0, demod-1, 'order'),
                               vals=vals.Ints(1, 8)
                               )

            self.add_parameter('demod{}_harmonic'.format(demod),
                               label=('Reference frequency multiplication' +
                                      ' factor'),
                               get_cmd=partial(self._demod_getter,
                                               demod-1, 'harmonic'),
                               set_cmd=partial(self._demod_setter,
                                               1, demod-1, 'harmonic'),
                               vals=vals.Ints(1, 999)
                               )

            self.add_parameter('demod{}_timeconstant'.format(demod),
                               label='Filter time constant',
                               get_cmd=partial(self._demod_getter,
                                               demod-1, 'timeconstant'),
                               set_cmd=partial(self._demod_setter,
                                               1, demod-1, 'timeconstant'),
                               unit='s'
                               )

            self.add_parameter('demod{}_samplerate'.format(demod),
                               label='Sample rate',
                               get_cmd=partial(self._demod_getter,
                                               demod-1, 'rate'),
                               set_cmd=partial(self._demod_setter,
                                               1, demod-1, 'rate'),
                               unit='Sa/s',
                               docstring="""
                                         Note: the value inserted by the user
                                         may be approximated to the
                                         nearest value supported by the
                                         instrument.
                                         """)


        ########################################
        # SWEEPER PARAMETERS

        self.add_parameter('sweeper_BWmode',
                           label='Sweeper bandwidth control mode',
                           set_cmd=partial(self.sweeper.set,
                                           'sweep/bandwidthcontrol'),
                           get_cmd=partial(self._sweep_getter,
                                           'sweep/bandwidthcontrol'),
                           val_mapping={'auto': 2, 'fixed': 1, 'current': 0},
                           docstring="""
                                     For each sweep point, the demodulator
                                     filter bandwidth (time constant) may
                                     be either set automatically, be the
                                     current demodulator bandwidth or be
                                     a fixed number; the sweeper_BW
                                     parameter.
                                     """
                           )

        self.add_parameter('sweeper_BW',
                           label='Fixed bandwidth sweeper bandwidth (NEP)',
                           set_cmd=partial(self.sweeper.set,
                                           'sweep/bandwidth'),
                           get_cmd=partial(self._sweep_getter,
                                           'sweep/bandwidth'),
                           docstring="""
                                     This is the NEP bandwidth used by the
                                     sweeper if sweeper_BWmode is set to
                                     'fixed'. If sweeper_BWmode is either
                                     'auto' or 'current', this value is
                                     ignored.
                                     """
                           )

        self.add_parameter('sweeper_start',
                            label='Start value of the sweep',
                            set_cmd=partial(self.sweeper.set,
                                            'sweep/start'),
                            get_cmd=partial(self._sweep_getter,
                                            'sweep/start'),
                            vals=vals.Numbers(0, 600e6))

        self.add_parameter('sweeper_stop',
                            label='Stop value of the sweep',
                            set_cmd=partial(self.sweeper.set,
                                            'sweep/stop'),
                            get_cmd=partial(self._sweep_getter,
                                            'sweep/stop'),
                            vals=vals.Numbers(0, 600e6))

        self.add_parameter('sweeper_samplecount',
                            label='Length of the sweep (pts)',
                            set_cmd=partial(self.sweeper.set,
                                            'sweep/samplecount'),
                            get_cmd=partial(self._sweep_getter,
                                            'sweep/samplecount'),
                            vals=vals.Ints(0, 100000))

        # helper dict used by sweeper_param parameter
        sweepparams = {'Aux Out 1 Offset': 'auxouts/0/offset',
                       'Aux Out 2 Offset': 'auxouts/1/offset',
                       'Aux Out 3 Offset': 'auxouts/2/offset',
                       'Aux Out 4 Offset': 'auxouts/3/offset',
                       'Demod 1 Phase Shift': 'demods/0/phaseshift',
                       'Demod 2 Phase Shift': 'demods/1/phaseshift',
                       'Demod 3 Phase Shift': 'demods/2/phaseshift',
                       'Demod 4 Phase Shift': 'demods/3/phaseshift',
                       'Demod 5 Phase Shift': 'demods/4/phaseshift',
                       'Demod 6 Phase Shift': 'demods/5/phaseshift',
                       'Demod 7 Phase Shift': 'demods/6/phaseshift',
                       'Demod 8 Phase Shift': 'demods/7/phaseshift',
                       'Osc 1 Frequency': 'oscs/0/freq',
                       'Osc 2 Frequency': 'oscs/1/freq',
                       'Output 1 Amplitude 4': 'sigouts/0/amplitudes/3',
                       'Output 1 Offset': 'sigouts/0/offset',
                       'Output 2 Amplitude 8': 'sigouts/1/amplitudes/7',
                       'Output 2 Offset': 'sigouts/1/offset'
                       }

        self.add_parameter('sweeper_param',
                           label='Parameter to sweep (sweep x-axis)',
                           set_cmd=partial(self.sweeper.set,
                                           'sweep/gridnode'),
                           val_mapping=sweepparams,
                           get_cmd=partial(self._sweep_getter,
                                           'sweep/gridnode'),
                           vals=vals.Enum(*list(sweepparams.keys()))
                           )

        sweepunits = {'Aux Out 1 Offset': 'V',
                      'Aux Out 2 Offset': 'V',
                      'Aux Out 3 Offset': 'V',
                      'Aux Out 4 Offset': 'V',
                      'Demod 1 Phase Shift': 'degrees',
                      'Demod 2 Phase Shift': 'degrees',
                      'Demod 3 Phase Shift': 'degrees',
                      'Demod 4 Phase Shift': 'degrees',
                      'Demod 5 Phase Shift': 'degrees',
                      'Demod 6 Phase Shift': 'degrees',
                      'Demod 7 Phase Shift': 'degrees',
                      'Demod 8 Phase Shift': 'degrees',
                      'Osc 1 Frequency': 'Hz',
                      'Osc 2 Frequency': 'Hz',
                      'Output 1 Amplitude 4': 'V',
                      'Output 1 Offset': 'V',
                      'Output 2 Amplitude 8': 'V',
                      'Output 2 Offset': 'V'
                      }

        self.add_parameter('sweeper_units',
                           label='Units of sweep x-axis',
                           get_cmd=self.sweeper_param.get,
                           get_parser=lambda x:sweepunits[x])

        # helper dict used by sweeper_mode parameter
        sweepmodes = {'Sequential': 0,
                      'Binary': 1,
                      'Biderectional': 2,
                      'Reverse': 3}

        self.add_parameter('sweeper_mode',
                            label='Sweep mode',
                            set_cmd=partial(self.sweeper.set,
                                            'sweep/scan'),
                            get_cmd=partial(self._sweep_getter, 'sweep/scan'),
                            val_mapping=sweepmodes,
                            vals=vals.Enum(*list(sweepmodes))
                            )

        self.add_parameter('sweeper_order',
                           label='Sweeper filter order',
                           set_cmd=partial(self.sweeper.set,
                                           'sweep/order'),
                           get_cmd=partial(self._sweep_getter,
                                           'sweep/order'),
                           vals=vals.Ints(1, 8),
                           docstring="""
                                     This value is invoked only when the
                                     sweeper_BWmode is set to 'fixed'.
                                     """)

        self.add_parameter('sweeper_settlingtime',
                           label=('Minimal settling time for the ' +
                                  'sweeper'),
                           set_cmd=partial(self.sweeper.set,
                                           'sweep/settling/time'),
                           get_cmd=partial(self._sweep_getter,
                                           'sweep/settling/time'),
                           vals=vals.Numbers(0),
                           unit='s',
                           docstring="""
                                     This is the minimal waiting time
                                     at each point during a sweep before the
                                     data acquisition starts. Note that the
                                     filter settings may result in a longer
                                     actual waiting/settling time.
                                     """
                           )

        self.add_parameter('sweeper_settlingtc',
                           label='Sweep filter settling time',
                           get_cmd=partial(self._sweep_getter,
                                           'sweep/settling/tc'),
                           unit='dim. less.',
                           docstring="""This settling time is in units of
                                        the filter time constant."""
                           )

        self.add_parameter('sweeper_averaging_samples',
                           label=('Minimal no. of samples to average at ' +
                                  'each sweep point'),
                           set_cmd=partial(self.sweeper.set,
                                           'sweep/averaging/sample'),
                           get_cmd=partial(self._sweep_getter,
                                           'sweep/averaging/sample'),
                           vals=vals.Ints(1),
                           docstring="""The actual number of samples is the
                                        maximum of this value and the 
                                        sweeper_averaging_time times the
                                        relevant sample rate."""
                           )

        self.add_parameter('sweeper_averaging_time',
                           label=('Minimal averaging time'),
                           set_cmd=partial(self.sweeper.set,
                                           'sweep/averaging/tc'),
                           get_cmd=partial(self._sweep_getter,
                                           'sweep/averaging/tc'),
                           unit='s',
                           docstring="""The actual number of samples is the
                                        maximum of this value times the
                                        relevant sample rate and the 
                                        sweeper_averaging_samples."""
                           )

        self.add_parameter('sweeper_sweeptime',
                           label='Expected sweep time',
                           unit='s',
                           get_cmd=self._get_sweep_time)

        self.add_parameter('sweeper_timeout',
                           label='Sweep timeout',
                           unit='s',
                           initial_value=10,
                           parameter_class=ManualParameter)

        # A "manual" parameter: a list of the signals for the sweeper
        # to subscribe to
        self._sweeper_signals = []

        # Set up the sweeper with a minimal number of required settings
        # (since some of the factory defaults are nonsensical)
        self.sweeper_BW.set(100)  # anything >0 will do
        self.sweeper_order.set(1)

    def _demod_setter(self, mode, demod, setting, value):
        """
        General set_cmd for demodulator parameters

        This function counts demodulators in a zero-indexed way.

        Args:
            mode (int): 0 means 'call setInt', 1 means 'call setDouble'
            demod (int): The demodulator in question (0-8)
            setting (str): The attribute to set, e.g. 'order'
            value (Union[int, float]): The value to set the attribute to
        """
        setstr = '/{}/demods/{}/{}'.format(self.device, demod, setting)

        if mode == 0:
            self.daq.setInt(setstr, value)
        if mode == 1:
            self.daq.setDouble(setstr, value)


    def _demod_getter(self, demod, setting):
        """
        General get_cmd for demodulator parameters

        The built-in self.daq.get commands returns a dictionary, but we
        want a single value

        This function counts demodulators in a zero-indexed way.
        """
        querystr = '/{}/demods/{}/{}'.format(self.device, demod, setting)
        returndict = self.daq.get(querystr)
        demod = str(demod)
        rawvalue = returndict[self.device]['demods'][demod][setting]['value']

        if isinstance(rawvalue, np.ndarray) and len(rawvalue) == 1:
            value = rawvalue[0]
        elif isinstance(rawvalue, list) and len(rawvalue) == 1:
            value = rawvalue[0]
        else:
            value = rawvalue

        return value

    @staticmethod
    def NEPBW_to_timeconstant(NEPBW, order):
        """
        Helper function to translate a NEP BW and a filter order
        to a filter time constant. Meant to be used when calculating
        sweeper sweep times.

        Note: precise only to within a few percent.

        Args:
            NEPBW (float): The NEP bandwidth in Hz
            order (int): The filter order

        Returns:
            float: The filter time constant in s.
        """
        const = {1: 0.249, 2: 0.124, 3: 0.093, 4: 0.078, 5: 0.068,
                 6: 0.061, 7: 0.056, 8: 0.052}
        tau_c = const[order]/NEPBW

        return tau_c

    def _get_sweep_time(self):
        """
        get_cmd for the sweeper_sweeptime parameter. 

        Note: this calculation is only an estimate and not precise to more
        than a few percent.

        Returns:
            Union[float, None]: None if the bandwidthcontrol setting is 
              'auto' (then all bets are off), otherwise a time in seconds.

        Raises:
            ValueError: if no signals are added to the sweep
        """

        # Possible TO-DO: cut down on the number of instrument
        # queries.

        if self._sweeper_signals == []:
            raise ValueError('No signals selected! Can not find sweep time.')

        mode = self.sweeper_BWmode.get()

        # The effective time constant of the demodulator depends on the 
        # sweeper/bandwidthcontrol setting.
        #
        # If this setting is 'current', the largest current
        # time constant of the involved demodulators is used
        #
        # If the setting is 'fixed', the NEP BW specified under 
        # sweep/bandwidth is used. The filter order is needed to convert
        # the NEP BW to a time constant

        demods = set([sig.split('/')[3] for sig in self._sweeper_signals])
        rates = []
        for demod in demods:
            rates.append(self._demod_getter(demod, 'rate'))
        rate = min(rates)

        if mode == 'current':
            tcs = []
            for demod in demods:
                tcs.append(self._demod_getter(demod, 'timeconstant'))

            tau_c = max(tcs)

        elif mode == 'fixed':
            order = self.sweeper_order()
            BW = self.sweeper_BW()

            tau_c = self.NEPBW_to_timeconstant(BW, order)

        elif mode == 'auto':
            return None
        
        settlingtime = max(self.sweeper_settlingtc.get()*tau_c,
                           self.sweeper_settlingtime.get())
        averagingtime = max(self.sweeper_averaging_time.get()*tau_c*rate,
                            self.sweeper_averaging_samples.get())/rate

        time_est = (settlingtime+averagingtime)*self.sweeper_samplecount.get()
        return time_est

    def _sweep_getter(self, setting):
        """
        General get_cmd for sweeper parameters

        The built-in sweeper.get command returns a dictionary, but we want
        single values.

        Args:
            setting (str): the path used by ZI to describe the setting, 
            e.g. 'sweep/settling/time'
        """
        returndict = self.sweeper.get(setting)  # this is a dict

        # The dict may have different 'depths' depending on the parameter.
        # The depth is encoded in the setting string
        keys = setting.split('/')[1:]

        while keys != []:
            key = keys.pop(0)
            returndict = returndict[key]
        rawvalue = returndict   

        if isinstance(rawvalue, np.ndarray) and len(rawvalue) == 1:
            value = rawvalue[0]
        elif isinstance(rawvalue, list) and len(rawvalue) == 1:
            value = rawvalue[0]
        else:
            value = rawvalue

        return value

    def add_signal_to_sweeper(self, demodulator, attribute):
        """
        Add a signal to the output of the sweeper. When the sweeper sweeps,
        the signals added to the sweeper are returned.

        Args:
            demodulator (int): A number from 1-8 choosing the demodulator.
              The same demodulator can be chosen several times for 
              different attributes, e.g. demod1 X, demod1 phase
            attribute (str): The attribute to record, e.g. phase or Y

        Raises:
            ValueError: if a demodulator outside the allowed range is 
              selected
            ValueError: if an attribute not in the list of allowed attributes
              is selected
        """

        # TO-DO: implement all returned attributes
        valid_attributes = ['X', 'Y', 'R', 'phase']

        # Validation
        if demodulator not in range(1, 9):
            raise ValueError('Can not select demodulator' +
                             ' {}. Only '.format(demodulator) +
                             'demodulators 1-8 are available.')
        if attribute not in valid_attributes:
            raise ValueError('Can not select attribute:'+
                             '{}. Only the following attributes are' +
                             ' available: ' + 
                             ('{}, '*len(attributes)).format(*attributes))

        # internally, we use strings very similar to the ones used by the 
        # instrument, but with the attribute added, e.g.
        # '/dev2189/demods/0/sample/X' means X of demodulator 1.
        signalstring = ('/' + self.device +
                        '/demods/{}/sample/{}'.format(demodulator-1,
                                                      attribute))
        if signalstring not in self._sweeper_signals:
            self._sweeper_signals.append(signalstring)

    def remove_signal_from_sweeper(self, demodulator, attribute):
        """
        Remove a signal from the output of the sweeper. If the signal
        has not previously been added, a warning is logged.

        Args:
            demodulator (int): A number from 1-8 choosing the demodulator.
              The same demodulator can be chosen several times for 
              different attributes, e.g. demod1 X, demod1 phase
            attribute (str): The attribute to record, e.g. phase or Y
        """ 

        signalstring = ('/' + self.device +
                        '/demods/{}/sample/{}'.format(demodulator-1,
                                                      attribute))
        if signalstring not in self._sweeper_signals:
            log.warning('Can not remove signal with {} of'.format(attribute) +
                        'demodulator {}, since it was'.format(demodulator) +
                        ' not previously added.')
        else:
            self._sweeper_signals.remove(signalstring)

    def print_sweeper_settings(self):
        """
        Print the current settings of the sweeper. If execute sweeper
        is called, the sweep described here will be performed.
        """ 
        print('ACQUISITION')
        toprint = ['sweeper_BWmode', 'sweeper_BW', 'sweeper_order',
                   'sweeper_averaging_samples', 'sweeper_averaging_time']
        for paramname in toprint:
            parameter = self.parameters[paramname]
            print('    {}: {} ({})'.format(parameter.label, parameter.get(),
                                           parameter.unit))

        print('HORISONTAL')
        toprint = ['sweeper_start', 'sweeper_stop', 
                   'sweeper_units',
                   'sweeper_samplecount',
                   'sweeper_param', 'sweeper_mode',
                   'sweeper_timeout']
        for paramname in toprint:
            parameter = self.parameters[paramname]
            print('    {}: {}'.format(parameter.label, parameter.get()))
        
        print('VERTICAL')
        count = 1
        for signal in self._sweeper_signals:
            (_, _, _, dm, _, attr) = signal.split('/')
            print('    Signal {}: Demodulator {}: {}'.format(count,
                                                             int(dm)+1,
                                                             attr))
            count += 1

        features = ['timeconstant', 'order', 'samplerate']
        print('DEMODULATORS')
        demods = []
        for signal in self._sweeper_signals:
            demods.append(int(signal.split('/')[3]))
        demods = set(demods)
        for dm in demods:
            for feat in features:
                parameter = self.parameters['demod{:d}_{}'.format(dm+1, feat)]
                fmt = (dm+1, parameter.label, parameter.get(), parameter.unit)
                print('    Demodulator {}: {}: {:.6f} ({})'.format(*fmt))
        print('META')
        swptime = self.sweeper_sweeptime()
        if swptime is not None:
            print('    Expected sweep time: {:.1f} (s)'.format(swptime))
        else:
            print('    Expected sweep time: N/A in auto mode')

    def execute_sweeper(self):
        """
        Execute the sweeper aka wrap the ziPython function

        Reads the internal signal list and makes the sweep.

        """

        # self.sweeper.set('sweep/samplecount', self.sweeper.get('sweep/samplecount')['samplecount'][0] )
        # self.sweeper.set('sweep/samplecount', self.sweeper_samplecount)
        #path = '/%s/demods/%d/sample' % (self.device, 0)

        if self._sweeper_signals == []:
            raise ValueError('No signals selected! Can not perform sweep.')

        # we must enable the demodulators we use
        # after the sweep, they should be returned to their original state
        streamsettings = []
        for sigstr in self._sweeper_signals:
            path = '/'.join(sigstr.split('/')[:-1])
            (_, dev, _, dmnum, _) = path.split('/')

            # if the setting has never changed, it doesn't exist
            # then we assume that it's zero (factory default)
            try:
                toget = path.replace('sample', 'enable')
                # ZI like nesting...
                setting = self.daq.get(toget)[dev]['demods'][dmnum]['enable']['value'][0]
            except KeyError:
                setting = 0
            streamsettings.append(setting)
            self.daq.setInt(path.replace('sample', 'enable'), 1)
        
            self.sweeper.subscribe(path)  # we'll get multiple subscriptions to the same demod... problem?

        self.sweeper.execute()

        start = time.time()
        timeout = self.sweeper_timeout.get()

        # @William:
        # The while loop below is taken from the example, I left the print statements for now..not sure 
        # we need them though. 
        while not self.sweeper.finished():  # Wait until the sweep is complete, with timeout.
            time.sleep(0.2)  # Where does this value come from?
            progress = self.sweeper.progress()
            #print("Individual sweep progress: {:.2%}.".format(progress[0]), end="\r")
            # Here we could read intermediate data via:
            # data = sweeper.read(True)...
            # and process it while the sweep is completing.
            # if device in data:
            # ...
            if (time.time() - start) > timeout:
                # If for some reason the sweep is blocking, force the end of the
                # measurement.
                print("\nSweep still not finished, forcing finish...")
                # should exit function with error message instead of returning data
                self.sweeper.finish()
        print("")
        stop = time.time()

        print('Sweep took roughly {} seconds'.format(stop-start))
        return_flat_dict = True
        data = self.sweeper.read(return_flat_dict)

        self.sweeper.unsubscribe('*')
        for (state, sigstr) in zip(streamsettings, self._sweeper_signals):
            path = '/'.join(sigstr.split('/')[:-1])
            self.daq.setInt(path.replace('sample', 'enable'), int(state))

        return self._parsesweepdata(data)

    def _parsesweepdata(self, sweepresult):
        """
        Parse the raw result of a sweep into just the data asked for by the 
        added sweeper signals
        """
        trans = {'X': 'x', 'Y': 'y', 'Aux Input 1': 'auxin0',
                 'Aux Input 2': 'auxin1', 'R': 'r', 'phase': 'phase'}
        returndata = []
        for signal in self._sweeper_signals:
            path = '/'.join(signal.split('/')[:-1])
            attr = signal.split('/')[-1]
            data = sweepresult[path][0][0][trans[attr]]
            returndata.append(data)

        return tuple(returndata)

    def close(self):
        """
        Override of the base class' close function
        """
        self.daq.disconnect()
        super().close()

