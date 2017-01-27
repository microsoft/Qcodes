import time
import logging
import numpy as np
from functools import partial

try:
    import zhinst.utils
except ImportError:
    raise ImportError('''Could not find Zurich Instruments Lab One software.
                         Please refer to the Zi UHF-LI User Manual for
                         download and installation instructions.
                      ''')

from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument.parameter import MultiParameter
from qcodes.instrument.base import Instrument
from qcodes.utils import validators as vals

log = logging.getLogger(__name__)


class Sweep(MultiParameter):
    """
    Parameter class for the ZIUHFLI instrument class for the sweeper.

    The get method returns a tuple of arrays, where each array contains the
    values of a signal added to the sweep (e.g. demodulator 4 phase).

    Attributes:
        names (tuple): Tuple of strings containing the names of the sweep
          signals (to be measured)
        units (tuple): Tuple of strings containg the units of the signals
        shapes (tuple): Tuple of tuples each containing the Length of a
          signal.
        setpoints (tuple): Tuple of N copies of the sweep x-axis points,
          where N is he number of measured signals
        setpoint_names (tuple): Tuple of N identical strings with the name
          of the sweep x-axis.

    """
    def __init__(self, name, instrument, **kwargs):
        # The __init__ requires that we supply names and shapes,
        # but there is no way to know what they could be known at this time.
        # They are updated via build_sweep.
        super().__init__(name, names=('',), shapes=((1,),), **kwargs)
        self._instrument = instrument

    def build_sweep(self):
        """
        Build a sweep with the current sweep settings. Must be called
        before the sweep can be executed.

        For developers:
        This is a general function for updating the sweeper.
        Every time a parameter of the sweeper is changed, this function
        must be called to update the sweeper. Although such behaviour is only
        strictly necessary for parameters that affect the setpoints of the
        Sweep parameter, having to call this function for any parameter is
        deemed more user friendly (easier to remember; when? -always).

        The function sets all (user specified) settings on the sweeper and
        additionally sets names, units, and setpoints for the Sweep
        parameter.

        """
        signals = self._instrument._sweeper_signals
        sweepdict = self._instrument._sweepdict

        log.info('Built a sweep')

        sigunits = {'X': 'V', 'Y': 'V', 'R': 'Vrms', 'Xrms': 'Vrms',
                    'Yrms': 'Vrms', 'Rrms': 'Vrms', 'phase': 'degrees'}
        names = []
        units = []
        for sig in signals:
            name = sig.split('/')[-1]
            names.append(name)
            units.append(sigunits[name])
        self.names = tuple(names)
        self.units = tuple(units)

        # TODO: what are good set point names?
        spnamedict = {'auxouts/0/offset': 'Volts',
                      'auxouts/1/offset': 'Volts',
                      'auxouts/2/offset': 'Volts',
                      'auxouts/3/offset': 'Volts',
                      'demods/0/phaseshift': 'degrees',
                      'demods/1/phaseshift': 'degrees',
                      'demods/2/phaseshift': 'degrees',
                      'demods/3/phaseshift': 'degrees',
                      'demods/4/phaseshift': 'degrees',
                      'demods/5/phaseshift': 'degrees',
                      'demods/6/phaseshift': 'degrees',
                      'demods/7/phaseshift': 'degrees',
                      'oscs/0/freq': 'Hz',
                      'oscs/1/freq': 'Hz',
                      'sigouts/0/amplitudes/3': 'Volts',
                      'sigouts/0/offset': 'Volts',
                      'sigouts/1/amplitudes/7': 'Volts',
                      'sigouts/1/offset': 'Volts'
                      }
        sp_name = spnamedict[sweepdict['gridnode']]

        self.setpoint_names = ((sp_name,),)*len(signals)
        start = sweepdict['start']
        stop = sweepdict['stop']
        npts = sweepdict['samplecount']
        # TODO: make sure that these setpoints are correct, i.e. actually
        # matching what the UHFLI does
        # TODO: support non-sequential sweep mode
        if sweepdict['xmapping'] == 'lin':
            sw = tuple(np.linspace(start, stop, npts))
        else:
            logstart = np.log10(start)
            logstop = np.log10(stop)
            sw = tuple(np.logspace(logstart, logstop, npts))
        self.setpoints = ((sw,),)*len(signals)
        self.shapes = ((npts,),)*len(signals)

        # Now actually send  the settings to the instrument
        for (setting, value) in sweepdict.items():
            setting = 'sweep/' + setting
            self._instrument.sweeper.set(setting, value)

        self._instrument.sweep_correctly_built = True

    def get(self):
        """
        Execute the sweeper and return the data corresponding to the
        subscribed signals.

        Returns:

            tuple: Tuple containg N numpy arrays where N is the number
              of signals added to the sweep.

        Raises:
            ValueError: If no signals have been added to the sweep
            ValueError: If a sweep setting has been modified since
              the last sweep, but Sweep.build_sweep has not been run
        """
        daq = self._instrument.daq
        signals = self._instrument._sweeper_signals
        sweeper = self._instrument.sweeper

        if signals == []:
            raise ValueError('No signals selected! Can not perform sweep.')

        if self._instrument.sweep_correctly_built is False:
            raise ValueError('The sweep has not been correctly built.' +
                             ' Please run Sweep.build_sweep.')

        # We must enable the demodulators we use.
        # After the sweep, they should be returned to their original state
        streamsettings = []  # This list keeps track of the pre-sweep settings
        for sigstr in signals:
            path = '/'.join(sigstr.split('/')[:-1])
            (_, dev, _, dmnum, _) = path.split('/')

            # If the setting has never changed, get returns an empty dict.
            # In that case, we assume that it's zero (factory default)
            try:
                toget = path.replace('sample', 'enable')
                # ZI like nesting inside dicts...
                setting = daq.get(toget)[dev]['demods'][dmnum]['enable']['value'][0]
            except KeyError:
                setting = 0
            streamsettings.append(setting)
            daq.setInt(path.replace('sample', 'enable'), 1)

            # We potentially subscribe several times to the same demodulator,
            # but that should not be a problem
            sweeper.subscribe(path)

        sweeper.execute()
        timeout = self._instrument.sweeper_timeout.get()
        start = time.time()
        while not sweeper.finished():  # Wait until the sweep is complete, with timeout.
            time.sleep(0.2)  # Check every 200 ms whether the sweep is done
            # Here we could read intermediate data via:
            # data = sweeper.read(True)...
            # and process it while the sweep is completing.
            if (time.time() - start) > timeout:
                # If for some reason the sweep is blocking, force the end of the
                # measurement.
                log.error("Sweep still not finished, forcing finish...")
                # should exit function with error message instead of returning data
                sweeper.finish()

        return_flat_dict = True
        data = sweeper.read(return_flat_dict)

        sweeper.unsubscribe('*')
        for (state, sigstr) in zip(streamsettings, signals):
            path = '/'.join(sigstr.split('/')[:-1])
            daq.setInt(path.replace('sample', 'enable'), int(state))

        return self._parsesweepdata(data)

    def _parsesweepdata(self, sweepresult):
        """
        Parse the raw result of a sweep into just the data asked for by the
        added sweeper signals. Used by Sweep.get.

        Args:
            sweepresult (dict): The dict returned by sweeper.read

        Returns:
            tuple: The requested signals in a tuple
        """
        trans = {'X': 'x', 'Y': 'y', 'Aux Input 1': 'auxin0',
                 'Aux Input 2': 'auxin1', 'R': 'r', 'phase': 'phase',
                 'Xrms': 'xpwr', 'Yrms': 'ypwr', 'Rrms': 'rpwr'}
        returndata = []

        for signal in self._instrument._sweeper_signals:
            path = '/'.join(signal.split('/')[:-1])
            attr = signal.split('/')[-1]
            data = sweepresult[path][0][0][trans[attr]]
            returndata.append(data)

        return tuple(returndata)

class ZIUHFLI(Instrument):
    """
    QCoDeS driver for ZI UHF-LI.

    Currently implementing demodulator settings and the sweeper functionality.

    Requires ZI Lab One software to be installed on the computer running QCoDeS.
    Furthermore, the Data Server and Web Server must be running and a connection
    between the two must be made.

    TODOs:
        * Add the scope
        * Add zoom-FFT
    """

    def __init__(self, name, device_ID, api_level=5, **kwargs):
        """
        Create an instance of the instrument.

        Args:
            name (str): The internal QCoDeS name of the instrument
            device_ID (str): The device name as listed in the web server.
            api_level (int): Compatibility mode of the API interface. Must be 5
              for the UHF.
        """

        super().__init__(name, **kwargs)
        (self.daq, self.device, self.props) = zhinst.utils.create_api_session(device_ID,
                                                                              api_level)

        self.sweeper = self.daq.sweep()
        self.sweeper.set('sweep/device', self.device)

        # this variable enforces building the sweep before using it
        self._sweep_cb = False

        @property
        def sweep_correctly_built(self):
            return self._sweep_cd

        @sweep_correctly_built.setter
        def sweep_correctly_built(self, value):
            if not isinstance(value, bool):
                raise ValueError('sweep_correctly_built')
            self._sweep_cb = value

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

            self.add_parameter('demod{}_phaseshift'.format(demod),
                               label='Phase shift',
                               unit='degrees',
                               get_cmd=partial(self._demod_getter,
                                               demod-1, 'phaseshift'),
                               set_cmd=partial(self._demod_setter,
                                               1, demod-1, 'phaseshift')
                               )

            # val_mapping for the demodX_signalin parameter
            dmsigins = {'Sig In 1': 0,
                        'Sig In 2': 1,
                        'Trigger 1': 2,
                        'Trigger 2': 3,
                        'Aux Out 1': 4,
                        'Aux Out 2': 5,
                        'Aux Out 3': 6,
                        'Aux Out 4': 7,
                        'Aux In 1': 8,
                        'Aux In 2': 9,
                        'Phi Demod 4': 10,
                        'Phi Demod 8': 11}

            self.add_parameter('demod{}_signalin'.format(demod),
                               label='Signal input',
                               get_cmd=partial(self._demod_getter,
                                               demod-1, 'adcselect'),
                               set_cmd=partial(self._demod_setter,
                                               0, demod-1, 'adcselect'),
                               val_mapping=dmsigins,
                               vals=vals.Enum(*list(dmsigins.keys()))
                               )

            self.add_parameter('demod{}_sinc'.format(demod),
                               label='Sinc filter',
                               get_cmd=partial(self._demod_getter,
                                               demod-1, 'sinc'),
                               set_cmd=partial(self._demod_setter,
                                               0, demod-1, 'sinc'),
                               val_mapping={'ON': 1, 'OFF': 0},
                               vals=vals.Enum('ON', 'OFF')
                               )

            self.add_parameter('demod{}_streaming'.format(demod),
                               label='Data streaming',
                               get_cmd=partial(self._demod_getter,
                                               demod-1, 'enable'),
                               set_cmd=partial(self._demod_setter,
                                               0, demod-1, 'enable'),
                               val_mapping={'ON': 1, 'OFF': 0},
                               vals=vals.Enum('ON', 'OFF')
                               )

            dmtrigs = {'Continuous': 0,
                       'Trigger in 3 Rise': 1,
                       'Trigger in 3 Fall': 2,
                       'Trigger in 3 Both': 3,
                       'Trigger in 3 High': 32,
                       'Trigger in 3 Low': 16,
                       'Trigger in 4 Rise': 4,
                       'Trigger in 4 Fall': 8,
                       'Trigger in 4 Both': 12,
                       'Trigger in 4 High': 128,
                       'Trigger in 4 Low': 64,
                       'Trigger in 3|4 Rise': 5,
                       'Trigger in 3|4 Fall': 10,
                       'Trigger in 3|4 Both': 15,
                       'Trigger in 3|4 High': 160,
                       'Trigger in 3|4 Low': 80}

            self.add_parameter('demod{}_trigger'.format(demod),
                               label='Trigger',
                               get_cmd=partial(self._demod_getter,
                                               demod-1, 'trigger'),
                               set_cmd=partial(self._demod_setter,
                                               0, demod-1, 'trigger'),
                               val_mapping=dmtrigs,
                               vals=vals.Enum(*list(dmtrigs.keys()))
                               )

        ########################################
        # SWEEPER PARAMETERS

        self.add_parameter('sweeper_BWmode',
                           label='Sweeper bandwidth control mode',
                           set_cmd=partial(self._sweep_setter,
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
                           set_cmd=partial(self._sweep_setter,
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
                            set_cmd=partial(self._sweep_setter,
                                            'sweep/start'),
                            get_cmd=partial(self._sweep_getter,
                                            'sweep/start'),
                            vals=vals.Numbers(0, 600e6))

        self.add_parameter('sweeper_stop',
                            label='Stop value of the sweep',
                            set_cmd=partial(self._sweep_setter,
                                            'sweep/stop'),
                            get_cmd=partial(self._sweep_getter,
                                            'sweep/stop'),
                            vals=vals.Numbers(0, 600e6))

        self.add_parameter('sweeper_samplecount',
                            label='Length of the sweep (pts)',
                            set_cmd=partial(self._sweep_setter,
                                            'sweep/samplecount'),
                            get_cmd=partial(self._sweep_getter,
                                            'sweep/samplecount'),
                            vals=vals.Ints(0, 100000))

        # val_mapping for sweeper_param parameter
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
                           set_cmd=partial(self._sweep_setter,
                                           'sweep/gridnode'),
                           val_mapping=sweepparams,
                           get_cmd=partial(self._sweep_getter,
                                           'sweep/gridnode'),
                           vals=vals.Enum(*list(sweepparams.keys()))
                           )

        # val_mapping for sweeper_units parameter
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

        # val_mapping for sweeper_mode parameter
        sweepmodes = {'Sequential': 0,
                      'Binary': 1,
                      'Biderectional': 2,
                      'Reverse': 3}

        self.add_parameter('sweeper_mode',
                            label='Sweep mode',
                            set_cmd=partial(self._sweep_setter,
                                            'sweep/scan'),
                            get_cmd=partial(self._sweep_getter, 'sweep/scan'),
                            val_mapping=sweepmodes,
                            vals=vals.Enum(*list(sweepmodes))
                            )

        self.add_parameter('sweeper_order',
                           label='Sweeper filter order',
                           set_cmd=partial(self._sweep_setter,
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
                           set_cmd=partial(self._sweep_setter,
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

        self.add_parameter('sweeper_inaccuracy',
                           label='Demodulator filter settling inaccuracy',
                           set_cmd=partial(self._sweep_setter,
                                           'sweep/settling/inaccuracy'),
                           docstring="""
                                     Demodulator filter settling inaccuracy
                                     defining the wait time between a sweep
                                     parameter change and recording of the
                                     next sweep point. The settling time is
                                     calculated as the time required to attain
                                     the specified remaining proportion [1e-13,
                                     0.1] of an incoming step function. Typical
                                     inaccuracy values: 10m for highest sweep
                                     speed for large signals, 100u for precise
                                     amplitude measurements, 100n for precise
                                     noise measurements. Depending on the
                                     order of the demodulator filter the settling
                                     inaccuracy will define the number of filter
                                     time constants the sweeper has to wait. The
                                     maximum between this value and the settling
                                     time is taken as wait time until the next
                                     sweep point is recorded.
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
                           set_cmd=partial(self._sweep_setter,
                                           'sweep/averaging/sample'),
                           get_cmd=partial(self._sweep_getter,
                                           'sweep/averaging/sample'),
                           vals=vals.Ints(1),
                           docstring="""
                                     The actual number of samples is the
                                     maximum of this value and the
                                     sweeper_averaging_time times the
                                     relevant sample rate.
                                     """
                           )

        self.add_parameter('sweeper_averaging_time',
                           label=('Minimal averaging time'),
                           set_cmd=partial(self._sweep_setter,
                                           'sweep/averaging/tc'),
                           get_cmd=partial(self._sweep_getter,
                                           'sweep/averaging/tc'),
                           unit='s',
                           docstring="""
                                     The actual number of samples is the
                                     maximum of this value times the
                                     relevant sample rate and the
                                     sweeper_averaging_samples."""
                           )

        self.add_parameter('sweeper_xmapping',
                           label='Sweeper x mapping',
                           set_cmd=partial(self._sweep_setter,
                                           'sweep/xmapping'),
                           get_cmd=partial(self._sweep_getter,
                                           'sweep/xmapping'),
                           val_mapping={'lin': 0, 'log': 1}
                           )

        self.add_parameter('sweeper_sweeptime',
                           label='Expected sweep time',
                           unit='s',
                           get_cmd=self._get_sweep_time)

        self.add_parameter('sweeper_timeout',
                           label='Sweep timeout',
                           unit='s',
                           initial_value=600,
                           parameter_class=ManualParameter)

        ########################################
        # SIGNAL INPUTS

        for sigin in range(1, 3):

            self.add_parameter('signal_input{}_range'.format(sigin),
                               label='Input range',
                               set_cmd=partial(self._sigin_setter,
                                               1, sigin-1, 'range'),
                               get_cmd=partial(self._sigin_getter,
                                               sigin-1, 'range'),
                               unit='V')

            self.add_parameter('signal_input{}_scaling'.format(sigin),
                               label='Input scaling',
                               set_cmd=partial(self._sigin_setter,
                                               1, sigin-1, 'scaling'),
                               get_cmd=partial(self._sigin_getter,
                                               sigin-1, 'scaling'),
                               )

            self.add_parameter('signal_input{}_AC'.format(sigin),
                               label='AC coupling',
                               set_cmd=partial(self._sigin_setter,
                                               0, sigin-1, 'ac'),
                               get_cmd=partial(self._sigin_getter,
                                               sigin-1, 'ac'),
                               val_mapping={'ON': 1, 'OFF': 0},
                               vals=vals.Enum('ON', 'OFF')
                               )

            self.add_parameter('signal_input{}_impedance'.format(sigin),
                               label='Input impedance',
                               set_cmd=partial(self._sigin_setter,
                                               0, sigin-1, 'imp50'),
                               get_cmd=partial(self._sigin_getter,
                                               sigin-1, 'imp50'),
                               val_mapping={50: 1, 1000: 0},
                               vals=vals.Enum(50, 1000)
                               )

            sigindiffs = {'Off': 0, 'Inverted': 1, 'Input 1 - Input 2': 2,
                          'Input 2 - Input 1': 3}
            self.add_parameter('signal_input{}_diff'.format(sigin),
                               label='Input signal subtraction',
                               set_cmd=partial(self._sigin_setter,
                                               0, sigin-1, 'diff'),
                               get_cmd=partial(self._sigin_getter,
                                               sigin-1, 'diff'),
                               val_mapping=sigindiffs,
                               vals=vals.Enum(*list(sigindiffs.keys())))

        ########################################
        # THE SWEEP ITSELF
        self.add_parameter('Sweep',
                           parameter_class=Sweep,
                           )

        # A "manual" parameter: a list of the signals for the sweeper
        # to subscribe to
        self._sweeper_signals = []

        # This is the dictionary keeping track of the sweeper settings
        # These are the default settings
        self._sweepdict = {'start': 1e6,
                           'stop': 10e6,
                           'samplecount': 25,
                           'bandwidthcontrol': 1,  # fixed mode
                           'bandwidth': 50,
                           'gridnode': 'oscs/0/freq',
                           'scan': 0,  # sequential scan
                           'order': 1,
                           'settling/time': 1e-6,
                           'settling/inaccuracy': 10e-3,
                           'averaging/sample': 25,
                           'averaging/tc': 100e-3,
                           'xmapping': 0,  # linear
                          }
        # Set up the sweeper with the above settings
        self.Sweep.build_sweep()

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

        returns:
            Union[int, float]: In all cases checked so far, a single value
                is returned.
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

    def _sigin_setter(self, mode, sigin, setting, value):
        """
        General set_cmd for signal input parameters

        This function counts signal inputs in a zero-indexed way.

        Args:
            mode (int): 0 means 'call setInt', 1 means 'call setDouble'
            demod (int): The signal input in question (0 or 1)
            setting (str): The attribute to set, e.g. 'scaling'
            value (Union[int, float]): The value to set the attribute to
        """
        setstr = '/{}/sigins/{}/{}'.format(self.device, sigin, setting)

        if mode == 0:
            self.daq.setInt(setstr, value)
        if mode == 1:
            self.daq.setDouble(setstr, value)

    def _sigin_getter(self, sigin, setting):
        """
        General get_cmd for signal input parameters

        The built-in self.daq.get commands returns a dictionary, but we
        want a single value

        This function counts signal inputs in a zero-indexed way.

        returns:
            Union[int, float]: In all cases checked so far, a single value
                is returned.
        """
        querystr = '/{}/sigins/{}/{}'.format(self.device, sigin, setting)
        returndict = self.daq.get(querystr)
        sigin = str(sigin)
        rawvalue = returndict[self.device]['sigins'][sigin][setting]['value']

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

        # Possible TODO: cut down on the number of instrument
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

    def _sweep_setter(self, setting, value):
        """
        set_cmd for all sweeper parameters. The value and setting are saved in
        a dictionary which is read by the Sweep parameter's build_sweep method
        and only then sent to the instrument.
        """
        key = '/'.join(setting.split('/')[1:])
        self._sweepdict[key] = value
        self.sweep_correctly_built = False

    def _sweep_getter(self, setting):
        """
        General get_cmd for sweeper parameters

        The built-in sweeper.get command returns a dictionary, but we want
        single values.

        Args:
            setting (str): the path used by ZI to describe the setting,
            e.g. 'sweep/settling/time'
        """
        # TODO: Should this look up in _sweepdict rather than query the
        # instrument?
        returndict = self.sweeper.get(setting)  # this is a dict

        # The dict may have different 'depths' depending on the parameter.
        # The depth is encoded in the setting string (number of '/')
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

        # TODO: implement all possibly returned attributes
        valid_attributes = ['X', 'Y', 'R', 'phase', 'Xrms', 'Yrms', 'Rrms']

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
                        ' demodulator {}, since it was'.format(demodulator) +
                        ' not previously added.')
        else:
            self._sweeper_signals.remove(signalstring)

    def print_sweeper_settings(self):
        """
        Pretty-print the current settings of the sweeper.
        If Sweep.build_sweep and Sweep.get are called, the sweep described
        here will be performed.
        """
        print('ACQUISITION')
        toprint = ['sweeper_BWmode', 'sweeper_BW', 'sweeper_order',
                   'sweeper_averaging_samples', 'sweeper_averaging_time',
                   'sweeper_settlingtime', 'sweeper_settlingtc']
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
            fmt = (count, int(dm)+1, attr)
            print('    Signal {}: Demodulator {}: {}'.format(*fmt))
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
        print('    Sweep timeout: {} ({})'.format(self.sweeper_timeout.get(),
                                                  's'))
        ready = self.sweep_correctly_built
        print('    Sweep built and ready to execute: {}'.format(ready))

    def close(self):
        """
        Override of the base class' close function
        """
        self.daq.disconnect()
        super().close()

