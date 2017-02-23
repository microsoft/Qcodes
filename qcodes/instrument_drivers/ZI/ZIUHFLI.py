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




class Scope(MultiParameter):
    """
    Class similar to the Sweeper class
    TO DO: update docs...
    Set scopeMode: time or frequency domaine

    """
    def __init__(self, instrument):
        # The __init__ requires that we supply names and shapes,
        # but there is no way to know what they could be known at this time.
        # They are updated via build_scope.
        super().__init__(name, names=('',), shapes=((1,),), **kwargs)
        self._instrument = instrument

    def build_scope(self):

        scopedict = self._instrument._scopedict

        log.info('Built a scope')

        # define the gridnode

        npts = scopedict['lenght']

        # self.setpoints = ((sw,),)*len(signals)
        # self.shapes = ((npts,),)*len(signals)

        # Send settings to device

        # for (setting, value) in scopedict.items():
        #     setting = 'sweep/' + setting
        #     self._instrument.sweeper.set(setting, value)



        # send the settings saved in scopedict to the device

        self._instrument.daq.sync()
        self._instrument.scope_correctly_built = True

    def get(self):
        """
        read scope data
        """
        # daq = self._instrument.daq
        # use load_labone_zibin() ? loads data saved in binary format from ziControl
        # zi.daq.subscribe('/%s/scopes/0/wave' % device)
        # zi.daq.poll(poll_length, poll_timeout, poll_flags, poll_return_flat_dict)
        # zi.daq.unsubscribe('*')



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
        # create (instantiate) an instance of each module we will use
        self.sweeper = self.daq.sweep()
        self.sweeper.set('sweep/device', self.device)
        self.scope = self.daq.scopeModule()
        # this variable enforces building the sweep and scope before using it
        self._sweep_cb = False
        self._scope_cb = False

        @property
        def sweep_correctly_built(self):
            return self._sweep_cd

        @sweep_correctly_built.setter
        def sweep_correctly_built(self, value):
            if not isinstance(value, bool):
                raise ValueError('sweep_correctly_built')
            self._sweep_cb = value

        @property
        def scope_correctly_built(self):
            return self._scope_cd

        @scope_correctly_built.setter
        def sweep_correctly_built(self, value):
            if not isinstance(value, bool):
                raise ValueError('scope_correctly_built')
            self._scope_cb = value

        ########################################
        # INSTRUMENT PARAMETERS

        ########################################
        # Oscillators
        for oscs in range(1,3):
            self.add_parameter('oscillator{}_freq'.format(oscs),
                               label='Frequency of oscillator {}'.format(oscs),
                               unit='Hz',
                               set_cmd=partial(self._setter, 'oscs',
                                                oscs-1, 1, 'freq'),
                               get_cmd=partial(self._getter, 'oscs',
                                                oscs-1, 1, 'freq'),
                               vals=vals.Numbers(0, 600e6))

        ########################################
        # DEMODULATOR PARAMETERS

        for demod in range(1, 9):
            self.add_parameter('demod{}_order'.format(demod),
                               label='Filter order',
                               get_cmd=partial(self._getter, 'demods',
                                               demod-1, 0, 'order'),
                               set_cmd=partial(self._setter, 'demods',
                                               demod-1, 0, 'order'),
                               vals=vals.Ints(1, 8)
                               )

            self.add_parameter('demod{}_harmonic'.format(demod),
                               label=('Reference frequency multiplication' +
                                      ' factor'),
                               get_cmd=partial(self._getter, 'demods',
                                               demod-1, 1, 'harmonic'),
                               set_cmd=partial(self._setter, 'demods',
                                               demod-1, 1, 'harmonic'),
                               vals=vals.Ints(1, 999)
                               )

            self.add_parameter('demod{}_timeconstant'.format(demod),
                               label='Filter time constant',
                               get_cmd=partial(self._getter, 'demods',
                                               demod-1, 1, 'timeconstant'),
                               set_cmd=partial(self._setter, 'demods',
                                               demod-1, 1, 'timeconstant'),
                               unit='s'
                               )

            self.add_parameter('demod{}_samplerate'.format(demod),
                               label='Sample rate',
                               get_cmd=partial(self._getter, 'demods',
                                               demod-1, 1, 'rate'),
                               set_cmd=partial(self._setter, 'demods',
                                               demod-1, 1, 'rate'),
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
                               get_cmd=partial(self._getter, 'demods',
                                               demod-1, 1, 'phaseshift'),
                               set_cmd=partial(self._setter, 'demods',
                                               demod-1, 1, 'phaseshift')
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
                               get_cmd=partial(self._getter, 'demods',
                                               demod-1, 0,'adcselect'),
                               set_cmd=partial(self._setter, 'demods',
                                               demod-1, 0, 'adcselect'),
                               val_mapping=dmsigins,
                               vals=vals.Enum(*list(dmsigins.keys()))
                               )

            self.add_parameter('demod{}_sinc'.format(demod),
                               label='Sinc filter',
                               get_cmd=partial(self._getter, 'demods',
                                               demod-1, 0, 'sinc'),
                               set_cmd=partial(self._setter, 'demods',
                                               demod-1, 0, 'sinc'),
                               val_mapping={'ON': 1, 'OFF': 0},
                               vals=vals.Enum('ON', 'OFF')
                               )

            self.add_parameter('demod{}_streaming'.format(demod),
                               label='Data streaming',
                               get_cmd=partial(self._getter, 'demods',
                                               demod-1, 0, 'enable'),
                               set_cmd=partial(self._setter, 'demods',
                                               demod-1, 0, 'enable'),
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
                               get_cmd=partial(self._getter, 'demods',
                                               demod-1, 0, 'trigger'),
                               set_cmd=partial(self._setter, 'demods',
                                               demod-1, 0, 'trigger'),
                               val_mapping=dmtrigs,
                               vals=vals.Enum(*list(dmtrigs.keys()))
                               )

        ########################################
        # SIGNAL INPUTS

        for sigin in range(1, 3):

            self.add_parameter('signal_input{}_range'.format(sigin),
                               label='Input range',
                               set_cmd=partial(self._setter, 'sigins',
                                               sigin-1, 1, 'range'),
                               get_cmd=partial(self._getter, 'sigins',
                                               sigin-1, 1, 'range'),
                               unit='V')

            self.add_parameter('signal_input{}_scaling'.format(sigin),
                               label='Input scaling',
                               set_cmd=partial(self._setter, 'sigins',
                                               sigin-1, 1, 'scaling'),
                               get_cmd=partial(self._getter, 'sigins',
                                               sigin-1, 1, 'scaling'),
                               )

            self.add_parameter('signal_input{}_AC'.format(sigin),
                               label='AC coupling',
                               set_cmd=partial(self._setter,'sigins',
                                               sigin-1, 0, 'ac'),
                               get_cmd=partial(self._getter, 'sigins',
                                               sigin-1, 0, 'ac'),
                               val_mapping={'ON': 1, 'OFF': 0},
                               vals=vals.Enum('ON', 'OFF')
                               )

            self.add_parameter('signal_input{}_impedance'.format(sigin),
                               label='Input impedance',
                               set_cmd=partial(self._setter, 'sigins',
                                                sigin-1, 0, 'imp50'),
                               get_cmd=partial(self._getter, 'sigins',
                                               sigin-1, 0, 'imp50'),
                               val_mapping={50: 1, 1000: 0},
                               vals=vals.Enum(50, 1000)
                               )

            sigindiffs = {'Off': 0, 'Inverted': 1, 'Input 1 - Input 2': 2,
                          'Input 2 - Input 1': 3}
            self.add_parameter('signal_input{}_diff'.format(sigin),
                               label='Input signal subtraction',
                               set_cmd=partial(self._setter, 'sigins',
                                                sigin-1, 0, 'diff'),
                               get_cmd=partial(self._getter, 'sigins',
                                               sigin-1, 0, 'diff'),
                               val_mapping=sigindiffs,
                               vals=vals.Enum(*list(sigindiffs.keys())))

        ########################################
        # SIGNAL OUTPUTS
        outputamps = {1: 'amplitudes/3', 2: 'amplitudes/7'}
        outputampenable = {1: 'enables/3', 2: 'enables/7'}

        for sigout in range(1,3):

            self.add_parameter('signal_output{}_on'.format(sigout),
                                label='Turn signal output on and off.',
                                set_cmd=partial(self._sigout_setter,
                                                sigout-1, 0, 'on'),
                                get_cmd=partial(self._sigout_getter,
                                                sigout-1, 0, 'on'),
                                val_mapping={'ON': 1, 'OFF': 0},
                                vals=vals.Enum('ON', 'OFF') )

            self.add_parameter('signal_output{}_imp50'.format(sigout),
                                label='Switch to turn on 50 Ohm impedance',
                                set_cmd=partial(self._sigout_setter,
                                                sigout-1, 0, 'imp50'),
                                get_cmd=partial(self._sigout_getter,
                                                sigout-1, 0, 'imp50'),
                                val_mapping={'ON': 1, 'OFF': 0},
                                vals=vals.Enum('ON', 'OFF') )

            self.add_parameter('signal_output{}_amplitude'.format(sigout),
                                label='Signal output amplitude',
                                set_cmd=partial(self._sigout_setter,
                                                sigout-1, 1, outputamps[sigout]),
                                get_cmd=partial(self._sigout_getter,
                                               sigout-1, 1, outputamps[sigout]),
                                unit='V')

            self.add_parameter('signal_output{}_ampdef'.format(sigout),
                                parameter_class=ManualParameter,
                                initial_value='Vpk',
                                label="Signal output amplitude's definition",
                                unit='V',
                                vals=vals.Enum('Vpk','Vrms', 'dBm'))

            self.add_parameter('signal_output{}_range'.format(sigout),
                                label='Signal output range',
                                set_cmd=partial(self._sigout_setter,
                                                sigout-1, 1, 'range'),
                                get_cmd=partial(self._sigout_getter,
                                                sigout-1, 1, 'range'),
                                vals=vals.Enum(0.075, 0.15, 0.75, 1.5))

            self.add_parameter('signal_output{}_offset'.format(sigout),
                                label='Signal output offset',
                                set_cmd=partial(self._sigout_setter,
                                                sigout-1, 1, 'offset'),
                                get_cmd=partial(self._sigout_getter,
                                                sigout-1, 1, 'offset'),
                                vals=vals.Numbers(-1.5, 1.5),
                                unit='V')

            self.add_parameter('signal_output{}_autorange'.format(sigout),
                                label='Enable signal output range.',
                                set_cmd=partial(self._sigout_setter,
                                                sigout-1, 0, 'autorange'),
                                get_cmd=partial(self._sigout_getter,
                                                sigout-1, 0, 'autorange'),
                                val_mapping={'ON': 1, 'OFF': 0},
                                vals=vals.Enum('ON', 'OFF') )

            self.add_parameter('signal_output{}_enable'.format(sigout),
                                label="Enable signal output's amplitude.",
                                set_cmd=partial(self._sigout_setter,
                                                sigout-1, 0, outputampenable[sigout]),
                                get_cmd=partial(self._sigout_getter,
                                                sigout-1, 0, outputampenable[sigout]),
                                val_mapping={'ON': 1, 'OFF': 0},
                                vals=vals.Enum('ON', 'OFF') )



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

        ########################################
        # SCOPE PARAMETERS
        # default parameters:

        self.add_parameter('scope_mode',
                            label="Scope's mode: time or frequency domain.",
                            set_cmd=partial(self._scope_setter, 1, 0,
                                            'mode'),
                            get_cmd=partial(self._scope_getter, 'mode'),
                            val_mapping={'Time Domain': 1,
                                         'Freq Domain FFT': 3},
                            vals=vals.Enum('Time Domain', 'Freq Domain FFT')
                            )

        self._samplingrate_codes = {'1.80 Ghz': 0,
                                   '900 MHz': 1,
                                   '450 MHz': 2,
                                   '225 MHz': 3,
                                   '113 MHz': 4,
                                   '56.2 MHz': 5,
                                   '28.1 MHz': 6,
                                   '14.0 MHz': 7,
                                   '7.03 MHz': 8,
                                   '3.50 MHz': 9,
                                   '1.75 MHz': 10,
                                   '880 kHz': 11,
                                   '440 kHz': 12,
                                   '220 kHz': 13,
                                   '110 kHz': 14,
                                   '54.9 kHz': 15,
                                   '27.5 kHz': 16}

        self.add_parameter('scope_samplingrate',
                            label="Scope's sampling rate",
                            set_cmd=partial(self._scope_setter, 0, 0,
                                            'time'),
                            get_cmd=partial(self._getter, 'scopes', 0,
                                            0, 'time'),
                            val_mapping=self._samplingrate_codes,
                            vals=vals.Enum(*list(self._samplingrate_codes.keys()))
                            )

        self.add_parameter('scope_length',
                            label="Length of scope trace (pts)",
                            set_cmd=partial(self._scope_setter, 0, 1,
                                            'length'),
                            get_cmd=partial(self._getter, 'scopes', 0,
                                            1, 'length'),
                            vals=vals.Numbers(4096, 128000000),
                            get_parser=int
                            )
        #scope duration: ManualParameter? validation depends on value of length
        self.add_parameter('scope_duration',
                           label="Scope trace duration",
                           set_cmd=partial(self._scope_setter, 0, 0,
                                           'duration'),
                           get_cmd=partial(self._scope_getter,
                                           'duration'),
                           vals=vals.Numbers(2.27e-6,4.660e3),
                           unit='s'
                           )

        inputselect = {'Signal Input 1': 0,
                       'Signal Input 2': 1,
                       'Trig Input 1': 2,
                       'Trig Input 2': 3,
                       'Aux Output 1': 4,
                       'Aux Output 2': 5,
                       'Aux Output 3': 6,
                       'Aux Output 4': 7,
                       'Aux In 1 Ch 1': 8,
                       'Aux In 1 Ch 2': 9,
                       'Osc phi Demod 4': 10,
                       'Osc phi Demod 8': 11,
                       'AU Cartesian 1': 112,
                       'AU Cartesian 2': 113,
                       'AU Polar 1': 128,
                       'AU Polar 2': 129,
                       }

        for demod in range(1,9):
            inputselect['Demod {} X'.format(demod)] = 15+demod
            inputselect['Demod {} Y'.format(demod)] = 31+demod
            inputselect['Demod {} R'.format(demod)] = 47+demod
            inputselect['Demod {} Phase'.format(demod)] = 63+demod

        for channel in range(1,3):
            self.add_parameter('scope_channel{}_input'.format(channel),
                            label="Scope's channel {} input source".format(channel),
                            set_cmd=partial(self._scope_setter, 0, 0,
                                            '{}/inputselect'.format(channel-1)),
                            get_cmd=partial(self._getter, 'scopes', 0,
                                            0, '{}/inputselect'.format(channel-1)),
                            val_mapping=inputselect,
                            vals=vals.Enum(*list(inputselect.keys()))
                            )
        #TO DO: Implement the average filter correctly. Use a parameter, function or method?

        # self.add_parameter('scope_avgfilter',
        #                     label="Scope's Avg Filter",
        #                     set_cmd=partial(self._scope_setter, 1,
        #                                     '/averager/'),
        #                     get_cmd=partial(self._scope_getter,
        #                                     '/averager/'),
        #                    # val_mapping={'None': 0, 'Exp Moving Avg':}, # to do: double check when it sets it to 1
        #                     vals=vals.Enum(*list(inputselect.keys()))
        #                     )

        self.add_parameter('scope_average_weight',
                            label="Scope Averages",
                            set_cmd=partial(self._scope_setter, 1, 0,
                                            '/averager/weight'),
                            get_cmd=partial(self._scope_getter,
                                            '/averager/weight'),
                            vals=vals.Numbers(min_value=1)
                            )

        self.add_parameter('scope_trig_enable',
                            label="Enable triggering for scope readout",
                            set_cmd=partial(self._setter, 'scopes', 0,
                                            0, 'trigenable'),
                            get_cmd=partial(self._getter, 'scopes', 0,
                                            0, 'trigenable'),
                            val_mapping={'ON': 1, 'OFF': 0},
                            vals=vals.Enum('ON', 'OFF')
                            )

        self.add_parameter('scope_trig_signal',
                            label="Trigger signal source",
                            set_cmd=partial(self._setter, 'scopes', 0,
                                            0, 'trigchannel'),
                            get_cmd=partial(self._getter, 'scopes', 0,
                                            0, 'trigchannel'),
                            val_mapping=inputselect,
                            vals=vals.Enum(*list(inputselect.keys()))
                            )

        slopes = {'None': 0, 'Rise': 1, 'Fall': 2, 'Both': 3}

        self.add_parameter('scope_trig_slope',
                            label="Scope's triggering slope",
                            set_cmd=partial(self._setter, 'scopes', 0,
                                            0, 'trigslope'),
                            get_cmd=partial(self._getter, 'scopes', 0,
                                            0, 'trigslope'),
                            val_mapping=slopes,
                            vals=vals.Enum(*list(slopes.keys()))
                            )

        # TODO: figure out how value/percent works for the trigger level
        self.add_parameter('scope_trig_level',
                            label="Scope trigger level",
                            set_cmd=partial(self._setter, 'scopes', 0,
                                            1, 'triglevel'),
                            get_cmd=partial(self._getter, 'scopes', 0,
                                            1, 'triglevel'),
                            vals=vals.Numbers()
                            )
        #TO DO: Find out how to enable/disable trigger lever. GUI's button does
        # not appear in logging.

        self.add_parameter('scope_trig_hystmode',
                            label="Enable triggering for scope readout.",
                            set_cmd=partial(self._setter, 'scopes', 0,
                                            0, 'trighysteresis/mode'),
                            get_cmd=partial(self._getter, 'scopes', 0,
                                            0, 'trighysteresis/mode'),
                            val_mapping={'deg': 0, '%': 1},
                            vals=vals.Enum('ON', 'OFF')
                            )

        self.add_parameter('scope_trig_hystrelative',
                            label="Trigger hysteresis, relative value in %",
                            set_cmd=partial(self._setter, 'scopes', 0,
                                            1, 'trighysteresis/relative'),
                            get_cmd=partial(self._getter, 'scopes', 0,
                                            1, 'trighysteresis/relative'),
                            # val_mapping= lambda x: 0.01*x,
                            vals=vals.Numbers(0)
                            )

        self.add_parameter('scope_trig_hystabsolute',
                            label="Trigger hysteresis, absolute value",
                            set_cmd=partial(self._setter, 'scopes', 0,
                                            1, 'trighysteresis/absolute'),
                            get_cmd=partial(self._getter, 'scopes', 0,
                                            1, 'trighysteresis/absolute'),
                            vals=vals.Numbers(0, 20)
                            )

        # make this a slave parameter off scope_holdoff_seconds
        # and scope_holdoff_events
        self.add_parameter('scope_trig_holdoffmode',
                            label="Scope trigger holdoff mode",
                            set_cmd=partial(self._setter, 'scopes', 0,
                                            0, 'trigholdoffmode'),
                            get_cmd=partial(self._getter, 'scopes', 0,
                                            0, 'trigholdoffmode'),
                            val_mapping={'s': 0, 'events': 1},
                            vals=vals.Enum('s', 'events')
                            )

        self.add_parameter('scope_trig_holdoffseconds',
                           label='Scope trigger holdoff',
                           set_cmd=partial(self._scope_setter, 0, 1,
                                           'trigholdoff'),
                           get_cmd=partial(self._getter, 'scopes', 0,
                                           1, 'trigholdoff'),
                           unit='s',
                           vals=vals.Numbers(20e-6, 10)
                           )

        self.add_parameter('scope_segments',
                           label='No. of segments returned by scope',
                           set_cmd=partial(self._setter, 'scopes', 0, 1,
                                           'segments/count'),
                           get_cmd=partial(self._getter, 'scopes', 0, 1,
                                          'segments/count'),
                           vals=vals.Ints(1, 32768)
                           )

        self.add_function('scope_reset_avg',
                            call_cmd=partial(self.scope.set,
                                             'scopeModule/averager/restart', 1),
                            )

    def _setter(self, module, number, mode, setting, value):
        """
        General function to set/send settings to the device.

        The module (e.g demodulator, input, output,..) number is counted in a
        zero indexed fashion.

        Args:
            module (str): The module (eg. demodulator, input, output, ..)
                to set.
            number (int): Module's index
            mode (bool): Indicating whether we are setting an int or double
            setting (str): The module's setting to set.
            value (int/double): The value to set.
        """

        setstr = '/{}/{}/{}/{}'.format(self.device, module, number, setting)

        if mode == 0:
            self.daq.setInt(setstr, value)
        if mode == 1:
            self.daq.setDouble(setstr, value)

    def _getter(self, module, number, mode, setting):
        """
        General get function for generic parameters. Note that some parameters
        use more specialised setter/getters.

        The module (e.g demodulator, input, output,..) number is counted in a
        zero indexed fashion.

        Args:
            module (str): The module (eg. demodulator, input, output, ..)
                we want to know the value of.
            number (int): Module's index
            mode (int): Indicating whether we are asking for an int or double.
                0: Int, 1: double.
            setting (str): The module's setting to set.
        returns:
            inquered value

        """

        querystr = '/{}/{}/{}/{}'.format(self.device, module, number, setting)
        if mode == 0:
            value = self.daq.getInt(querystr)
        if mode == 1:
            value = self.daq.getDouble(querystr)

        # Weird exception, samplingrate returns a string
        return value

    def _sigout_setter(self, number, mode, setting, value):
        """
        Function to set signal output's settings. Specific setter function is needed as
        parameters depend on each other and need to be checked and updated accordingly.

        Args:
            number (int):
            mode (bool): Indicating whether we are asking for an int or double
            setting (str): The module's setting to set.
            value (int/float):
        """

        def amp_valid():
            nonlocal value
            ampdef_val = self.parameters['signal_output{}_ampdef'.format(number+1)].get()
            autorange_val =  self.parameters['signal_output{}_autorange'.format(number+1)].get()

            if autorange_val == 'ON':
                imp50_val =  self.parameters['signal_output{}_imp50'.format(number+1)].get()
                imp50_dic = {'OFF': 1.5, 'ON': 0.75}
                range_val = imp50_dic[imp50_val]

            else:
                range_val =  round(self.parameters['signal_output{}_range'.format(number+1)].get(),3)

            amp_val_dict={'Vpk': lambda value: value,
                        'Vrms': lambda value: value*sqrt(2),
                        'dBm': lambda value: 10**((value-10)/20) }

            if -range_val < amp_val_dict[ampdef_val](value) > range_val:
                raise ValueError('Signal Output:'
                                + ' Amplitude too high for chosen range.')
            value = amp_val_dict[ampdef_val](value)

        def offset_valid():
            nonlocal value
            nonlocal number
            range_val =  round(self.parameters['signal_output{}_range'.format(number+1)].get(),3)
            amp_val = round(self.parameters['signal_output{}_amplitude'.format(number+1)].get(),3)
            autorange_val = self.parameters['signal_output{}_autorange'.format(number+1)].get()
            if -range_val< value+amp_val > range_val:
                raise ValueError('Signal Output: Offset too high for chosen range.')

        def range_valid():
            nonlocal value
            nonlocal number
            autorange_val =  self.parameters['signal_output{}_autorange'.format(number+1)].get()
            imp50_val =  self.parameters['signal_output{}_imp50'.format(number+1)].get()
            imp50_dic = {'OFF': [1.5, 0.15], 'ON': [0.75, 0.075]}

            if autorange_val == "ON":
                raise ValueError('Signal Output :' \
                                + ' Cannot set range as autorange is turned on.')

            if value not in imp50_dic[imp50_val]:
                raise ValueError('Signal Output: Choose a valid range:' \
                                    + '[0.75, 0.075] if imp50 is on, [1.5, 0.15] otherwise.')

        def ampdef_valid():
            # check which amplitude definition you can use. dBm is only with imp50 "ON" possible
             imp50_val =  self.parameters['signal_output{}_imp50'.format(number+1)].get()
             imp50_ampdef_dict = {'ON': ['Vpk','Vrms', 'dBm'], 'OFF': ['Vpk','Vrms']}
             if value not in imp50_ampdef_dict[imp50_val]:
                raise ValueError('Signal Output: Choose a valid amplitude definition' \
                                + "['Vpk','Vrms', 'dBm'] if imp50 is on, ['Vpk','Vrms'] otherwise.")

        def imp50_valid():
            amp_val = round(self.parameters['signal_output{}_amplitude'.format(number+1)].get(),3)
            amp_val = round(self.parameters['signal_output{}_offset'.format(number+1)].get(),3)

        dynamic_validation = {'range': range_valid,
                            'ampdef': ampdef_valid,
                            'amplitudes/3': amp_valid,
                            'amplitudes/7': amp_valid,
                            'offset': offset_valid}


        def update_range_offset_amp():
            range_val = self.parameters['signal_output{}_range'.format(number+1)].get()
            offset_val = self.parameters['signal_output{}_offset'.format(number+1)].get()
            amp_val = self.parameters['signal_output{}_amplitude'.format(number+1)].get()
            if -range_val < offset_val + amp_val > range_val:
                #The GUI would allow higher values but it would clip the signal.
                raise ValueError('Signal Output: Amplitude and/or offset out of range.')

        def update_offset():
            self.parameters['signal_output{}_offset'.format(number+1)].get()

        def update_amp():
            self.parameters['signal_output{}_amplitude'.format(number+1)].get()

        def update_range():
            self.parameters['signal_output{}_autorange'.format(number+1)].get()

        # parameters whic will potentially change other parameters
        changing_param = {'imp50': [update_range_offset_amp, update_range],
                        'autorange': [update_range],
                        'range': [update_offset, update_amp],
                        'amplitudes/3': [update_range, update_amp], # needed if we are setting in dBm: switchy knob
                        'amplitudes/7': [update_range, update_amp], # needed if we are setting in dBm: switchy knob
                        'offset': [update_range]} # range can change if autorange is selected

        setstr = '/{}/sigouts/{}/{}'.format(self.device, number, setting)

        if setting in dynamic_validation:
            dynamic_validation[setting]()

        if mode == 0:
            self.daq.setInt(setstr, value)
        if mode == 1:
            self.daq.setDouble(setstr, value)

        if setting in changing_param:
            [f() for f in changing_param[setting]]

    def _sigout_getter(self, number, mode, setting):
        """
        Function to query the settings of signal outputs. Specific setter function is needed as
        parameters depend on each other and need to be checked and updated accordingly.

        Args:
            number (int):
            mode (bool): Indicating whether we are asking for an int or double
            setting (str): The module's setting to set.
        """

        querystr = '/{}/sigouts/{}/{}'.format(self.device, number, setting)
        if mode == 0:
            value = self.daq.getInt(querystr)
        if mode == 1:
            value = self.daq.getDouble(querystr)

        return value

    def _list_nodes(self, node):
        """
        Returns a list with all nodes in the sub-tree below the specified node.

        Args:
            node (str): Module of which you want to know the parameters.
        return:
            list of sub-nodes
        """
        node_list = self.daq.getList('/{}/{}/'.format(self.device, node))
        return node_list

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
            rates.append(self._getter('demods', demod, 1, 'rate'))
        rate = min(rates)

        if mode == 'current':
            tcs = []
            for demod in demods:
                tcs.append(self._getter('demods', demod, 1, 'timeconstant'))

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

    def _scope_setter(self, scopemodule, mode, setting, value):
        """
        set_cmd for all scope parameters. The value and setting are saved in
        a dictionary which is read by the Scope parameter's build_scope method
        and only then sent to the instrument.

        Args:
            scopemodule (int): Indicates whether this is a setting of the
                scopeModule or not. 1: it is a scopeModule setting,
                0: it is not.
            mode (int): Indicates whether we are setting an int or a float.
                0: int, 1: float. NOTE: Ignored if scopemodule==1.
            setting (str): The setting, e.g. 'length'.
            value (Union[int, float, str]): The value to set.
        """
        # Because setpoints need to be built
        self.scope_correctly_built = False

        # Some parameters are linked to each other in specific ways
        # Therefore, we need special actions for setting these parameters

        SRtranslation = {'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9,
                         'khz': 1e3, 'Mhz': 1e6, 'Ghz': 1e9}

        def setlength(value):
            # TODO: add validation. The GUI seems to correect this value
            self.daq.setDouble('/{}/scopes/0/length'.format(self.device),
                               value)
            SR_str = self.parameters['scope_samplingrate'].get()
            (number, unit) = SR_str.split(' ')
            SR = float(number)*SRtranslation[unit]
            self.parameters['scope_duration']._save_val(value/SR)
            self.daq.setInt('/{}/scopes/0/length'.format(self.device), value)

        def setduration(value):
            # TODO: validation?
            SR_str = self.parameters['scope_samplingrate'].get()
            (number, unit) = SR_str.split(' ')
            SR = float(number)*SRtranslation[unit]
            N = int(np.round(value*SR))
            self.parameters['scope_length']._save_val(N)
            self.parameters['scope_duration']._save_val(value)
            self.daq.setInt('/{}/scopes/0/length'.format(self.device), N)

        def setholdoffseconds(value):
            self.parameters['scope_trig_holdoffmode'].set('s')
            self.daq.setDouble('/{}/scopes/0/trigholdoff'.format(self.device),
                               value)

        def setsamplingrate(value):
            # When the sample rate is changed, the number of points of the trace
            # remains unchanged and the duration changes accordingly
            newSR_str = dict(zip(self._samplingrate_codes.values(),
                                 self._samplingrate_codes.keys()))[value]
            (number, unit) = newSR_str.split(' ')
            newSR = float(number)*SRtranslation[unit]
            oldSR_str = self.parameters['scope_samplingrate'].get()
            (number, unit) = oldSR_str.split(' ')
            oldSR = float(number)*SRtranslation[unit]
            oldduration = self.parameters['scope_duration'].get()
            newduration = oldduration*oldSR/newSR
            self.parameters['scope_duration']._save_val(newduration)
            self.daq.setInt('/{}/scopes/0/time'.format(self.device), value)

        specialcases = {'length': setlength,
                        'duration': setduration,
                        'scope_trig_holdoffseconds': setholdoffseconds,
                        'time': setsamplingrate}

        if setting in specialcases:
            specialcases[setting](value)
            self.daq.sync()
            return
        else:
            # We have two different parameter types: those under
            # /scopes/0/ and those under scopeModule/
            if scopemodule:
                self.daq.set('scopeModule/{}'.format(setting), value)
            elif mode == 0:
                self.daq.setInt('/{}/scopes/0/{}'.format(self.device,
                                                         setting), value)
            elif mode == 1:
                self.daq.setDouble('/{}/scopes/0/{}'.format(self.device,
                                                            setting), value)
            return

    def _scope_getter(self, setting):
        """
        get_cmd for scopeModule parameters

        """
        # There are a few special cases
        SRtranslation = {'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9,
                         'khz': 1e3, 'Mhz': 1e6, 'Ghz': 1e9}

        def getduration():
            SR_str = self.parameters['scope_samplingrate'].get()
            (number, unit) = SR_str.split(' ')
            SR = float(number)*SRtranslation[unit]
            N = self.parameters['scope_length'].get()
            duration = N/SR
            return duration

        specialcases = {'duration': getduration}

        if setting in specialcases:
            value = specialcases[setting]()
        else:
            querystr = 'scopeModule/' + setting
            returndict =  self.scope.get(querystr)
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

    def close(self):
        """
        Override of the base class' close function
        """
        self.daq.disconnect()
        super().close()
