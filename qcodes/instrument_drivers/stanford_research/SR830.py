import time
from functools import partial
from typing import Any, Iterable, Tuple, Union

import numpy as np

from qcodes import VisaInstrument
from qcodes.instrument.parameter import (
    ArrayParameter,
    Parameter,
    ParameterWithSetpoints,
    ParamRawDataType,
)
from qcodes.utils.validators import Arrays, ComplexNumbers, Enum, Ints, Numbers, Strings


class ChannelTrace(ParameterWithSetpoints):
    """
    Parameter class for the two channel buffers
    """

    def __init__(self, name: str, channel: int, **kwargs: Any) -> None:
        """
        Args:
            name: The name of the parameter
            channel: The relevant channel (1 or 2). The name should
                should match this.
        """
        super().__init__(name, **kwargs)

        self._valid_channels = (1, 2)

        if channel not in self._valid_channels:
            raise ValueError('Invalid channel specifier. SR830 only has '
                             'channels 1 and 2.')

        if not isinstance(self.root_instrument, SR830):
            raise ValueError('Invalid parent instrument. ChannelBuffer '
                             'can only live on an SR830.')

        self.channel = channel
        self.update_unit()

    def update_unit(self) -> None:
        assert isinstance(self.root_instrument, SR830)
        params = self.root_instrument.parameters
        if params[f'ch{self.channel}_ratio'].get() != 'none':
            self.unit = '%'
        else:
            disp = params[f'ch{self.channel}_display'].get()
            if disp == 'Phase':
                self.unit = 'deg'
            else:
                self.unit = 'V'
            self.label = disp

    def get_raw(self) -> ParamRawDataType:
        """
        Get command. Returns numpy array
        """
        assert isinstance(self.root_instrument, SR830)
        N = self.root_instrument.buffer_npts()
        if N == 0:
            raise ValueError('No points stored in SR830 data buffer.'
                             ' Can not poll anything.')

        # poll raw binary data
        self.root_instrument.write(f'TRCL ? {self.channel}, 0, {N}')
        rawdata = self.root_instrument.visa_handle.read_raw()

        # parse it
        realdata = np.frombuffer(rawdata, dtype="<i2")
        numbers = realdata[::2] * 2.0 ** (realdata[1::2] - 124)

        return numbers


class ChannelBuffer(ArrayParameter):
    """
    Parameter class for the two channel buffers

    Currently always returns the entire buffer
    TODO (WilliamHPNielsen): Make it possible to query parts of the buffer.
    The instrument natively supports this in its TRCL call.
    """

    def __init__(self, name: str, instrument: 'SR830', channel: int) -> None:
        """
        Args:
            name: The name of the parameter
            instrument: The parent instrument
            channel: The relevant channel (1 or 2). The name should
                should match this.
        """
        self._valid_channels = (1, 2)

        if channel not in self._valid_channels:
            raise ValueError('Invalid channel specifier. SR830 only has '
                             'channels 1 and 2.')

        if not isinstance(instrument, SR830):
            raise ValueError('Invalid parent instrument. ChannelBuffer '
                             'can only live on an SR830.')

        super().__init__(
            name,
            shape=(1,),  # dummy initial shape
            unit="V",  # dummy initial unit
            setpoint_names=("Time",),
            setpoint_labels=("Time",),
            setpoint_units=("s",),
            docstring="Holds an acquired (part of the) data buffer of one channel.",
            instrument=instrument,
        )

        self.channel = channel

    def prepare_buffer_readout(self) -> None:
        """
        Function to generate the setpoints for the channel buffer and
        get the right units
        """
        assert isinstance(self.instrument, SR830)
        N = self.instrument.buffer_npts()  # problem if this is zero?
        # TODO (WilliamHPNielsen): what if SR was changed during acquisition?
        SR = self.instrument.buffer_SR()
        if SR == 'Trigger':
            self.setpoint_units = ('',)
            self.setpoint_names = ('trig_events',)
            self.setpoint_labels = ('Trigger event number',)
            self.setpoints = (tuple(np.arange(0, N)),)
        else:
            dt = 1/SR
            self.setpoint_units = ('s',)
            self.setpoint_names = ('Time',)
            self.setpoint_labels = ('Time',)
            self.setpoints = (tuple(np.linspace(0, N*dt, N)),)

        self.shape = (N,)

        params = self.instrument.parameters
        # YES, it should be: comparing to the string 'none' and not
        # the None literal
        if params[f'ch{self.channel}_ratio'].get() != 'none':
            self.unit = '%'
        else:
            disp = params[f'ch{self.channel}_display'].get()
            if disp == 'Phase':
                self.unit = 'deg'
            else:
                self.unit = 'V'

        if self.channel == 1:
            self.instrument._buffer1_ready = True
        else:
            self.instrument._buffer2_ready = True

    def get_raw(self) -> ParamRawDataType:
        """
        Get command. Returns numpy array
        """
        assert isinstance(self.instrument, SR830)
        if self.channel == 1:
            ready = self.instrument._buffer1_ready
        else:
            ready = self.instrument._buffer2_ready

        if not ready:
            raise RuntimeError('Buffer not ready. Please run '
                               'prepare_buffer_readout')
        N = self.instrument.buffer_npts()
        if N == 0:
            raise ValueError('No points stored in SR830 data buffer.'
                             ' Can not poll anything.')

        # poll raw binary data
        self.instrument.write(f"TRCL ? {self.channel}, 0, {N}")
        rawdata = self.instrument.visa_handle.read_raw()

        # parse it
        realdata = np.frombuffer(rawdata, dtype="<i2")
        numbers = realdata[::2] * 2.0 ** (realdata[1::2] - 124)
        if self.shape[0] != N:
            raise RuntimeError(
                f"SR830 got {N} points in buffer expected {self.shape[0]}"
            )
        return numbers


class SR830(VisaInstrument):
    """
    This is the qcodes driver for the Stanford Research Systems SR830
    Lock-in Amplifier
    """

    _VOLT_TO_N = {2e-9:    0, 5e-9:    1, 10e-9:  2,
                  20e-9:   3, 50e-9:   4, 100e-9: 5,
                  200e-9:  6, 500e-9:  7, 1e-6:   8,
                  2e-6:    9, 5e-6:   10, 10e-6:  11,
                  20e-6:  12, 50e-6:  13, 100e-6: 14,
                  200e-6: 15, 500e-6: 16, 1e-3:   17,
                  2e-3:   18, 5e-3:   19, 10e-3:  20,
                  20e-3:  21, 50e-3:  22, 100e-3: 23,
                  200e-3: 24, 500e-3: 25, 1:      26}
    _N_TO_VOLT = {v: k for k, v in _VOLT_TO_N.items()}

    _CURR_TO_N = {2e-15:    0, 5e-15:    1, 10e-15:  2,
                  20e-15:   3, 50e-15:   4, 100e-15: 5,
                  200e-15:  6, 500e-15:  7, 1e-12:   8,
                  2e-12:    9, 5e-12:   10, 10e-12:  11,
                  20e-12:  12, 50e-12:  13, 100e-12: 14,
                  200e-12: 15, 500e-12: 16, 1e-9:    17,
                  2e-9:    18, 5e-9:    19, 10e-9:   20,
                  20e-9:   21, 50e-9:   22, 100e-9:  23,
                  200e-9:  24, 500e-9:  25, 1e-6:    26}
    _N_TO_CURR = {v: k for k, v in _CURR_TO_N.items()}

    _VOLT_ENUM = Enum(*_VOLT_TO_N.keys())
    _CURR_ENUM = Enum(*_CURR_TO_N.keys())

    _INPUT_CONFIG_TO_N = {
        'a': 0,
        'a-b': 1,
        'I 1M': 2,
        'I 100M': 3,
    }

    _N_TO_INPUT_CONFIG = {v: k for k, v in _INPUT_CONFIG_TO_N.items()}

    def __init__(self, name: str, address: str, **kwargs: Any):
        super().__init__(name, address, **kwargs)

        # Reference and phase
        self.add_parameter('phase',
                           label='Phase',
                           get_cmd='PHAS?',
                           get_parser=float,
                           set_cmd='PHAS {:.2f}',
                           unit='deg',
                           vals=Numbers(min_value=-360, max_value=729.99))

        self.add_parameter('reference_source',
                           label='Reference source',
                           get_cmd='FMOD?',
                           set_cmd='FMOD {}',
                           val_mapping={
                               'external': 0,
                               'internal': 1,
                           },
                           vals=Enum('external', 'internal'))

        self.add_parameter('frequency',
                           label='Frequency',
                           get_cmd='FREQ?',
                           get_parser=float,
                           set_cmd='FREQ {:.4f}',
                           unit='Hz',
                           vals=Numbers(min_value=1e-3, max_value=102e3))

        self.add_parameter('ext_trigger',
                           label='External trigger',
                           get_cmd='RSLP?',
                           set_cmd='RSLP {}',
                           val_mapping={
                               'sine': 0,
                               'TTL rising': 1,
                               'TTL falling': 2,
                           })

        self.add_parameter('harmonic',
                           label='Harmonic',
                           get_cmd='HARM?',
                           get_parser=int,
                           set_cmd='HARM {:d}',
                           vals=Ints(min_value=1, max_value=19999))

        self.add_parameter('amplitude',
                           label='Amplitude',
                           get_cmd='SLVL?',
                           get_parser=float,
                           set_cmd='SLVL {:.3f}',
                           unit='V',
                           vals=Numbers(min_value=0.004, max_value=5.000))

        # Input and filter
        self.add_parameter('input_config',
                           label='Input configuration',
                           get_cmd='ISRC?',
                           get_parser=self._get_input_config,
                           set_cmd='ISRC {}',
                           set_parser=self._set_input_config,
                           vals=Enum(*self._INPUT_CONFIG_TO_N.keys()))

        self.add_parameter('input_shield',
                           label='Input shield',
                           get_cmd='IGND?',
                           set_cmd='IGND {}',
                           val_mapping={
                               'float': 0,
                               'ground': 1,
                           })

        self.add_parameter('input_coupling',
                           label='Input coupling',
                           get_cmd='ICPL?',
                           set_cmd='ICPL {}',
                           val_mapping={
                               'AC': 0,
                               'DC': 1,
                           })

        self.add_parameter('notch_filter',
                           label='Notch filter',
                           get_cmd='ILIN?',
                           set_cmd='ILIN {}',
                           val_mapping={
                               'off': 0,
                               'line in': 1,
                               '2x line in': 2,
                               'both': 3,
                           })

        # Gain and time constant
        self.add_parameter(name='sensitivity',
                           label='Sensitivity',
                           get_cmd='SENS?',
                           set_cmd='SENS {:d}',
                           get_parser=self._get_sensitivity,
                           set_parser=self._set_sensitivity
                           )

        self.add_parameter('reserve',
                           label='Reserve',
                           get_cmd='RMOD?',
                           set_cmd='RMOD {}',
                           val_mapping={
                               'high': 0,
                               'normal': 1,
                               'low noise': 2,
                           })

        self.add_parameter('time_constant',
                           label='Time constant',
                           get_cmd='OFLT?',
                           set_cmd='OFLT {}',
                           unit='s',
                           val_mapping={
                               10e-6:  0, 30e-6:  1,
                               100e-6: 2, 300e-6: 3,
                               1e-3:   4, 3e-3:   5,
                               10e-3:  6, 30e-3:  7,
                               100e-3: 8, 300e-3: 9,
                               1:     10, 3:     11,
                               10:    12, 30:    13,
                               100:   14, 300:   15,
                               1e3:   16, 3e3:   17,
                               10e3:  18, 30e3:  19,
                           })

        self.add_parameter('filter_slope',
                           label='Filter slope',
                           get_cmd='OFSL?',
                           set_cmd='OFSL {}',
                           unit='dB/oct',
                           val_mapping={
                               6: 0,
                               12: 1,
                               18: 2,
                               24: 3,
                           })

        self.add_parameter('sync_filter',
                           label='Sync filter',
                           get_cmd='SYNC?',
                           set_cmd='SYNC {}',
                           val_mapping={
                               'off': 0,
                               'on': 1,
                           })

        def parse_offset_get(s: str) -> Tuple[float, int]:
            parts = s.split(',')

            return float(parts[0]), int(parts[1])

        # TODO: Parameters that can be set with multiple arguments
        # For the OEXP command for example two arguments are needed
        self.add_parameter('X_offset',
                           get_cmd='OEXP? 1',
                           get_parser=parse_offset_get)

        self.add_parameter('Y_offset',
                           get_cmd='OEXP? 2',
                           get_parser=parse_offset_get)

        self.add_parameter('R_offset',
                           get_cmd='OEXP? 3',
                           get_parser=parse_offset_get)

        # Aux input/output
        for i in [1, 2, 3, 4]:
            self.add_parameter(f'aux_in{i}',
                               label=f'Aux input {i}',
                               get_cmd=f'OAUX? {i}',
                               get_parser=float,
                               unit='V')

            self.add_parameter(f'aux_out{i}',
                               label=f'Aux output {i}',
                               get_cmd=f'AUXV? {i}',
                               get_parser=float,
                               set_cmd=f'AUXV {i}, {{}}',
                               unit='V')

        # Setup
        self.add_parameter('output_interface',
                           label='Output interface',
                           get_cmd='OUTX?',
                           set_cmd='OUTX {}',
                           val_mapping={
                               'RS232': '0\n',
                               'GPIB': '1\n',
                           })

        # Data transfer
        self.add_parameter('X',
                           get_cmd='OUTP? 1',
                           get_parser=float,
                           unit='V')

        self.add_parameter('Y',
                           get_cmd='OUTP? 2',
                           get_parser=float,
                           unit='V')

        self.add_parameter('R',
                           get_cmd='OUTP? 3',
                           get_parser=float,
                           unit='V')

        self.add_parameter('P',
                           get_cmd='OUTP? 4',
                           get_parser=float,
                           unit='deg')

        self.add_parameter(
            "complex_voltage",
            label="Voltage",
            get_cmd=self._get_complex_voltage,
            unit="V",
            docstring="Complex voltage parameter "
            "calculated from X, Y phase using "
            "Z = X +j*Y",
            vals=ComplexNumbers(),
        )

        # Data buffer settings
        self.add_parameter('buffer_SR',
                           label='Buffer sample rate',
                           get_cmd='SRAT ?',
                           set_cmd=self._set_buffer_SR,
                           unit='Hz',
                           val_mapping={62.5e-3: 0,
                                        0.125: 1,
                                        0.250: 2,
                                        0.5: 3,
                                        1: 4, 2: 5,
                                        4: 6, 8: 7,
                                        16: 8, 32: 9,
                                        64: 10, 128: 11,
                                        256: 12, 512: 13,
                                        'Trigger': 14},
                           get_parser=int
                           )

        self.add_parameter('buffer_acq_mode',
                           label='Buffer acquistion mode',
                           get_cmd='SEND ?',
                           set_cmd='SEND {}',
                           val_mapping={'single shot': 0,
                                        'loop': 1},
                           get_parser=int)

        self.add_parameter('buffer_trig_mode',
                           label='Buffer trigger start mode',
                           get_cmd='TSTR ?',
                           set_cmd='TSTR {}',
                           val_mapping={'ON': 1, 'OFF': 0},
                           get_parser=int)

        self.add_parameter('buffer_npts',
                           label='Buffer number of stored points',
                           get_cmd='SPTS ?',
                           get_parser=int)

        self.add_parameter('sweep_setpoints',
                           parameter_class=GeneratedSetPoints,
                           vals=Arrays(shape=(self.buffer_npts.get,)))

        # Channel setup
        for ch in range(1, 3):

            # detailed validation and mapping performed in set/get functions
            self.add_parameter(f'ch{ch}_ratio',
                               label=f'Channel {ch} ratio',
                               get_cmd=partial(self._get_ch_ratio, ch),
                               set_cmd=partial(self._set_ch_ratio, ch),
                               vals=Strings())
            self.add_parameter(f'ch{ch}_display',
                               label=f'Channel {ch} display',
                               get_cmd=partial(self._get_ch_display, ch),
                               set_cmd=partial(self._set_ch_display, ch),
                               vals=Strings())
            self.add_parameter(f'ch{ch}_databuffer',
                               channel=ch,
                               parameter_class=ChannelBuffer)
            self.add_parameter(f'ch{ch}_datatrace',
                               channel=ch,
                               vals=Arrays(shape=(self.buffer_npts.get,)),
                               setpoints=(self.sweep_setpoints,),
                               parameter_class=ChannelTrace)

        # Auto functions
        self.add_function('auto_gain', call_cmd='AGAN')
        self.add_function('auto_reserve', call_cmd='ARSV')
        self.add_function('auto_phase', call_cmd='APHS')
        self.add_function('auto_offset', call_cmd='AOFF {0}',
                          args=[Enum(1, 2, 3)])

        # Interface
        self.add_function('reset', call_cmd='*RST')

        self.add_function('disable_front_panel', call_cmd='OVRM 0')
        self.add_function('enable_front_panel', call_cmd='OVRM 1')

        self.add_function('send_trigger', call_cmd='TRIG',
                          docstring=("Send a software trigger. "
                                     "This command has the same effect as a "
                                     "trigger at the rear panel trigger"
                                     " input."))

        self.add_function('buffer_start', call_cmd='STRT',
                          docstring=("The buffer_start command starts or "
                                     "resumes data storage. buffer_start"
                                     " is ignored if storage is already in"
                                     " progress."))

        self.add_function('buffer_pause', call_cmd='PAUS',
                          docstring=("The buffer_pause command pauses data "
                                     "storage. If storage is already paused "
                                     "or reset then this command is ignored."))

        self.add_function('buffer_reset', call_cmd='REST',
                          docstring=("The buffer_reset command resets the data"
                                     " buffers. The buffer_reset command can "
                                     "be sent at any time - any storage in "
                                     "progress, paused or not, will be reset."
                                     " This command will erase the data "
                                     "buffer."))

        # Initialize the proper units of the outputs and sensitivities
        self.input_config()

        # start keeping track of buffer setpoints
        self._buffer1_ready = False
        self._buffer2_ready = False

        self.connect_message()


    SNAP_PARAMETERS = {
            'x': '1',
            'y': '2',
            'r': '3',
            'p': '4',
        'phase': '4',
            'θ': '4',
         'aux1': '5',
         'aux2': '6',
         'aux3': '7',
         'aux4': '8',
         'freq': '9',
          'ch1': '10',
          'ch2': '11'
    }

    def snap(self, *parameters: str) -> Tuple[float, ...]:
        """
        Get between 2 and 6 parameters at a single instant. This provides a
        coherent snapshot of measured signals. Pick up to 6 from: X, Y, R, θ,
        the aux inputs 1-4, frequency, or what is currently displayed on
        channels 1 and 2.

        Reading X and Y (or R and θ) gives a coherent snapshot of the signal.
        Snap is important when the time constant is very short, a time constant
        less than 100 ms.

        Args:
            *parameters: From 2 to 6 strings of names of parameters for which
                the values are requested. including: 'x', 'y', 'r', 'p',
                'phase' or 'θ', 'aux1', 'aux2', 'aux3', 'aux4', 'freq',
                'ch1', and 'ch2'.

        Returns:
            A tuple of floating point values in the same order as requested.

        Examples:
            >>> lockin.snap('x','y') -> tuple(x,y)

            >>> lockin.snap('aux1','aux2','freq','phase')
            >>> -> tuple(aux1,aux2,freq,phase)

        Note:
            Volts for x, y, r, and aux 1-4
            Degrees for θ
            Hertz for freq
            Unknown for ch1 and ch2. It will depend on what was set.

             - If X,Y,R and θ are all read, then the values of X,Y are recorded
               approximately 10 µs apart from R,θ. Thus, the values of X and Y
               may not yield the exact values of R and θ from a single snap.
             - The values of the Aux Inputs may have an uncertainty of
               up to 32 µs.
             - The frequency is computed only every other period or 40 ms,
               whichever is longer.
        """
        if not 2 <= len(parameters) <= 6:
            raise KeyError(
                'It is only possible to request values of 2 to 6 parameters'
                ' at a time.')

        for name in parameters:
            if name.lower() not in self.SNAP_PARAMETERS:
                raise KeyError(f'{name} is an unknown parameter. Refer'
                               f' to `SNAP_PARAMETERS` for a list of valid'
                               f' parameter names')

        p_ids = [self.SNAP_PARAMETERS[name.lower()] for name in parameters]
        output = self.ask(f'SNAP? {",".join(p_ids)}')

        return tuple(float(val) for val in output.split(','))

    def increment_sensitivity(self) -> bool:
        """
        Increment the sensitivity setting of the lock-in. This is equivalent
        to pushing the sensitivity up button on the front panel. This has no
        effect if the sensitivity is already at the maximum.

        Returns:
            Whether or not the sensitivity was actually changed.
        """
        return self._change_sensitivity(1)

    def decrement_sensitivity(self) -> bool:
        """
        Decrement the sensitivity setting of the lock-in. This is equivalent
        to pushing the sensitivity down button on the front panel. This has no
        effect if the sensitivity is already at the minimum.

        Returns:
            Whether or not the sensitivity was actually changed.
        """
        return self._change_sensitivity(-1)

    def _change_sensitivity(self, dn: int) -> bool:
        if self.input_config() in ['a', 'a-b']:
            n_to = self._N_TO_VOLT
            to_n = self._VOLT_TO_N
        else:
            n_to = self._N_TO_CURR
            to_n = self._CURR_TO_N

        n = to_n[self.sensitivity()]

        if n + dn > max(n_to.keys()) or n + dn < min(n_to.keys()):
            return False

        self.sensitivity.set(n_to[n + dn])
        return True

    def _set_buffer_SR(self, SR: int) -> None:
        self.write(f'SRAT {SR}')
        self._buffer1_ready = False
        self._buffer2_ready = False
        self.sweep_setpoints.update_units_if_constant_sample_rate()

    def _get_ch_ratio(self, channel: int) -> str:
        val_mapping = {1: {0: 'none',
                           1: 'Aux In 1',
                           2: 'Aux In 2'},
                       2: {0: 'none',
                           1: 'Aux In 3',
                           2: 'Aux In 4'}}
        resp = int(self.ask(f'DDEF ? {channel}').split(',')[1])

        return val_mapping[channel][resp]

    def _set_ch_ratio(self, channel: int, ratio: str) -> None:
        val_mapping = {1: {'none': 0,
                           'Aux In 1': 1,
                           'Aux In 2': 2},
                       2: {'none': 0,
                           'Aux In 3': 1,
                           'Aux In 4': 2}}
        vals = val_mapping[channel].keys()
        if ratio not in vals:
            raise ValueError(f'{ratio} not in {vals}')
        ratio_int = val_mapping[channel][ratio]
        disp_val = int(self.ask(f'DDEF ? {channel}').split(',')[0])
        self.write(f'DDEF {channel}, {disp_val}, {ratio_int}')
        self._buffer_ready = False

    def _get_ch_display(self, channel: int) -> str:
        val_mapping = {1: {0: 'X',
                           1: 'R',
                           2: 'X Noise',
                           3: 'Aux In 1',
                           4: 'Aux In 2'},
                       2: {0: 'Y',
                           1: 'Phase',
                           2: 'Y Noise',
                           3: 'Aux In 3',
                           4: 'Aux In 4'}}
        resp = int(self.ask(f'DDEF ? {channel}').split(',')[0])

        return val_mapping[channel][resp]

    def _set_ch_display(self, channel: int, disp: str) -> None:
        val_mapping = {1: {'X': 0,
                           'R': 1,
                           'X Noise': 2,
                           'Aux In 1': 3,
                           'Aux In 2': 4},
                       2: {'Y': 0,
                           'Phase': 1,
                           'Y Noise': 2,
                           'Aux In 3': 3,
                           'Aux In 4': 4}}
        vals = val_mapping[channel].keys()
        if disp not in vals:
            raise ValueError(f'{disp} not in {vals}')
        disp_int = val_mapping[channel][disp]
        # Since ratio AND display are set simultaneously,
        # we get and then re-set the current ratio value
        ratio_val = int(self.ask(f'DDEF ? {channel}').split(',')[1])
        self.write(f'DDEF {channel}, {disp_int}, {ratio_val}')
        self._buffer_ready = False
        # we update the unit of the datatrace
        # according to the choice of channel
        params = self.parameters
        dataparam = params[f'ch{channel}_datatrace']
        assert isinstance(dataparam, ChannelTrace)
        dataparam.update_unit()

    def _set_units(self, unit: str) -> None:
        # TODO:
        # make a public parameter function that allows to change the units
        for param in [self.X, self.Y, self.R, self.sensitivity]:
            param.unit = unit

    def _get_complex_voltage(self) -> complex:
        x, y = self.snap("X", "Y")
        return x + 1.0j * y

    def _get_input_config(self, s: int) -> str:
        mode = self._N_TO_INPUT_CONFIG[int(s)]

        if mode in ['a', 'a-b']:
            self.sensitivity.vals = self._VOLT_ENUM
            self._set_units('V')
        else:
            self.sensitivity.vals = self._CURR_ENUM
            self._set_units('A')

        return mode

    def _set_input_config(self, s: str) -> int:
        if s in ['a', 'a-b']:
            self.sensitivity.vals = self._VOLT_ENUM
            self._set_units('V')
        else:
            self.sensitivity.vals = self._CURR_ENUM
            self._set_units('A')

        return self._INPUT_CONFIG_TO_N[s]

    def _get_sensitivity(self, s: int) -> float:
        if self.input_config() in ['a', 'a-b']:
            return self._N_TO_VOLT[int(s)]
        else:
            return self._N_TO_CURR[int(s)]

    def _set_sensitivity(self, s: float) -> int:
        if self.input_config() in ['a', 'a-b']:
            return self._VOLT_TO_N[s]
        else:
            return self._CURR_TO_N[s]

    def autorange(self, max_changes: int = 1) -> None:
        """
        Automatically changes the sensitivity of the instrument according to
        the R value and defined max_changes.

        Args:
            max_changes: Maximum number of steps allowing the function to
                automatically change the sensitivity (default is 1). The actual
                number of steps needed to change to the optimal sensitivity may
                be more or less than this maximum.
        """
        def autorange_once() -> bool:
            r = self.R()
            sens = self.sensitivity()
            if r > 0.9 * sens:
                return self.increment_sensitivity()
            elif r < 0.1 * sens:
                return self.decrement_sensitivity()
            return False

        sets = 0
        while autorange_once() and sets < max_changes:
            sets += 1
            time.sleep(self.time_constant())

    def set_sweep_parameters(self,
                             sweep_param: Parameter,
                             start: float,
                             stop: float,
                             n_points: int = 10,
                             label: Union[str, None] = None) -> None:

        self.sweep_setpoints.sweep_array = np.linspace(start, stop, n_points)
        self.sweep_setpoints.unit = sweep_param.unit
        if label is not None:
            self.sweep_setpoints.label = label
        elif sweep_param.label is not None:
            self.sweep_setpoints.label = sweep_param.label


class GeneratedSetPoints(Parameter):
    """
    A parameter that generates a setpoint array from start, stop and num points
    parameters.
    """
    def __init__(self,
                 sweep_array: Iterable[Union[float, int]] = np.linspace(0, 1, 10),
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sweep_array = sweep_array
        self.update_units_if_constant_sample_rate()

    def update_units_if_constant_sample_rate(self) -> None:
        """
        If the buffer is filled at a constant sample rate,
        update the unit to "s" and label to "Time";
        otherwise do nothing
        """
        assert isinstance(self.root_instrument, SR830)
        SR = self.root_instrument.buffer_SR.get()
        if SR != 'Trigger':
            self.unit = 's'
            self.label = 'Time'

    def set_raw(self, value: Iterable[Union[float, int]]) -> None:
        self.sweep_array = value

    def get_raw(self) -> ParamRawDataType:
        assert isinstance(self.root_instrument, SR830)
        SR = self.root_instrument.buffer_SR.get()
        if SR == 'Trigger':
            return self.sweep_array
        else:
            N = self.root_instrument.buffer_npts.get()
            dt = 1/SR

            return np.linspace(0, N*dt, N)
