import numpy as np
import time
import logging
from typing import Optional, Sequence, Dict

from qcodes import VisaInstrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes.utils.validators import Numbers, Ints, Enum
from qcodes.instrument.parameter import ArrayParameter

log = logging.getLogger(__name__)


class SR86xBufferReadout(ArrayParameter):
    """
    The parameter array that holds read out data. We need this to be compatible with qcodes.Measure

    Args
    ----
        name (str)
        instrument (SR86x): This argument is unused, but needed because the add_parameter method of the Instrument
                            base class adds this as a kwarg.
    """
    def __init__(self, name: str, instrument: 'SR86x') ->None:

        unit = "deg"
        if name in ["X", "Y", "R"]:
            unit = "V"

        super().__init__(name,
                         shape=(1,),  # dummy initial shape
                         unit=unit,
                         setpoint_names=('Time',),
                         setpoint_labels=('Time',),
                         setpoint_units=('s',),
                         docstring='Holds an acquired (part of the) data buffer of one channel.')

        self.name = name
        self._capture_data = None

    def prepare_readout(self, capture_data: np.array) ->None:
        """
        Prepare this parameter for readout.

        Args
        ----
        capture_data (np.array)
        """
        self._capture_data = capture_data

        data_len = len(capture_data)
        self.shape = (data_len,)
        self.setpoint_units = ('',)
        self.setpoint_names = ('sample_nr',)
        self.setpoint_labels = ('Sample number',)
        self.setpoints = (tuple(np.arange(0, data_len)),)

    def get_raw(self) ->np.ndarray:
        """
        Public method to access the capture data
        """
        if self._capture_data is None:
            err_str = "Cannot return data for parameter {}. Please prepare for ".format(self.name)
            err_str = err_str + "readout by calling 'get_capture_data' with appropriate configuration settings"
            raise ValueError(err_str)

        return self._capture_data


class SR86xBuffer(InstrumentChannel):
    """
    The buffer module for the SR86x driver. This driver has been verified to work with the SR860 and SR865.
    For reference, please consult the SR860 manual: http://thinksrs.com/downloads/PDFs/Manuals/SR860m.pdf
    """

    def __init__(self, parent: 'SR86x', name: str) ->None:
        super().__init__(parent, name)
        self._parent = parent

        self.add_parameter(
            "capture_length_in_kb",
            label="get/set capture length",
            get_cmd="CAPTURELEN?",
            set_cmd="CAPTURELEN {}",
            set_parser=self._set_capture_len_parser,
            get_parser=int,
            unit="kB"
        )

        self.add_parameter(  # Configure which parameters we want to capture
            "capture_config",
            label="capture configuration",
            get_cmd="CAPTURECFG?",
            set_cmd="CAPTURECFG {}",
            val_mapping={"X": "0", "X,Y": "1", "R,T": "2", "X,Y,R,T": "3"}
        )

        self.add_parameter(
            "capture_rate_max",
            label="capture rate maximum",
            get_cmd="CAPTURERATEMAX?",
            get_parser=float
        )

        self.add_parameter(
            "capture_rate",
            label="capture rate raw",
            get_cmd="CAPTURERATE?",
            set_cmd="CAPTURERATE {}",
            get_parser=float,
            set_parser=self._set_capture_rate_parser
        )

        max_rate = self.capture_rate_max()
        self.available_frequencies = [max_rate / 2 ** i for i in range(20)]

        self.add_parameter(  # Are we capturing at the moment?
            "capture_status",
            label="capture status",
            get_cmd="CAPTURESTAT?"
        )

        self.add_parameter(
            "count_capture_bytes",
            label="capture bytes",
            get_cmd="CAPTUREBYTES?",
            unit="B"
        )

        self.add_parameter(
            "count_capture_kilobytes",
            label="capture kilobytes",
            get_cmd="CAPTUREPROG?",
            unit="kB"
        )

        for parameter_name in ["X", "Y", "R", "T"]:
            self.add_parameter(
                parameter_name,
                parameter_class=SR86xBufferReadout
            )

        self.bytes_per_sample = 4

    def snapshot_base(self, update: bool = False,
                      params_to_skip_update: Sequence[str] = None) -> Dict:
        if params_to_skip_update is None:
            params_to_skip_update = []
        # we omit count_capture_kilobytes from the snapshot because
        # it can only be read after a completed capture and will
        # timeout otherwise when the snapshot is updated, e.g. at
        # station creation time
        params_to_skip_update = list(params_to_skip_update)
        params_to_skip_update.append('count_capture_kilobytes')

        snapshot = super().snapshot_base(update, params_to_skip_update)
        return snapshot

    @staticmethod
    def _set_capture_len_parser(value: int) -> int:

        if value % 2:
            log.warning("the capture length needs to be even. Setting to {}".format(value + 1))
            value += 1

        if not 1 <= value <= 4096:
            raise ValueError("the capture length should be between 1 and 4096")

        return value

    def _set_capture_rate_parser(self, capture_rate_hz: float) -> int:
        """
        According to the manual, the capture rate query returns a value in Hz, but then setting this value it is
        expected to give a value n, where the capture rate in Hz is given by capture_rate_hz =  max_rate / 2 ** n.
        Please see page 136 of the manual. Here n is an integer in the range [0, 20]

        Args:
            capture_rate_hz (float): The desired capture rate in Hz. If the desired rate is more then 1 Hz from the
                                        nearest valid rate, a warning is issued and the nearest valid rate it used.

        Returns:
            n_round (int)
        """
        max_rate = self.capture_rate_max()
        n = np.log2(max_rate / capture_rate_hz)
        n_round = int(round(n))

        if not 0 <= n_round <= 20:
            raise ValueError("The chosen frequency is invalid. Please consult the SR860 manual at page 136. The maximum"
                             " capture rate is {}".format(max_rate))

        nearest_valid_rate = max_rate / 2 ** n_round
        if abs(capture_rate_hz - nearest_valid_rate) > 1:
            log.warning("Warning: Setting capture rate to {:.5} Hz".format(nearest_valid_rate))
            available_frequencies = ", ".join([str(f) for f in self.available_frequencies])
            log.warning("The available frequencies are: {}".format(available_frequencies))

        return n_round

    def start_capture(self, acquisition_mode: str, trigger_mode: str) -> None:
        """
        Start an acquisition. Please see page 137 of the manual for a detailed explanation.

        Args:
            acquisition_mode (str):  "ONE" | "CONT".
            trigger_mode (str): IMM | TRIG | SAMP
        """

        if acquisition_mode not in ["ONE", "CONT"]:
            raise ValueError("The acquisition mode needs to be either 'ONE' or 'CONT'")

        if trigger_mode not in ["IMM", "TRIG", "SAMP"]:
            raise ValueError("The trigger mode needs to be either 'IMM', 'TRIG' or 'SAMP'")

        cmd_str = "CAPTURESTART  {},{}".format(acquisition_mode, trigger_mode)
        self.write(cmd_str)

    def stop_capture(self):
        """
        Stop a capture
        """
        self.write("CAPTURESTOP")

    def get_capture_data(self, sample_count: int) -> dict:
        """
        Read capture data from the buffer.

        Args:
             sample_count (int): number of samples to read from the buffer

        Returns:
            data (dict): The keys in the dictionary is the variables we have captures. For instance, if before the
                            capture we specify 'capture_config("X,Y")', then the keys will be "X" and "Y".
        """
        capture_variables = self.capture_config().split(",")
        n_variables = len(capture_variables)

        total_size_in_kb = int(np.ceil(n_variables * sample_count * self.bytes_per_sample / 1024))
        # We will samples one kb more then strictly speaking required and trim the data length to the requested
        # sample count. In this way, we are always sure that the user requested number of samples is returned.
        total_size_in_kb += 1

        if total_size_in_kb > 64:
            raise ValueError("Number of samples specified is larger then the buffer size")

        values = self._parent.visa_handle.query_binary_values("CAPTUREGET? 0,{}".format(total_size_in_kb),
                                                              datatype='f', is_big_endian=False)
        values = np.array(values)
        values = values[values != 0]

        data = {k: v for k, v in zip(capture_variables, values.reshape((-1, n_variables)).T)}

        for capture_variable in capture_variables:
            buffer_parameter = getattr(self, capture_variable)
            buffer_parameter.prepare_readout(data[capture_variable])

        return data

    def capture_samples(self, sample_count: int) ->dict:
        """
        Capture a number of samples. This convenience function provides an example how we use the start and stop
        methods. We acquire the samples by sleeping for a time and then reading the buffer.

        Args:
            sample_count (int)

        Returns:
            dict
        """
        capture_rate = self.capture_rate()
        capture_time = sample_count / capture_rate

        self.start_capture("CONT", "IMM")
        time.sleep(capture_time)
        self.stop_capture()

        return self.get_capture_data(sample_count)


class SR86x(VisaInstrument):
    """
    This is the code for Stanford_SR865 Lock-in Amplifier
    """
    _VOLT_TO_N = {1: 0, 500e-3: 1, 200e-3: 2,
                  100e-3: 3, 50e-3: 4, 20e-3: 5,
                  10e-3: 6, 5e-3: 7, 2e-3: 8,
                  1e-3: 9, 500e-6: 10, 200e-6: 11,
                  100e-6: 12, 50e-6: 13, 20e-6: 14,
                  10e-6: 15, 5e-6: 16, 2e-6: 17,
                  1e-6: 18, 500e-9: 19, 200e-9: 20,
                  100e-9: 21, 50e-9: 22, 20e-9: 23,
                  10e-9: 24, 5e-9: 25, 2e-9: 26,
                  1e-9: 27}
    _N_TO_VOLT = {v: k for k, v in _VOLT_TO_N.items()}

    _CURR_TO_N = {1e-6: 0, 500e-9: 1, 200e-9: 2,
                  100e-9: 3, 50e-9: 4, 20e-9: 5,
                  10e-9: 6, 5e-9: 7, 2e-9: 8,
                  1e-9: 9, 500e-12: 10, 200e-12: 11,
                  100e-12: 12, 50e-12: 13, 20e-12: 14,
                  10e-12: 15, 5e-12: 16, 2e-12: 17,
                  1e-12: 18, 500e-15: 19, 200e-15: 20,
                  100e-15: 21, 50e-15: 22, 20e-15: 23,
                  10e-15: 24, 5e-15: 25, 2e-15: 26,
                  1e-15: 27}
    _N_TO_CURR = {v: k for k, v in _CURR_TO_N.items()}

    _VOLT_ENUM = Enum(*_VOLT_TO_N.keys())
    _CURR_ENUM = Enum(*_CURR_TO_N.keys())

    _INPUT_SIGNAL_TO_N = {
        'voltage': 0,
        'current': 1,
    }
    _N_TO_INPUT_SIGNAL = {v: k for k, v in _INPUT_SIGNAL_TO_N.items()}

    def __init__(self, name, address, max_frequency, reset=False, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)
        self._max_frequency = max_frequency
        # Reference commands
        self.add_parameter(name='frequency',
                           label='Frequency',
                           unit='Hz',
                           get_cmd='FREQ?',
                           set_cmd='FREQ {}',
                           get_parser=float,
                           vals=Numbers(min_value=1e-3, max_value=self._max_frequency))
        self.add_parameter(name='sine_outdc',
                           label='Sine out dc level',
                           unit='V',
                           get_cmd='SOFF?',
                           set_cmd='SOFF {:.3f}',
                           get_parser=float,
                           vals=Numbers(min_value=-5, max_value=5))
        self.add_parameter(name='amplitude',
                           label='Amplitude',
                           unit='V',
                           get_cmd='SLVL?',
                           set_cmd='SLVL {}',
                           get_parser=float,
                           vals=Numbers(min_value=10e-9, max_value=2))
        self.add_parameter(name='harmonic',
                           label='Harmonic',
                           get_cmd='HARM?',
                           get_parser=int,
                           set_cmd='HARM {:d}',
                           vals=Ints(min_value=1, max_value=99))
        self.add_parameter(name='phase',
                           label='Phase',
                           unit='deg',
                           get_cmd='PHAS?',
                           set_cmd='PHAS {}',
                           get_parser=float,
                           vals=Numbers(min_value=-3.6e5, max_value=3.6e5))
        # Signal commands
        self.add_parameter(name='sensitivity',
                           label='Sensitivity',
                           get_cmd='SCAL?',
                           set_cmd='SCAL {:d}',
                           get_parser=self._get_sensitivity,
                           set_parser=self._set_sensitivity
                           )
        self.add_parameter(name='filter_slope',
                           label='Filter slope',
                           unit='dB/oct',
                           get_cmd='OFSL?',
                           set_cmd='OFSL {}',
                           val_mapping={6: 0,
                                        12: 1,
                                        18: 2,
                                        24: 3})
        self.add_parameter(name='sync_filter',
                           label='Sync filter',
                           get_cmd='SYNC?',
                           set_cmd='SYNC {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1})
        self.add_parameter(name='noise_bandwidth',
                           label='Noise bandwidth',
                           unit='Hz',
                           get_cmd='ENBW?',
                           get_parser=float)
        self.add_parameter(name='signal_strength',
                           label='Signal strength indicator',
                           get_cmd='ILVL?',
                           get_parser=int)
        self.add_parameter(name='signal_input',
                           label='Signal input',
                           get_cmd='IVMD?',
                           get_parser=self._get_input_config,
                           set_cmd='IVMD {}',
                           set_parser=self._set_input_config,
                           vals=Enum(*self._INPUT_SIGNAL_TO_N.keys()))
        self.add_parameter(name='input_range',
                           label='Input range',
                           unit='V',
                           get_cmd='IRNG?',
                           set_cmd='IRNG {}',
                           val_mapping={1: 0,
                                        300e-3: 1,
                                        100e-3: 2,
                                        30e-3: 3,
                                        10e-3: 4})
        self.add_parameter(name='input_config',
                           label='Input configuration',
                           get_cmd='ISRC?',
                           set_cmd='ISRC {}',
                           val_mapping={'a': 0,
                                        'a-b': 1})
        self.add_parameter(name='input_shield',
                           label='Input shield',
                           get_cmd='IGND?',
                           set_cmd='IGND {}',
                           val_mapping={'float': 0,
                                        'ground': 1})
        self.add_parameter(name='input_gain',
                           label='Input gain',
                           unit='ohm',
                           get_cmd='ICUR?',
                           set_cmd='ICUR {}',
                           val_mapping={1e6: 0,
                                        100e6: 1})
        self.add_parameter(name='adv_filter',
                           label='Advanced filter',
                           get_cmd='ADVFILT?',
                           set_cmd='ADVFILT {}',
                           val_mapping={'OFF': 0,
                                        'ON': 1})
        self.add_parameter(name='input_coupling',
                           label='Input coupling',
                           get_cmd='ICPL?',
                           set_cmd='ICPL {}',
                           val_mapping={'ac': 0, 'dc': 1})
        self.add_parameter(name='time_constant',
                           label='Time constant',
                           unit='s',
                           get_cmd='OFLT?',
                           set_cmd='OFLT {}',
                           val_mapping={1e-6: 0, 3e-6: 1,
                                        10e-6: 2, 30e-6: 3,
                                        100e-6: 4, 300e-6: 5,
                                        1e-3: 6, 3e-3: 7,
                                        10e-3: 8, 30e-3: 9,
                                        100e-3: 10, 300e-3: 11,
                                        1: 12, 3: 13,
                                        10: 14, 30: 15,
                                        100: 16, 300: 17,
                                        1e3: 18, 3e3: 19,
                                        10e3: 20, 30e3: 21})
        # Auto functions
        self.add_function('auto_range', call_cmd='ARNG')
        self.add_function('auto_scale', call_cmd='ASCL')
        self.add_function('auto_phase', call_cmd='APHS')

        # Data transfer
        # first 4 parameters from a list of 16 below.
        self.add_parameter('X',
                           label='In-phase Magnitude',
                           get_cmd='OUTP? 0',
                           get_parser=float,
                           unit='V')
        self.add_parameter('Y',
                           label='Out-phase Magnitude',
                           get_cmd='OUTP? 1',
                           get_parser=float,
                           unit='V')
        self.add_parameter('R',
                           label='Magnitude',
                           get_cmd='OUTP? 2',
                           get_parser=float,
                           unit='V')
        self.add_parameter('P',
                           label='Phase',
                           get_cmd='OUTP? 3',
                           get_parser=float,
                           unit='deg')

        # CH1/CH2 Output Commands
        self.add_parameter('X_offset',
                           label='X offset ',
                           unit='%',
                           get_cmd='COFP? 0',
                           set_cmd='COFP 0, {}',
                           get_parser=float,
                           vals=Numbers(min_value=-999.99, max_value=999.99))
        self.add_parameter('Y_offset',
                           label='Y offset',
                           unit='%',
                           get_cmd='COFP? 1',
                           set_cmd='COFP 1, {}',
                           get_parser=float,
                           vals=Numbers(min_value=-999.99, max_value=999.99))
        self.add_parameter('R_offset',
                           label='R offset',
                           unit='%',
                           get_cmd='COFP? 2',
                           set_cmd='COFP 2, {}',
                           get_parser=float,
                           vals=Numbers(min_value=-999.99, max_value=999.99))
        self.add_parameter('X_expand',
                           label='X expand multiplier',
                           get_cmd='CEXP? 0',
                           set_cmd='CEXP 0, {}',
                           val_mapping={'OFF': '0',
                                        'X10': '1',
                                        'X100': '2'})
        self.add_parameter('Y_expand',
                           label='Y expand multiplier',
                           get_cmd='CEXP? 1',
                           set_cmd='CEXP 1, {}',
                           val_mapping={'OFF': 0,
                                        'X10': 1,
                                        'X100': 2})
        self.add_parameter('R_expand',
                           label='R expand multiplier',
                           get_cmd='CEXP? 2',
                           set_cmd='CEXP 2, {}',
                           val_mapping={'OFF': 0,
                                        'X10': 1,
                                        'X100': 2})
        # Aux input/output
        for i in [0, 1, 2, 3]:
            self.add_parameter('aux_in{}'.format(i),
                               label='Aux input {}'.format(i),
                               get_cmd='OAUX? {}'.format(i),
                               get_parser=float,
                               unit='V')

            self.add_parameter('aux_out{}'.format(i),
                               label='Aux output {}'.format(i),
                               get_cmd='AUXV? {}'.format(i),
                               get_parser=float,
                               set_cmd='AUXV {0}, {{}}'.format(i),
                               unit='V')

        # Interface
        self.add_function('reset', call_cmd='*RST')

        self.add_function('disable_front_panel', call_cmd='OVRM 0')
        self.add_function('enable_front_panel', call_cmd='OVRM 1')

        buffer = SR86xBuffer(self, "{}_buffer".format(self.name))
        self.add_submodule("buffer", buffer)

        self.input_config()
        self.connect_message()

    def _set_units(self, unit):
        for param in [self.X, self.Y, self.R, self.sensitivity]:
            param.unit = unit

    def _get_input_config(self, s):
        mode = self._N_TO_INPUT_SIGNAL[int(s)]

        if mode == 'voltage':
            self.sensitivity.vals = self._VOLT_ENUM
            self._set_units('V')
        else:
            self.sensitivity.vals = self._CURR_ENUM
            self._set_units('A')

        return mode

    def _set_input_config(self, s):
        if s == 'voltage':
            self.sensitivity.vals = self._VOLT_ENUM
            self._set_units('V')
        else:
            self.sensitivity.vals = self._CURR_ENUM
            self._set_units('A')

        return self._INPUT_SIGNAL_TO_N[s]

    def _get_sensitivity(self, s):
        if self.signal_input() == 'voltage':
            return self._N_TO_VOLT[int(s)]
        else:
            return self._N_TO_CURR[int(s)]

    def _set_sensitivity(self, s):
        if self.signal_input() == 'voltage':
            return self._VOLT_TO_N[s]
        else:
            return self._CURR_TO_N[s]
