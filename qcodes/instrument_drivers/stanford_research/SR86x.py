import numpy as np
import logging
from typing import Sequence, Dict, Callable, Tuple

from qcodes import VisaInstrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils.validators import Numbers, Ints, Enum
from qcodes.instrument.parameter import ArrayParameter


log = logging.getLogger(__name__)


class SR86xBufferReadout(ArrayParameter):
    """
    The parameter array that holds read out data. We need this to be compatible
    with qcodes.Measure

    Args:
        name
        instrument
            This argument is unused, but needed because the add_parameter
            method of the Instrument base class adds this as a kwarg.
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
                         docstring='Holds an acquired (part of the) data '
                                   'buffer of one channel.')

        self.name = name
        self._capture_data = None

    def prepare_readout(self, capture_data: np.array) -> None:
        """
        Prepare this parameter for readout.

        Args:
            capture_data
        """
        self._capture_data = capture_data

        data_len = len(capture_data)
        self.shape = (data_len,)
        self.setpoint_units = ('',)
        self.setpoint_names = ('sample_nr',)
        self.setpoint_labels = ('Sample number',)
        self.setpoints = (tuple(np.arange(0, data_len)),)

    def get_raw(self) -> np.ndarray:
        """
        Public method to access the capture data
        """
        if self._capture_data is None:
            raise ValueError(f"Cannot return data for parameter {self.name}. "
                             f"Please prepare for readout by calling "
                             f"'get_capture_data' with appropriate "
                             f"configuration settings")

        return self._capture_data


class SR86xBuffer(InstrumentChannel):
    """
    The buffer module for the SR86x driver. This driver has been verified to
    work with the SR860 and SR865. For reference, please consult the SR860
    manual: http://thinksrs.com/downloads/PDFs/Manuals/SR860m.pdf
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
        self.bytes_per_sample = 4
        self.min_capture_length_in_kb = 1  # i.e. minimum buffer size
        self.max_capture_length_in_kb = 4096  # i.e. maximum buffer size
        # Maximum amount of kB that can be read per single CAPTUREGET command
        self.max_size_per_reading_in_kb = 64

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
            label="captured bytes",
            get_cmd="CAPTUREBYTES?",
            unit="B",
            get_parser=int,
            docstring="Number of bytes captured so far in the buffer. Can be "
                      "used to track live progress."
        )

        self.add_parameter(
            "count_capture_kilobytes",
            label="captured kilobytes",
            get_cmd="CAPTUREPROG?",
            unit="kB",
            docstring="Number of kilobytes captured so far in the buffer, "
                      "rounded-up to 2 kilobyte chunks. Capture must be "
                      "stopped before requesting the value of this "
                      "parameter. If the acquisition wrapped during operating "
                      "in Continuous mode, then the returned value is "
                      "simply equal to the current capture length."
        )

        for parameter_name in ["X", "Y", "R", "T"]:
            self.add_parameter(
                parameter_name,
                parameter_class=SR86xBufferReadout
            )

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

    def _set_capture_len_parser(self, capture_length_in_kb: int) -> int:
        """
        Parse the capture length in kB according to the way buffer treats it
        (refer to the manual for details). The given value has to fit in the
        range and has to be even, otherwise this function raises exceptions.

        Args:
            capture_length_in_kb: The desired capture length in kB.

        Returns:
            capture_length_in_kb
        """
        if capture_length_in_kb % 2:
            raise ValueError("The capture length should be an even number")

        if not self.min_capture_length_in_kb \
                <= capture_length_in_kb \
                <= self.max_capture_length_in_kb:
            raise ValueError(f"The capture length should be between "
                             f"{self.min_capture_length_in_kb} and "
                             f"{self.max_capture_length_in_kb}")

        return capture_length_in_kb

    def set_capture_rate_to_maximum(self) -> None:
        """
        Sets the capture rate to maximum. The maximum capture rate is
        retrieved from the device, and depends on the current value of the
        time constant.
        """
        self.capture_rate(self.capture_rate_max())

    def _set_capture_rate_parser(self, capture_rate_hz: float) -> int:
        """
        According to the manual, the capture rate query returns a value in
        Hz, but then setting this value it is expected to give a value n,
        where the capture rate in Hz is given by
        capture_rate_hz =  max_rate / 2 ** n. Please see page 136 of the
        manual. Here n is an integer in the range [0, 20].

        Args:
            capture_rate_hz
                The desired capture rate in Hz. If the desired rate is more
                than 1 Hz from the nearest valid rate, a warning is issued
                and the nearest valid rate it used.

        Returns:
            n_round
        """
        max_rate = self.capture_rate_max()
        n = np.log2(max_rate / capture_rate_hz)
        n_round = int(round(n))

        if not 0 <= n_round <= 20:
            raise ValueError(f"The chosen frequency is invalid. Please "
                             f"consult the SR860 manual at page 136. "
                             f"The maximum capture rate is {max_rate}")

        nearest_valid_rate = max_rate / 2 ** n_round
        if abs(capture_rate_hz - nearest_valid_rate) > 1:
            available_frequencies = ", ".join(
                [str(f) for f in self.available_frequencies])
            log.warning("Warning: Setting capture rate to {:.5} Hz"
                        .format(nearest_valid_rate))
            log.warning("The available frequencies are: {}"
                        .format(available_frequencies))

        return n_round

    def start_capture(self, acquisition_mode: str, trigger_mode: str) -> None:
        """
        Start an acquisition. Please see page 137 of the manual for a detailed
        explanation.

        Args:
            acquisition_mode
                "ONE" | "CONT"
            trigger_mode
                "IMM" | "TRIG" | "SAMP"
        """

        if acquisition_mode not in ["ONE", "CONT"]:
            raise ValueError(
                "The acquisition mode needs to be either 'ONE' or 'CONT'")

        if trigger_mode not in ["IMM", "TRIG", "SAMP"]:
            raise ValueError(
                "The trigger mode needs to be either 'IMM', 'TRIG' or 'SAMP'")

        cmd_str = f"CAPTURESTART {acquisition_mode}, {trigger_mode}"
        self.write(cmd_str)

    def stop_capture(self):
        """Stop a capture"""
        self.write("CAPTURESTOP")

    def _get_list_of_capture_variable_names(self):
        """
        Retrieve the list of names of variables (readouts) that are
        set to be captured
        """
        return self.capture_config().split(",")

    def _get_number_of_capture_variables(self):
        """
        Retrieve the number of variables (readouts) that are
        set to be captured
        """
        capture_variables = self._get_list_of_capture_variable_names()
        n_variables = len(capture_variables)
        return n_variables

    def _calc_capture_size_in_kb(self, sample_count: int) ->int:
        """
        Given the number of samples to capture, calculate the capture length
        that the buffer needs to be set to in order to fit the requested
        number of samples. Note that the number of activated readouts is
        taken into account.
        """
        n_variables = self._get_number_of_capture_variables()
        total_size_in_kb = int(
            np.ceil(n_variables * sample_count * self.bytes_per_sample / 1024)
        )
        # Make sure that the total size in kb is an even number, as expected by
        # the instrument
        if total_size_in_kb % 2:
            total_size_in_kb += 1
        return total_size_in_kb

    def set_capture_length_to_fit_samples(self, sample_count: int) -> None:
        """
        Set the capture length of the buffer to fit the given number of
        samples.

        Args:
            sample_count
                Number of samples that the buffer has to fit
        """
        total_size_in_kb = self._calc_capture_size_in_kb(sample_count)
        self.capture_length_in_kb(total_size_in_kb)

    def wait_until_samples_captured(self, sample_count: int) -> None:
        """
        Wait until the given number of samples is captured. This function
        is blocking and has to be used with caution because it does not have
        a timeout.

        Args:
            sample_count
                Number of samples that needs to be captured
        """
        n_captured_bytes = 0
        n_variables = self._get_number_of_capture_variables()
        n_bytes_to_capture = sample_count * n_variables * self.bytes_per_sample
        while n_captured_bytes < n_bytes_to_capture:
            n_captured_bytes = self.count_capture_bytes()

    def get_capture_data(self, sample_count: int) -> dict:
        """
        Read the given number of samples of the capture data from the buffer.

        Args:
            sample_count
                number of samples to read from the buffer

        Returns:
            data
                The keys in the dictionary correspond to the captured
                variables. For instance, if before the capture, the capture
                config was set as 'capture_config("X,Y")', then the keys will
                be "X" and "Y". The values in the dictionary are numpy arrays
                of numbers.
        """
        total_size_in_kb = self._calc_capture_size_in_kb(sample_count)
        capture_variables = self._get_list_of_capture_variable_names()
        n_variables = self._get_number_of_capture_variables()

        values = self._get_raw_capture_data(total_size_in_kb)

        # Remove zeros which mark the end part of the buffer that is not
        # filled with captured data
        values = values[values != 0]

        values = values.reshape((-1, n_variables)).T
        values = values[:, :sample_count]

        data = {k: v for k, v in zip(capture_variables, values)}

        for capture_variable in capture_variables:
            buffer_parameter = getattr(self, capture_variable)
            buffer_parameter.prepare_readout(data[capture_variable])

        return data

    def _get_raw_capture_data(self, size_in_kb: int) -> np.ndarray:
        """
        Read data from the buffer from its beginning avoiding the instrument
        limit of 64 kilobytes per reading.

        Args:
            size_in_kb
                Size of the data that needs to be read; if it exceeds the
                capture length, an exception is raised.

        Returns:
            A one-dimensional numpy array of the requested data. Note that the
            returned array contains data for all the variables that are
            mentioned in the capture config.
        """
        current_capture_length = self.capture_length_in_kb()
        if size_in_kb > current_capture_length:
            raise ValueError(f"The size of the requested data ({size_in_kb}kB) "
                             f"is larger than current capture length of the "
                             f"buffer ({current_capture_length}kB).")

        values = np.array([])
        data_size_to_read_in_kb = size_in_kb
        n_readings = 0

        while data_size_to_read_in_kb > 0:
            offset = n_readings * self.max_size_per_reading_in_kb

            if data_size_to_read_in_kb > self.max_size_per_reading_in_kb:
                size_of_this_reading = self.max_size_per_reading_in_kb
            else:
                size_of_this_reading = data_size_to_read_in_kb

            data_from_this_reading = self._get_raw_capture_data_block(
                size_of_this_reading,
                offset_in_kb=offset)
            values = np.append(values, data_from_this_reading)

            data_size_to_read_in_kb -= size_of_this_reading
            n_readings += 1

        return values

    def _get_raw_capture_data_block(self,
                                    size_in_kb: int,
                                    offset_in_kb: int=0
                                    ) -> np.ndarray:
        """
        Read data from the buffer. The maximum amount of data that can be
        read with this function (size_in_kb) is 64kB (this limitation comes
        from the instrument). The offset argument can be used to navigate
        along the buffer.

        An exception will be raised if either size_in_kb or offset_in_kb are
        longer that the *current* capture length (number of kB of data that is
        captured so far rounded up to 2kB chunks). If (offset_in_kb +
        size_in_kb) is longer than the *current* capture length,
        the instrument returns the wrapped data.

        For more information, refer to the description of the "CAPTUREGET"
        command in the manual.

        Args:
            size_in_kb
                Amount of data in kB that is to be read from the buffer
            offset_in_kb
                Offset within the buffer of where to read the data; for
                example, when 0 is specified, the data is read from the start
                of the buffer

        Returns:
            A one-dimensional numpy array of the requested data. Note that the
            returned array contains data for all the variables that are
            mentioned in the capture config.
        """
        if size_in_kb > self.max_size_per_reading_in_kb:
            raise ValueError(f"The size of the requested data ({size_in_kb}kB) "
                             f"is larger than maximum size that can be read "
                             f"at once ({self.max_size_per_reading_in_kb}kB).")

        # Calculate the size of the data captured so far, in kB, rounded up
        # to 2kB chunks
        size_of_currently_captured_data = int(
            np.ceil(np.ceil(self.count_capture_bytes() / 1024) / 2) * 2
        )

        if size_in_kb > size_of_currently_captured_data:
            raise ValueError(f"The size of the requested data ({size_in_kb}kB) "
                             f"cannot be larger than the size of currently "
                             f"captured data rounded up to 2kB chunks "
                             f"({size_of_currently_captured_data}kB)")

        if offset_in_kb > size_of_currently_captured_data:
            raise ValueError(f"The offset for reading the requested data "
                             f"({offset_in_kb}kB) cannot be larger than the "
                             f"size of currently captured data rounded up to "
                             f"2kB chunks "
                             f"({size_of_currently_captured_data}kB)")

        values = self._parent.visa_handle.query_binary_values(
            f"CAPTUREGET? {offset_in_kb}, {size_in_kb}",
            datatype='f',
            is_big_endian=False,
            expect_termination=False)
        # the sr86x does not include an extra termination char on binary
        # messages so we set expect_termination to False

        return np.array(values)

    def capture_one_sample_per_trigger(self,
                                       trigger_count: int,
                                       start_triggers_pulsetrain: Callable
                                       ) -> dict:
        """
        Capture one sample per each trigger, and return when the specified
        number of triggers has been received.

        Args:
            trigger_count
                Number of triggers to capture samples for
            start_triggers_pulsetrain
                By calling this *non-blocking* function, the train of trigger
                pulses should start

        Returns:
            data
                The keys in the dictionary correspond to the captured
                variables. For instance, if before the capture, the capture
                config was set as 'capture_config("X,Y")', then the keys will
                be "X" and "Y". The values in the dictionary are numpy arrays
                of numbers.
        """
        self.set_capture_length_to_fit_samples(trigger_count)
        self.start_capture("ONE", "SAMP")
        start_triggers_pulsetrain()
        self.wait_until_samples_captured(trigger_count)
        self.stop_capture()
        return self.get_capture_data(trigger_count)

    def capture_samples_after_trigger(self,
                                      sample_count: int,
                                      send_trigger: Callable
                                      ) -> dict:
        """
        Capture a number of samples after a trigger has been received.
        Please refer to page 135 of the manual for details.

        Args:
            sample_count
                Number of samples to capture
            send_trigger
                By calling this *non-blocking* function, one trigger should
                be sent that will initiate the capture

        Returns:
            data
                The keys in the dictionary correspond to the captured
                variables. For instance, if before the capture, the capture
                config was set as 'capture_config("X,Y")', then the keys will
                be "X" and "Y". The values in the dictionary are numpy arrays
                of numbers.
        """
        self.set_capture_length_to_fit_samples(sample_count)
        self.start_capture("ONE", "TRIG")
        send_trigger()
        self.wait_until_samples_captured(sample_count)
        self.stop_capture()
        return self.get_capture_data(sample_count)

    def capture_samples(self, sample_count: int) -> dict:
        """
        Capture a number of samples at a capture rate, starting immediately.
        Unlike the "continuous" capture mode, here the buffer does not get
        overwritten with the new data once the buffer is full.

        The function blocks until the required number of samples is acquired,
        and returns them.

        Args:
            sample_count
                Number of samples to capture

        Returns:
            data
                The keys in the dictionary correspond to the captured
                variables. For instance, if before the capture, the capture
                config was set as 'capture_config("X,Y")', then the keys will
                be "X" and "Y". The values in the dictionary are numpy arrays
                of numbers.
        """
        self.set_capture_length_to_fit_samples(sample_count)
        self.start_capture("ONE", "IMM")
        self.wait_until_samples_captured(sample_count)
        self.stop_capture()
        return self.get_capture_data(sample_count)


class SR86xDataChannel(InstrumentChannel):
    """
    Implements a data channel of SR86x lock-in amplifier. Parameters that are
    assigned to these channels get plotted on the display of the instrument.
    Moreover, there are commands that allow to conveniently retrieve the values
    of the parameters that are currently assigned to the data channels.

    This class relies on the available parameter names that should be
    mentioned in the lock-in amplifier class in `PARAMETER_NAMES` attribute.

    Args:
        parent
            an instance of SR86x driver
        name
            data channel name that is to be used to refernce it from the parent
        cmd_id
            this ID is used in VISA commands to refer to this data channel,
            usually is an integer number
        channel_name
            this name can also be used in VISA commands along with
            channel_id; it is not used in this implementation, but is added
            for reference
        channel_color
            every data channel is also referred to by the color with which it
            is being plotted on the instrument's screen; added here only for
            reference
    """
    def __init__(self, parent: 'SR86x', name: str, cmd_id: str,
                 cmd_id_name: str=None, color: str=None) -> None:
        super().__init__(parent, name)

        self._cmd_id = cmd_id
        self._cmd_id_name = cmd_id_name
        self._color = color

        self.add_parameter(f'assigned_parameter',
                           label=f'Data channel {cmd_id} parameter',
                           docstring=f'Allows to set and get the '
                                     f'parameter that is assigned to data '
                                     f'channel {cmd_id}',
                           set_cmd=f'CDSP {cmd_id}, {{}}',
                           get_cmd=f'CDSP? {cmd_id}',
                           val_mapping=self.parent.PARAMETER_NAMES
                           )

    @property
    def cmd_id(self):
        return self._cmd_id

    @property
    def cmd_id_name(self):
        return self._cmd_id_name

    @property
    def color(self):
        return self._color


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

    PARAMETER_NAMES = {
                'X': '0',   # X output, 'X'
                'Y': '1',   # Y output, 'Y'
                'R': '2',   # R output, 'R'
                'P': '3',   # theta output, 'THeta'
          'aux_in1': '4',   # Aux In 1, 'IN1'
          'aux_in2': '5',   # Aux In 2, 'IN2'
          'aux_in3': '6',   # Aux In 3, 'IN3'
          'aux_in4': '7',   # Aux In 4, 'IN4'
           'Xnoise': '8',   # X noise, 'XNOise'
           'Ynoise': '9',   # Y noise, 'YNOise'
         'aux_out1': '10',  # Aux Out 1, 'OUT1'
         'aux_out2': '11',  # Aux Out 2, 'OUT2'
            'phase': '12',  # Reference Phase, 'PHAse'
        'amplitude': '13',  # Sine Out Amplitude, 'SAMp'
       'sine_outdc': '14',  # DC Level, 'LEVel'
        'frequency': '15',  # Int. Ref. Frequency, 'FInt'
    'frequency_ext': '16',  # Ext. Ref. Frequency, 'FExt'
    }

    _N_DATA_CHANNELS = 4

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
                           vals=Numbers(
                               min_value=1e-3,
                               max_value=self._max_frequency)
                           )
        self.add_parameter(name='sine_outdc',
                           label='Sine out dc level',
                           unit='V',
                           get_cmd='SOFF?',
                           set_cmd='SOFF {}',
                           get_parser=float,
                           vals=Numbers(min_value=-5, max_value=5))
        self.add_parameter(name='amplitude',
                           label='Amplitude',
                           unit='V',
                           get_cmd='SLVL?',
                           set_cmd='SLVL {}',
                           get_parser=float,
                           vals=Numbers(min_value=0, max_value=2))
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

        self.add_parameter(
            name="external_reference_trigger",
            label="External reference trigger mode",
            get_cmd="RTRG?",
            set_cmd="RTRG {}",
            val_mapping={
                "SIN": 0,
                "POS": 1,
                "POSTTL": 1,
                "NEG": 2,
                "NEGTTL": 2,
            },
            docstring="The triggering mode for synchronization of the "
                      "internal reference signal with the externally provided "
                      "one"
        )

        self.add_parameter(
            name="reference_source",
            label="Reference source",
            get_cmd="RSRC?",
            set_cmd="RSRC {}",
            val_mapping={
                "INT": 0,
                "EXT": 1,
                "DUAL": 2,
                "CHOP": 3
            },
            docstring="The source of the reference signal"
        )

        self.add_parameter(
            name="external_reference_trigger_input_resistance",
            label="External reference trigger input resistance",
            get_cmd="REFZ?",
            set_cmd="REFZ {}",
            val_mapping={
                "50": 0,
                "50OHMS": 0,
                0: 0,
                "1M": 1,
                "1MEG": 1,
                1: 1,
            },
            docstring="Input resistance of the input for the external "
                      "reference signal"
        )

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

        # Data channels:
        # 'DAT1' (green), 'DAT2' (blue), 'DAT3' (yellow), 'DAT4' (orange)
        data_channels = ChannelList(self, "data_channels", SR86xDataChannel,
                                    snapshotable=False)
        for num, color in zip(range(self._N_DATA_CHANNELS),
                              ('green', 'blue', 'yellow', 'orange')):
            cmd_id = f"{num}"
            cmd_id_name = f"DAT{num + 1}"
            ch_name = f"data_channel_{num + 1}"

            data_channel = SR86xDataChannel(
                self, ch_name, cmd_id, cmd_id_name, color)

            data_channels.append(data_channel)
            self.add_submodule(ch_name, data_channel)

        data_channels.lock()
        self.add_submodule("data_channels", data_channels)

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

    def get_values(self, *parameter_names: str) -> Tuple[float, ...]:
        """
        Get values of 2 or 3 parameters that are measured by the lock-in
        amplifier. These values are guaranteed to come from the same
        measurement cycle as opposed to getting values of parameters one by
        one (for example, by calling `sr.X()`, and then `sr.Y()`.

        Args:
            *parameter_names
                2 or 3 names of parameters for which the values are
                requested; valid names can be found in `PARAMETER_NAMES`
                attribute of the driver class

        Returns:
            a tuple of 2 or 3 floating point values

        """
        if not 2 <= len(parameter_names) <= 3:
            raise KeyError(
                'It is only possible to request values of 2 or 3 parameters '
                'at a time.')

        for name in parameter_names:
            if name not in self.PARAMETER_NAMES:
                raise KeyError(f'{name} is not a valid parameter name. Refer '
                               f'to `PARAMETER_NAMES` for a list of valid '
                               f'parameter names')

        p_ids = [self.PARAMETER_NAMES[name] for name in parameter_names]
        output = self.ask(f'SNAP? {",".join(p_ids)}')
        return tuple(float(val) for val in output.split(','))

    def get_data_channels_values(self) -> Tuple[float, ...]:
        """
        Queries the current values of the data channels

        Returns:
            tuple of 4 values of the data channels
        """
        output = self.ask('SNAPD?')
        return tuple(float(val) for val in output.split(','))

    def get_data_channels_parameters(self, query_instrument: bool=True
                                     ) -> Tuple[str, ...]:
        """
        Convenience method to query a list of parameters which the data
        channels are currently assigned to.

        Args:
            query_instrument
                If set to False, the internally cashed names of the parameters
                will be returned; if True, then the names will be queried
                through the instrument

        Returns:
            a tuple of 4 strings of parameter names
        """
        if query_instrument:
            method_name = 'get'
        else:
            method_name = 'get_latest'

        return tuple(
            getattr(getattr(self.data_channels[i], 'assigned_parameter'),
                    method_name)()
            for i in range(self._N_DATA_CHANNELS)
        )

    def get_data_channels_dict(self, requery_names: bool=False
                               ) -> Dict[str, float]:
        """
        Returns a dictionary where the keys are parameter names currently
        assigned to the data channels, and values are the values of those
        parameters.

        Args:
            requery_names
                if False, the currently assigned parameter names will not be
                queries from the instrument in order to save time on
                communication, in this case the cached assigned parameter
                names will be used for the keys of the dicitonary; if True,
                the assigned parameter names will be queried from the
                instrument

        Returns:
            a dictionary where keys are names of parameters assigned to the
            data channels, and values are the values of those parameters
        """
        parameter_names = self.get_data_channels_parameters(requery_names)
        parameter_values = self.get_data_channels_values()
        return dict(zip(parameter_names, parameter_values))
