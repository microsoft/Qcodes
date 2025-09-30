from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from qcodes.instrument import (
    InstrumentBaseKWArgs,
    InstrumentChannel,
)
from qcodes.parameters import ArrayParameter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from typing_extensions import Unpack

    from qcodes.parameters import Parameter

    from .SR86x import SR86x

log = logging.getLogger(__name__)


class SR86xBufferReadout(ArrayParameter):
    """
    The parameter array that holds read out data. We need this to be compatible
    with qcodes.Measure

    Args:
        name: Name of the parameter.
        instrument: The instrument to add this parameter to.

    """

    def __init__(self, name: str, instrument: SR86x, **kwargs: Any) -> None:
        unit = "deg"
        if name in ["X", "Y", "R"]:
            unit = "V"

        super().__init__(
            name,
            shape=(1,),  # dummy initial shape
            unit=unit,
            setpoint_names=("Time",),
            setpoint_labels=("Time",),
            setpoint_units=("s",),
            instrument=instrument,
            docstring="Holds an acquired (part of the) data buffer of one channel.",
            **kwargs,
        )

        self._capture_data: npt.NDArray | None = None

    def prepare_readout(self, capture_data: npt.NDArray) -> None:
        """
        Prepare this parameter for readout.

        Args:
            capture_data: The data to capture.

        """
        self._capture_data = capture_data

        data_len = len(capture_data)
        self.shape = (data_len,)
        self.setpoint_units = ("",)
        self.setpoint_names = ("sample_nr",)
        self.setpoint_labels = ("Sample number",)
        self.setpoints = (tuple(np.arange(0, data_len)),)

    def get_raw(self) -> npt.NDArray:
        """
        Public method to access the capture data
        """
        if self._capture_data is None:
            raise ValueError(
                f"Cannot return data for parameter {self.name}. "
                f"Please prepare for readout by calling "
                f"'get_capture_data' with appropriate "
                f"configuration settings"
            )

        return self._capture_data


class SR86xBuffer(InstrumentChannel):
    """
    Buffer module for the SR86x drivers.

    This driver has been verified to work with the SR860 and SR865.
    For reference, please consult the SR860
    manual: http://thinksrs.com/downloads/PDFs/Manuals/SR860m.pdf
    """

    def __init__(
        self, parent: SR86x, name: str, **kwargs: Unpack[InstrumentBaseKWArgs]
    ) -> None:
        super().__init__(parent, name, **kwargs)

        self.capture_length_in_kb: Parameter = self.add_parameter(
            "capture_length_in_kb",
            label="get/set capture length",
            get_cmd="CAPTURELEN?",
            set_cmd="CAPTURELEN {}",
            set_parser=self._set_capture_len_parser,
            get_parser=int,
            unit="kB",
        )
        """Parameter capture_length_in_kb"""
        self.bytes_per_sample = 4
        self.min_capture_length_in_kb = 1  # i.e. minimum buffer size
        self.max_capture_length_in_kb = 4096  # i.e. maximum buffer size
        # Maximum amount of kB that can be read per single CAPTUREGET command
        self.max_size_per_reading_in_kb = 64

        self.capture_config: Parameter = (
            self.add_parameter(  # Configure which parameters we want to capture
                "capture_config",
                label="capture configuration",
                get_cmd="CAPTURECFG?",
                set_cmd="CAPTURECFG {}",
                val_mapping={"X": "0", "X,Y": "1", "R,T": "2", "X,Y,R,T": "3"},
            )
        )
        """Parameter capture_config"""

        self.capture_rate_max: Parameter = self.add_parameter(
            "capture_rate_max",
            label="capture rate maximum",
            get_cmd="CAPTURERATEMAX?",
            get_parser=float,
        )
        """Parameter capture_rate_max"""

        self.capture_rate: Parameter = self.add_parameter(
            "capture_rate",
            label="capture rate raw",
            get_cmd="CAPTURERATE?",
            set_cmd="CAPTURERATE {}",
            get_parser=float,
            set_parser=self._set_capture_rate_parser,
        )
        """Parameter capture_rate"""

        max_rate = self.capture_rate_max()
        self.available_frequencies = [max_rate / 2**i for i in range(20)]

        self.capture_status: Parameter = (
            self.add_parameter(  # Are we capturing at the moment?
                "capture_status", label="capture status", get_cmd="CAPTURESTAT?"
            )
        )
        """Parameter capture_status"""

        self.count_capture_bytes: Parameter = self.add_parameter(
            "count_capture_bytes",
            label="captured bytes",
            get_cmd="CAPTUREBYTES?",
            unit="B",
            get_parser=int,
            docstring="Number of bytes captured so far in the buffer. Can be "
            "used to track live progress.",
        )
        """Number of bytes captured so far in the buffer. Can be used to track live progress."""

        self.count_capture_kilobytes: Parameter = self.add_parameter(
            "count_capture_kilobytes",
            label="captured kilobytes",
            get_cmd="CAPTUREPROG?",
            unit="kB",
            docstring="Number of kilobytes captured so far in the buffer, "
            "rounded-up to 2 kilobyte chunks. Capture must be "
            "stopped before requesting the value of this "
            "parameter. If the acquisition wrapped during operating "
            "in Continuous mode, then the returned value is "
            "simply equal to the current capture length.",
        )
        """
        Number of kilobytes captured so far in the buffer, rounded-up to 2 kilobyte chunks.
        Capture must be stopped before requesting the value of this parameter.
        If the acquisition wrapped during operating in Continuous mode,
        then the returned value is simply equal to the current capture length.
        """

        for parameter_name in ["X", "Y", "R", "T"]:
            self.add_parameter(parameter_name, parameter_class=SR86xBufferReadout)

    def snapshot_base(
        self,
        update: bool | None = False,
        params_to_skip_update: Sequence[str] | None = None,
    ) -> dict[Any, Any]:
        if params_to_skip_update is None:
            params_to_skip_update = []
        # we omit count_capture_kilobytes from the snapshot because
        # it can only be read after a completed capture and will
        # timeout otherwise when the snapshot is updated, e.g. at
        # station creation time
        params_to_skip_update = list(params_to_skip_update)
        params_to_skip_update.append("count_capture_kilobytes")

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

        if (
            not self.min_capture_length_in_kb
            <= capture_length_in_kb
            <= self.max_capture_length_in_kb
        ):
            raise ValueError(
                f"The capture length should be between "
                f"{self.min_capture_length_in_kb} and "
                f"{self.max_capture_length_in_kb}"
            )

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
            capture_rate_hz: The desired capture rate in Hz. If the desired
                rate is more than 1 Hz from the nearest valid rate, a warning
                is issued and the nearest valid rate it used.

        Returns:
            n_round

        """
        max_rate = self.capture_rate_max()
        n = np.log2(max_rate / capture_rate_hz)
        n_round = round(n)

        if not 0 <= n_round <= 20:
            raise ValueError(
                f"The chosen frequency is invalid. Please "
                f"consult the SR860 manual at page 136. "
                f"The maximum capture rate is {max_rate}"
            )

        nearest_valid_rate = max_rate / 2**n_round
        if abs(capture_rate_hz - nearest_valid_rate) > 1:
            available_frequencies = ", ".join(
                str(f) for f in self.available_frequencies
            )
            log.warning(f"Warning: Setting capture rate to {nearest_valid_rate:.5} Hz")
            log.warning(f"The available frequencies are: {available_frequencies}")

        return n_round

    def start_capture(self, acquisition_mode: str, trigger_mode: str) -> None:
        """
        Start an acquisition. Please see page 137 of the manual for a detailed
        explanation.

        Args:
            acquisition_mode: "ONE" | "CONT"
            trigger_mode: "IMM" | "TRIG" | "SAMP"

        """

        if acquisition_mode not in ["ONE", "CONT"]:
            raise ValueError("The acquisition mode needs to be either 'ONE' or 'CONT'")

        if trigger_mode not in ["IMM", "TRIG", "SAMP"]:
            raise ValueError(
                "The trigger mode needs to be either 'IMM', 'TRIG' or 'SAMP'"
            )

        cmd_str = f"CAPTURESTART {acquisition_mode}, {trigger_mode}"
        self.write(cmd_str)

    def stop_capture(self) -> None:
        """Stop a capture"""
        self.write("CAPTURESTOP")

    def _get_list_of_capture_variable_names(self) -> list[str]:
        """
        Retrieve the list of names of variables (readouts) that are
        set to be captured
        """
        return self.capture_config().split(",")

    def _get_number_of_capture_variables(self) -> int:
        """
        Retrieve the number of variables (readouts) that are
        set to be captured
        """
        capture_variables = self._get_list_of_capture_variable_names()
        n_variables = len(capture_variables)
        return n_variables

    def _calc_capture_size_in_kb(self, sample_count: int) -> int:
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
            sample_count: Number of samples that the buffer has to fit

        """
        total_size_in_kb = self._calc_capture_size_in_kb(sample_count)
        self.capture_length_in_kb(total_size_in_kb)

    def wait_until_samples_captured(self, sample_count: int) -> None:
        """
        Wait until the given number of samples is captured. This function
        is blocking and has to be used with caution because it does not have
        a timeout.

        Args:
            sample_count: Number of samples that needs to be captured

        """
        n_captured_bytes = 0
        n_variables = self._get_number_of_capture_variables()
        n_bytes_to_capture = sample_count * n_variables * self.bytes_per_sample
        while n_captured_bytes < n_bytes_to_capture:
            n_captured_bytes = self.count_capture_bytes()

    def get_capture_data(self, sample_count: int) -> dict[str, npt.NDArray]:
        """
        Read the given number of samples of the capture data from the buffer.

        Args:
            sample_count: number of samples to read from the buffer

        Returns:
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

    def _get_raw_capture_data(self, size_in_kb: int) -> npt.NDArray:
        """
        Read data from the buffer from its beginning avoiding the instrument
        limit of 64 kilobytes per reading.

        Args:
            size_in_kb :Size of the data that needs to be read; if it exceeds
                the capture length, an exception is raised.

        Returns:
            A one-dimensional numpy array of the requested data. Note that the
            returned array contains data for all the variables that are
            mentioned in the capture config.

        """
        current_capture_length = self.capture_length_in_kb()
        if size_in_kb > current_capture_length:
            raise ValueError(
                f"The size of the requested data ({size_in_kb}kB) "
                f"is larger than current capture length of the "
                f"buffer ({current_capture_length}kB)."
            )

        values: npt.NDArray = np.array([])
        data_size_to_read_in_kb = size_in_kb
        n_readings = 0

        while data_size_to_read_in_kb > 0:
            offset = n_readings * self.max_size_per_reading_in_kb

            if data_size_to_read_in_kb > self.max_size_per_reading_in_kb:
                size_of_this_reading = self.max_size_per_reading_in_kb
            else:
                size_of_this_reading = data_size_to_read_in_kb

            data_from_this_reading = self._get_raw_capture_data_block(
                size_of_this_reading, offset_in_kb=offset
            )
            values = np.append(values, data_from_this_reading)

            data_size_to_read_in_kb -= size_of_this_reading
            n_readings += 1

        return values

    def _get_raw_capture_data_block(
        self, size_in_kb: int, offset_in_kb: int = 0
    ) -> npt.NDArray:
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
            size_in_kb: Amount of data in kB that is to be read from the buffer
            offset_in_kb: Offset within the buffer of where to read the data;
                for example, when 0 is specified, the data is read from the
                start of the buffer.

        Returns:
            A one-dimensional numpy array of the requested data. Note that the
            returned array contains data for all the variables that are
            mentioned in the capture config.

        """
        if size_in_kb > self.max_size_per_reading_in_kb:
            raise ValueError(
                f"The size of the requested data ({size_in_kb}kB) "
                f"is larger than maximum size that can be read "
                f"at once ({self.max_size_per_reading_in_kb}kB)."
            )

        # Calculate the size of the data captured so far, in kB, rounded up
        # to 2kB chunks
        size_of_currently_captured_data = int(
            np.ceil(np.ceil(self.count_capture_bytes() / 1024) / 2) * 2
        )

        if size_in_kb > size_of_currently_captured_data:
            raise ValueError(
                f"The size of the requested data ({size_in_kb}kB) "
                f"cannot be larger than the size of currently "
                f"captured data rounded up to 2kB chunks "
                f"({size_of_currently_captured_data}kB)"
            )

        if offset_in_kb > size_of_currently_captured_data:
            raise ValueError(
                f"The offset for reading the requested data "
                f"({offset_in_kb}kB) cannot be larger than the "
                f"size of currently captured data rounded up to "
                f"2kB chunks "
                f"({size_of_currently_captured_data}kB)"
            )

        values = self._parent.visa_handle.query_binary_values(
            f"CAPTUREGET? {offset_in_kb}, {size_in_kb}",
            datatype="f",
            is_big_endian=False,
            expect_termination=False,
        )
        # the sr86x does not include an extra termination char on binary
        # messages so we set expect_termination to False

        return np.array(values)

    def capture_one_sample_per_trigger(
        self, trigger_count: int, start_triggers_pulsetrain: Callable[..., Any]
    ) -> dict[str, npt.NDArray]:
        """
        Capture one sample per each trigger, and return when the specified
        number of triggers has been received.

        Args:
            trigger_count: Number of triggers to capture samples for
            start_triggers_pulsetrain: By calling this *non-blocking*
                function, the train of trigger pulses should start

        Returns:
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

    def capture_samples_after_trigger(
        self, sample_count: int, send_trigger: Callable[..., Any]
    ) -> dict[str, npt.NDArray]:
        """
        Capture a number of samples after a trigger has been received.
        Please refer to page 135 of the manual for details.

        Args:
            sample_count: Number of samples to capture
            send_trigger: By calling this *non-blocking* function, one trigger
                should be sent that will initiate the capture

        Returns:
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

    def capture_samples(self, sample_count: int) -> dict[str, npt.NDArray]:
        """
        Capture a number of samples at a capture rate, starting immediately.
        Unlike the "continuous" capture mode, here the buffer does not get
        overwritten with the new data once the buffer is full.

        The function blocks until the required number of samples is acquired,
        and returns them.

        Args:
            sample_count: Number of samples to capture

        Returns:
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
        parent: an instance of SR86x driver
        name: data channel name that is to be used to reference it from the
            parent
        cmd_id: this ID is used in VISA commands to refer to this data channel,
            usually is an integer number
        cmd_id_name: this name can also be used in VISA commands along with
            channel_id; it is not used in this implementation, but is added
            for reference
        color: every data channel is also referred to by the color with which it
            is being plotted on the instrument's screen; added here only for
            reference

    """

    def __init__(
        self,
        parent: SR86x,
        name: str,
        cmd_id: str,
        cmd_id_name: str | None = None,
        color: str | None = None,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ) -> None:
        super().__init__(parent, name, **kwargs)

        self._cmd_id = cmd_id
        self._cmd_id_name = cmd_id_name
        self._color = color

        self.assigned_parameter: Parameter = self.add_parameter(
            "assigned_parameter",
            label=f"Data channel {cmd_id} parameter",
            docstring=f"Allows to set and get the "
            f"parameter that is assigned to data "
            f"channel {cmd_id}",
            set_cmd=f"CDSP {cmd_id}, {{}}",
            get_cmd=f"CDSP? {cmd_id}",
            val_mapping=self.parent.PARAMETER_NAMES,
        )
        """Allows to set and get the parameter that is assigned to the channel"""

    @property
    def cmd_id(self) -> str:
        return self._cmd_id

    @property
    def cmd_id_name(self) -> str | None:
        return self._cmd_id_name

    @property
    def color(self) -> str | None:
        return self._color
