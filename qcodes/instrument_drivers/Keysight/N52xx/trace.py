import numpy as np
from typing import Sequence
import re
import logging

from qcodes import InstrumentChannel, ArrayParameter, Instrument
from qcodes.utils.validators import Enum

logger = logging.getLogger()


class TraceParameter(ArrayParameter):
    def __init__(
        self,
        name: str,
        instrument: 'Instrument',
        channel: 'InstrumentChannel',
        trace: 'N52xxTrace',
        sweep_format: str,
        label: str,
        unit: str,
    ) -> None:

        self._sweep_format = sweep_format
        self._channel = channel
        self._trace = trace

        super().__init__(
            name,
            instrument=instrument,
            label=label,
            unit=unit,
            setpoint_names=('frequency',),
            setpoint_labels=('Frequency',),
            setpoint_units=('Hz',),
            shape=(0,),
            setpoints=((0,),),
        )

    @property
    def shape(self) -> tuple:
        return self._channel.points(),

    @shape.setter
    def shape(self, val: Sequence[int]) -> None:
        pass

    @property
    def setpoints(self) -> tuple:
        start = self._channel.start()
        stop = self._channel.stop()
        return np.linspace(start, stop, self.shape[0]),

    @setpoints.setter
    def setpoints(self, val: Sequence[int]) -> None:
        pass

    def get_raw(self) -> Sequence[float]:
        return self._trace._get_raw_data(self._sweep_format)


class N52xxTrace(InstrumentChannel):
    """
    Allow operations on individual N52xx traces.
    """

    data_formats = {
        "log_magnitude": {"sweep_format": "MLOG", "unit": "dBm"},
        "linear_magnitude": {"sweep_format": "MLIN", "unit": "-"},
        "phase": {"sweep_format": "PHAS", "unit": "deg"},
        "unwrapped_phase": {"sweep_format": "UPH", "unit": "deg"},
        "group_delay": {"sweep_format": "GDEL", "unit": "s"},
        "real": {"sweep_format": "REAL", "unit": "-"},
        "imaginary": {"sweep_format": "IMAG", "unit": "-"}
    }

    def __init__(
            self, parent: 'Instrument', channel: 'InstrumentChannel',
            name: str, trace_type: str, present_on_instrument: bool=False
    ) -> None:

        self.validate_trace_type(trace_type)

        super().__init__(parent, name)
        self._channel = channel
        self._channel_number = channel.channel_number
        self._trace_type = trace_type

        self.add_parameter(
            'format',
            get_cmd=f'CALC{self._channel_number}:FORM?',
            set_cmd=f'CALC{self._channel_number}:FORM {{}}',
            vals=Enum(*[d["sweep_format"] for d in self.data_formats.values()])
        )

        self.add_parameter(
            "measurement_type",
            get_cmd=lambda: self._trace_type,
            set_cmd=self._change_trace_type
        )

        for format_name, format_args in self.data_formats.items():
            self.add_parameter(
                format_name,
                parameter_class=TraceParameter,
                channel=self._channel,
                trace=self,
                label=format_name,
                **format_args
            )

        self._present_on_instrument = present_on_instrument

    def _change_trace_type(self, new_type) ->None:
        """
        Change the measurement type of this trace
        """
        self.validate_trace_type(new_type)
        self.select()
        self.parent.write(
            f"CALC{self._channel_number}:PAR:MOD:EXT '{new_type}'")

        self._trace_type = new_type

    @staticmethod
    def validate_trace_type(trace_type: str) ->None:
        if re.fullmatch(r"S\d\d", trace_type) is None:
            raise ValueError(
                "The trace type needs to be in the form Sxy where "
                "'x' and 'y' are integers"
            )

    @property
    def present_on_instrument(self) ->bool:
        return self._present_on_instrument

    @property
    def name_on_instrument(self) ->str:
        return self.short_name

    def select(self) -> None:
        if not self._present_on_instrument:
            raise RuntimeError(
                "Trace is not present on the instrument (anymore). It was "
                "either deleted or never uploaded in the first place"
            )
        # Warning: writing self.write here will cause an infinite recursion
        self.parent.write(
            f"CALC{self._channel_number}:PAR:SEL {self.short_name}")

    def write(self, cmd: str) -> None:
        self.select()
        super().write(cmd)

    def ask(self, cmd: str) -> str:
        self.select()
        return super().ask(cmd)

    def _get_raw_data(self, format_str: str) -> np.ndarray:
        """
        Args:
            format_str (str): Our data is complex values (that is, has a
            magnitude and phase). Possible value are
                * "MLOG" (log_magnitude)
                * "MLIN" (linear magnitude)
                * "PHAS" (phase)
                * "UPH" (unwrapped phase)
                * "GDEL" (group delay)
                * "REAL"
                * "IMAG"
        """
        visa_handle = self.parent.visa_handle

        self.format(format_str)
        self.select()
        data = np.array(visa_handle.query_binary_values(
            f'CALC{self._channel_number}:DATA? FDATA', datatype='f',
            is_big_endian=True
        ))

        return data

    def upload_to_instrument(self) -> None:
        """
        upload to instrument
        """
        # Do not do self.write; self.select will not work yet as the
        # instrument has not been uploaded yet
        self.parent.write(
            f'CALC{self._channel_number}:PAR:EXT {self.short_name}, '
            f'{self._trace_type}'
        )

        self._present_on_instrument = True

    def delete(self) -> None:
        """
        delete from instrument
        """
        self.parent.write(
            f'CALC{self._channel_number}:PAR:DEL {self.short_name}')
        self._present_on_instrument = False

    def __repr__(self):
        return f"<Trace Type: {self._trace_type}>"
