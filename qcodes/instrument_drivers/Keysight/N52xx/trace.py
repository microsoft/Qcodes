import re
import numpy as np
from typing import List, Sequence, Any

from ._N52xx_channel_ext import N52xxInstrumentChannel
from qcodes import Instrument, InstrumentChannel, ArrayParameter
from qcodes.utils.validators import Enum


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


class N52xxTrace(N52xxInstrumentChannel):
    data_formats = {
        "magnitude": {"sweep_format": "MLOG", "unit": "dBm"},
        "log_magnitude": {"sweep_format": "MLOG", "unit": "dBm"},
        "linear_magnitude": {"sweep_format": "MLIN", "unit": "-"},
        "phase": {"sweep_format": "PHAS", "unit": "deg"},
        "unwrapped_phase": {"sweep_format": "UPH", "unit": "deg"},
        "group_delay": {"sweep_format": "GDEL", "unit": "s"},
        "real": {"sweep_format": "REAL", "unit": "-"},
        "imaginary": {"sweep_format": "IMAG", "unit": "-"}
    }

    @classmethod
    def load_from_instrument(
            cls, parent, **kwargs) -> List['N52xxInstrumentChannel']:

        obj_list = []
        for name, trace_type in cls._discover_list_from_instrument(parent):
            obj = cls(parent, identifier=name, trace_type=trace_type,
                      existence=True, **kwargs)
            obj_list.append(obj)

        return obj_list

    @classmethod
    def _discover_list_from_instrument(cls, parent, **kwargs) -> List[Any]:
        result = parent.base_instrument.ask(
            f"CALC{parent.channel}:PAR:CAT:EXT?")
        if result == "NO CATALOG":
            return []

        trace_info = result.strip().strip("\"").split(",")
        trace_names = trace_info[::2]
        trace_types = trace_info[1::2]
        return list(zip(trace_names, trace_types))

    @classmethod
    def make_unique_id(cls, parent, **kwargs):
        trace_type = kwargs["trace_type"]
        name = trace_type

        if (name, trace_type) in cls._discover_list_from_instrument(parent):
            raise ValueError(
                f"A trace of type {trace_type} already exists "
                f"on channel {parent.channel}"
            )

        return name

    def __init__(self, parent, identifier, existence=False, channel_list=None,
                 **kwargs):

        super().__init__(parent, identifier, existence=existence,
                         channel_list=channel_list)
        self._channel = parent.channel
        self._trace_type = kwargs["trace_type"]
        self.validate_trace_type(self._trace_type)

        self.add_parameter(
            'format',
            get_cmd=f'CALC{self._channel}:FORM?',
            set_cmd=f'CALC{self._channel}:FORM {{}}',
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
                channel=self.parent,
                trace=self,
                label=format_name,
                **format_args
            )

    def _create(self):
        # Write via self.parent to avoid a call to self.select. This will cause
        # the instrument to return an error as the trace does not exist yet
        self.parent.write(
            f'CALC{self._channel}:PAR:EXT {self.short_name}, '
            f'{self._trace_type}'
        )

    def _delete(self):
        self.parent.write(
            f'CALC{self._channel}:PAR:DEL {self.short_name}'
        )

    def write(self, cmd: str) -> None:
        self.select()
        super().write(cmd)

    def ask(self, cmd: str) ->None:
        self.select()
        return super().ask(cmd)

    def _change_trace_type(self, new_type) -> None:
        """
        Change the measurement type of this trace
        """
        self.validate_trace_type(new_type)
        self.write(
            f"CALC{self._channel}:PAR:MOD:EXT '{new_type}'")

        self._trace_type = new_type

    @staticmethod
    def validate_trace_type(trace_type: str) -> None:
        if re.fullmatch(r"S\d\d", trace_type) is None:
            raise ValueError(
                f"The trace type format {trace_type}. This needs to be in the "
                f"form Sxy where 'x' and 'y' are integers"
            )

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
        visa_handle = self.base_instrument.visa_handle

        self.format(format_str)
        self.select()
        data = np.array(visa_handle.query_binary_values(
            f'CALC{self._channel}:DATA? FDATA', datatype='f',
            is_big_endian=True
        ))

        return data

    def select(self) -> None:
        # Warning: writing self.write here will cause an infinite recursion
        self.parent.write(
            f"CALC{self._channel}:PAR:SEL {self.short_name}")
