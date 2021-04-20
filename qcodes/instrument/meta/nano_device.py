from typing import List, Dict, Tuple, Any

from qcodes.instrument.parameter import Parameter
from qcodes.instrument.channel import InstrumentChannel
from qcodes.station import Station

from qcodes.instrument.meta.meta_instrument import (
    MetaParameter,
    MetaInstrument
)


class NanoDeviceParameter(MetaParameter):
    """
    Meta for a NanoDevice parameter.

    Args:
        name: Parameter name
        endpoints: Endpoints to connect to
        endpoint_names: Names of endpoints to connect to
        channel: Optionally set the channel this parameter refers to.
            Defaults to None.
    """
    def __init__(
        self,
        name: str,
        endpoints: Tuple[Parameter],
        endpoint_names: Tuple[str],
        channel: InstrumentChannel = None,
        **kwargs
    ):
        self._channel = channel
        super().__init__(
            name=name,
            endpoints=endpoints,
            endpoint_names=endpoint_names,
            **kwargs
        )

    @property
    def channel(self):
        return self._channel

    def add_parameter(
        self,
        cls,
        name: str,
        endpoint: Parameter
    ) -> Parameter:
        """Add a sub parameter to this parameter,
        e.g. my_instrument.param.sub_param

        Args:
            name: Name of sub parameter
            endpoint: Endpoint parameter to connect to

        Returns:
            Parameter: The parameter instance
        """
        assert not hasattr(
            self, name
        ), f"Duplicate parameter name {name}."
        parameter = cls(
            name=name,
            endpoints=(endpoint,),
            endpoint_names=(endpoint.name,),
            channel=self.channel,
            instrument=None
        )
        setattr(self, name, parameter)
        return parameter

    def __repr__(self):
        output = f"NanoDeviceParameter(name={self.name}"
        if self.channel:
            output += f", channel={self.channel.name}"
        if self.endpoints:
            endpoints = ", ".join([str(_) for _ in self.endpoints])
            output += f", endpoints=({endpoints})"
        output += ")"
        return output


class NanoDevice(MetaInstrument):
    """
    Meta instrument for a quantum device on a chip

    Args:
        name: NanoDevice name
        station: Measurement station with real instruments
        endpoints: Parameter endpoints to connect to
        initial_values: Default values to set on instrument load
        set_initial_values_on_load: Set default values on load.
            Defaults to False.
    """
    param_cls = NanoDeviceParameter

    def __init__(
        self,
        name: str,
        station: Station,
        aliases: Dict[str, List[str]],
        initial_values: Dict[str, Any],
        set_initial_values_on_load: bool = False
    ):
        self._aliases = aliases

        super().__init__(
            name=name,
            station=station,
            aliases=aliases,
            initial_values=initial_values,
            set_initial_values_on_load=set_initial_values_on_load
        )

    def _add_parameters(
        self,
        station: Station,
        aliases: Dict[str, List[str]],
        setters: Dict[str, Dict[str, Any]],
        units: Dict[str, Dict[str, str]]
    ):
        for param_name, paths in aliases.items():
            try:
                channel = self.parse_instrument_path(
                    station=station,
                    path=self._channels[param_name]
                )
            except AttributeError:
                channel = None

            self._add_parameter(
                param_name=param_name,
                station=station,
                paths=paths,
                channel=channel,
                setter=setters.get(param_name),
                unit=units.get(param_name)
            )
