from typing import List, Dict, Tuple, Any

from qcodes.instrument.parameter import Parameter
from qcodes.instrument.channel import InstrumentChannel
from qcodes.station import Station

from qcodes.instrument_drivers.meta.meta_instrument import (
    MetaParameter,
    MetaInstrument
)


class DeviceMetaParameter(MetaParameter):
    """Meta for a device parameter."""
    def __init__(
        self,
        name: str,
        endpoints: Tuple[Parameter],
        endpoint_names: Tuple[str],
        channel: InstrumentChannel = None,
        **kwargs
    ):
        """Meta parameter for a Device on a ChipMeta instrument

        Args:
            name: Parameter name
            endpoints: Endpoints to connect to
            endpoint_names: Names of endpoints to connect to
            channel: Optionally set the channel this parameter refers to.
                Defaults to None.
        """
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
        output = f"DeviceMetaParameter(name={self.name}"
        if self.channel:
            output += f", channel={self.channel.name}"
        if self.endpoints:
            endpoints = ", ".join([str(_) for _ in self.endpoints])
            output += f", endpoints=({endpoints})"
        output += ")"
        return output


class DeviceMeta(MetaInstrument):
    """Meta instrument for a quantum device on a chip"""
    param_cls = DeviceMetaParameter

    @staticmethod
    def _connect_aliases(
        connections: Dict[str, List[str]],
        channels: Dict[str, List[str]]
    ):
        aliases = {}
        for key, channel_name in channels.items():
            aliases[key] = connections.get(channel_name)
        return aliases

    def __init__(
        self,
        name: str,
        station: Station,
        channels: Dict[str, List[str]],
        initial_values: Dict[str, Any],
        connections: Dict[str, List[str]],
        set_initial_values_on_load: bool = False
    ):
        """Create a DeviceMeta instrument

        Args:
            name: DeviceMeta name
            station: Measurement station with real instruments
            channels: Channels to connect
            initial_values: Default values to set on instrument load
            connections: Connections from channels to endpoint instrument parameters
            set_initial_values_on_load: Set default values on load.
                Defaults to False.
        """
        self._connections = connections
        self._channels = channels

        super().__init__(
            name=name,
            station=station,
            aliases=self._connect_aliases(connections, channels),
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
            channel = self.parse_instrument_path(
                station=station,
                path=self._channels[param_name]
            )

            self._add_parameter(
                param_name=param_name,
                station=station,
                paths=paths,
                channel=channel,
                setter=setters.get(param_name),
                unit=units.get(param_name)
            )
