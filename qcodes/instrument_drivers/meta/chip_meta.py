from typing import List, Dict, Tuple, Union, Any

from qcodes.instrument.parameter import Parameter
from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.channel import InstrumentChannel
from qcodes.station import Station

from qcodes.instrument_drivers.meta.instrument_meta import MetaParameter, InstrumentMeta


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
            name (str): Parameter name
            endpoints (Tuple[Parameter]): Endpoints to connect to
            endpoint_names (Tuple[str]): Names of endpoints to connect to
            channel (InstrumentChannel, optional): Optionally set the channel 
                this parameter refers to. Defaults to None.
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
            name (str): Name of sub parameter
            endpoint (Parameter): Endpoint parameter to connect to

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
            channel=self.channel
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


class DeviceMeta(InstrumentMeta):
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
        default_values: Dict[str, Dict[str, Any]],
        connections: Dict[str, List[str]],
        set_defaults_on_load: bool = False
    ):
        """Create a DeviceMeta instrument

        Args:
            name (str): DeviceMeta name
            station (Station): Measurement station with real instruments
            channels (Dict[str, List[str]]): Channels to connect
            default_values (Dict[str, Dict[str, Any]]): Default values to set on
                instrument load
            connections (Dict[str, List[str]]): Connections from channels to
                endpoint instrument parameters
            set_defaults_on_load (bool, optional): Set default values on load.
                Defaults to False.
        """
        self._connections = connections
        self._channels = channels

        super().__init__(
            name=name,
            station=station,
            aliases=self._connect_aliases(connections, channels),
            default_values=default_values,
            set_defaults_on_load=set_defaults_on_load
        )

    def _add_parameters(
        self,
        station: Station,
        aliases: Dict[str, List[str]],
        setters: Dict[str, Dict[str, Any]]
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
                setter=setters.get(param_name)
            )


class ChipMeta(InstrumentBase):
    """Meta instrument for a chip"""
    def __init__(
        self,
        name: str,
        station: "Station",
        devices: Dict[str, Dict[str, List[str]]],
        default_values: Dict[str, Dict[str, Any]],
        connections: Dict[str, Dict[str, Union[str, List[str]]]],
        set_defaults_on_load: bool = False,
        **kwargs):
        """
        Create a ChipMeta instrument

        Args:
            name (str): Chip name
            station (Station): Measurement station with real instruments
            devices (Dict[str, Dict[str, List[str]]]): Devices on the chip,
                for each a DeviceMeta is created
            default_values (Dict[str, Dict[str, Any]]): Default values to set on load
            connections (Dict[str, Dict[str, Union[str, List[str]]]]): [description]
            set_defaults_on_load (bool, optional): [description]. Defaults to False.
        """
        super().__init__(name=name, **kwargs)

        for device_name, channels in devices.items():
            device = DeviceMeta(
                name=device_name,
                station=station,
                channels=channels,
                default_values=default_values.get(device_name),
                connections=connections,
                set_defaults_on_load=set_defaults_on_load
            )

            self.add_submodule(
                device_name,
                device
            )

    def __repr__(self):
        devices = ", ".join(self.submodules.keys())
        return f"ChipMeta(name={self.name}, devices={devices})"
