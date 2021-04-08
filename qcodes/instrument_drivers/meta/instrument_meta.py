from typing import List, Dict, Tuple, Union, Any, Optional

from collections import namedtuple
from functools import partial

from qcodes.instrument.parameter import Parameter
from qcodes.instrument.base import InstrumentBase
from qcodes.station import Station

import logging

_log = logging.getLogger(__name__)


class MetaParameter(Parameter):
    """Meta parameter that returns one or more endpoint parameter values"""
    def __init__(
        self,
        name: str,
        endpoints: Tuple[Parameter],
        endpoint_names: Tuple[str],
        setter: callable = None,
        **kwargs
    ):
        """Meta parameter that acts as an alias to a real instrument parameter

        Args:
            name (str): Meta parameter name
            endpoints (Tuple[Parameter]): One or more endpoint parameters to alias
            endpoint_names (Tuple[str]): One or more endpoint names
            setter (callable, optional): Optional setter function to override 
                endpoint parameter setter. Defaults to None.
        """
        super().__init__(name, **kwargs)
        self._endpoints = endpoints
        endpoint_names = endpoint_names or [_e.name for _e in endpoints]
        self._namedtuple = namedtuple(name, endpoint_names)
        # Generate sub parameters if >1 endpoints,
        # e.g. my_meta_param.X, my_meta_param.Y
        self._sub_parameters(endpoints, endpoint_names)
        self._setter = setter

    def _sub_parameters(self, endpoints, endpoint_names):
        """Generate sub parameters if there are > 1 endpoints."""
        if len(endpoints) > 1:
            parameters = []
            for endpoint, endpoint_name in zip(
                endpoints,
                endpoint_names
            ):
                parameter = self.add_parameter(
                    cls=self.__class__,
                    name=endpoint_name,
                    endpoint=endpoint
                )
                parameters.append(parameter)
            self._parameters = tuple(parameters)
        else:
            self._parameters = (self,)

    def add_parameter(
        self,
        cls,
        name: str,
        endpoint: Parameter
    ) -> Parameter:
        """Add a sub parameter to the meta parameter

        Args:
            name (str): Sub parameter name
            endpoint (Parameter): Endpoint parameter object

        Returns:
            Parameter: Parameter instance
        """
        assert not hasattr(
            self, name
        ), f"Duplicate parameter name {name}."
        parameter = cls(
            name=name,
            endpoints=(endpoint,),
            endpoint_names=(endpoint.name,)
        )
        setattr(self, name, parameter)
        return parameter

    @property
    def endpoints(self):
        """Get endpoint parameters"""
        return self._endpoints

    @property
    def parameters(self):
        """Get all sub parameters"""
        return self._parameters

    def get_raw(self):
        """Get parameter raw value"""
        if len(self.endpoints) > 1:
            return self._namedtuple(*[_e() for _e in self.endpoints])
        return self.endpoints[0]()

    def set_raw(self, value: float):
        """Set parameter raw value

        Args:
            value (float): Parameter value to set

        Returns:
            float: Returns the parameter value
        """
        if self._setter is not None:
            return self._setter(value)
        elif len(self.endpoints) == 1:
            return self._endpoints[0](value)
 
    def connect(self, *params):
        """Connect endpoint parameter"""
        self._endpoints = params

    def disconnect(self):
        """Disconnect endpoint parameter"""
        self._endpoints = ()

    def __repr__(self):
        output = f"MetaParameter(name={self.name}"
        if self.endpoints:
            endpoints = ", ".join([str(_) for _ in self.endpoints])
            output += f", endpoints=({endpoints})"
        output += ")"
        return output


class InstrumentMeta(InstrumentBase):
    """Meta instrument for aliasing instrument parameters"""
    param_cls = MetaParameter

    @staticmethod
    def parse_instrument_path(station: Station, path: Union[str, List[str]]):
        """Parse a string path and return the object relative to the station,
        e.g. "my_instrument.my_param" returns station.my_instrument.my_param

        Args:
            station (Station): Measurement station
            path (Union[str, List[str]]): Relative path to parse
        """
        def _parse_path(parent, elem):
            child = getattr(parent, elem[0])
            if len(elem) == 1:
                return child
            return _parse_path(child, elem[1:])

        return _parse_path(station, path.split("."))

    def __init__(
        self,
        name: str,
        station: Station,
        aliases: Dict[str, List[str]],
        default_values: Dict[str, Any] = None,
        set_defaults_on_load: bool = False,
        setters: Dict[str, Dict[str, Any]] = None,
        metadata: Optional[Dict[Any, Any]] = None):
        """InstrumentMeta class for creating a meta instrument that aliases one or more 
        parameter endpoints from real instruments.

        Example usage in instrument YAML:

        mux:
            type: forq.meta.InstrumentMeta
            init:
            aliases:
                drive:
                - rf_source1.frequency
                - rf_source1.power
                - rf_source1.phase
                - rf_source1.status
                mixer:
                - rf_source2.frequency
                - rf_source2.power
                - rf_source2.phase
                - rf_source2.status
                readout:
                - dmm.volt

        Args:
            name (str): Instrument name
            station (Station): Real instrument station to connect the endpoints to.
            aliases (Dict[str, List[str]]): Aliases and the endpoints they connect to.
            default_values (Dict[str, Any], optional): Default values to set on meta instrument.
                Defaults to None.
            set_defaults_on_load (bool, optional): Flag to set defaults on load or not. Defaults to False.
            setters (Dict[str, Dict[str, Any]], optional): Optional setter methods to use instead
                of calling the .set() method on the endpoint parameters. Defaults to None.
            metadata (Optional[Dict[Any, Any]], optional): Optional metadata to pass to instrument.
                Defaults to None.
        """
        super().__init__(name=name, metadata=metadata)
        self._add_parameters(
            station=station,
            aliases=aliases,
            setters=setters or {}
        )
        self._default_values = default_values or {}
        if set_defaults_on_load:
            self.set_defaults()

    def set_defaults(self, dry_run: bool = False):
        """Set default parameters on meta instrument

        Args:
            dry_run (bool, optional): Dry run to test if defaults are set correctly.
                Defaults to False.
        """
        _log.debug(f"Setting default values: {self._default_values}")
        for path, value in self._default_values.items():
            param = self.parse_instrument_path(self, path=path)
            msg = f"Setting parameter {self.name}.{path} to {value}."
            if not dry_run:
                _log.debug(msg)
                if hasattr(param, "set"):
                    param.set(value)
                else:
                    _log.debug("No set method found, trying to assign value.")
                    if "." in path:
                        name = path.split(".")[-1]
                        parent_path = ".".join(path.split(".")[:-1])
                        parent = self.parse_instrument_path(self, path=parent_path)
                    else:
                        parent, name = self, path
                    print(parent, name, value)
                    setattr(parent, name, value)
            else:
                print(f"Dry run: {msg}")

    def _add_parameters(
        self,
        station: Station,
        aliases: Dict[str, List[str]],
        setters: Dict[str, Dict[str, Any]]
    ):
        """Add parameters to meta instrument based on specified aliases, endpoints
        and setter methods"""
        for param_name, paths in aliases.items():
            self._add_parameter(
                param_name=param_name,
                station=station,
                paths=paths,
                setter=setters.get(param_name)
            )

    @staticmethod
    def _endpoint_names(endpoints: List[Parameter]):
        """Get the endpoint names"""
        endpoint_names = [_e.name for _e in endpoints]
        if len(endpoint_names) != len(set(endpoint_names)):
            endpoint_names = [
                f"{_e}{n}" for n, _e in enumerate(endpoint_names)
            ]
        return endpoint_names

    def _add_parameter(
        self,
        param_name: str,
        station: Station,
        paths: List[str],
        setter: Dict[str, Any],
        **kwargs
    ):
        """Create meta parameter that links to a given set of paths
        (e.g. my_instrument.my_param) on the station"""
        endpoints = tuple(
            self.parse_instrument_path(station, path) for path in paths
        )

        if setter is not None:
           setter_fn = self.parse_instrument_path(station, setter.pop("method"))
           setter = partial(setter_fn, **setter)

        self.add_parameter(
            name=param_name,
            parameter_class=self.param_cls,
            endpoints=endpoints,
            endpoint_names=self._endpoint_names(endpoints),
            setter=setter,
            **kwargs
        )

    def __repr__(self):
        params = ", ".join(self.parameters.keys())
        return f"InstrumentMeta(name={self.name}, parameters={params})"


class ChannelInstrumentMeta(InstrumentMeta):
    """Meta instrument that auto generates aliases for a given ChannelList"""
    def __init__(
        self,
        name: str,
        station: Station,
        channels: str,
        aliases: Dict[str, List[str]],
        default_values: Dict[str, Any] = None,
        set_defaults_on_load: bool = False,
        **kwargs):
        """Create a ChannelInstrumentMeta instrument

        Args:
            name (str): Instrument name
            station (Station): Station with real instruments to connect to
            channels (str): Path to channels, e.g. my_instrument.channels
            aliases (Dict[str, List[str]]): Aliases to specify for instrument,
                these are auto-generated per channel
            default_values (Dict[str, Any], optional): Default values to set on
                instrument load. Defaults to None.
            set_defaults_on_load (bool, optional): Flag to set defaults on load.
                Defaults to False.
        """
        _channels = self.parse_instrument_path(station=station, path=channels)
        _aliases = {}
        for channel in _channels:
            ins = channel.root_instrument.name
            chan_no = str(channel.channel_number()).zfill(2)
            for alias, paths in aliases.items():
                _paths = [
                    f"{ins}.ch{chan_no}.{path}" for path in paths
                ]
                _aliases[f"{alias}{chan_no}"] = _paths

        super().__init__(
            name=name,
            station=station,
            aliases=_aliases,
            default_values=default_values,
            set_defaults_on_load=set_defaults_on_load,
            **kwargs
        )
