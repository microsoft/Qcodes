from typing import List, Dict, Union, Any, Optional

from functools import partial

from qcodes.instrument.delegate.delegate_group_parameter import (
    DelegateGroup,
    DelegateGroupParameter
)
from qcodes.instrument.parameter import Parameter
from qcodes.instrument.base import InstrumentBase
from qcodes.station import Station

import logging

_log = logging.getLogger(__name__)


class DelegateInstrument(InstrumentBase):
    """DelegateInstrument class for creating an instrument with one or
    more delegate parameters that connect to real instrument parameters.

    Example usage in instrument YAML:

    field:
        type: qcodes.instrument.delegate.DelegateInstrument
        init:
        aliases:
            X:
            - field_X.field
            ramp_rate:
            - field_X.ramp_rate
        set_initial_values_on_load: true
        initial_values:
            ramp_rate: 0.02
        setters:
            X:
            method: field_X.set_field
            block: false
        units:
            X: T
            ramp_rate: T/min

    this will generate an instrument named "field" with methods for
    delegate parameters:
        field.X()
        field.ramp_rate()

    that are delegate parameters for:
        field_X.field()
        field_X.ramp_rate()

    Additionally, this will set field_X.ramp_rate(0.02) on load and
    override the field.X.set() method with
        field_X.set_field(value, block=False),
    as opposed to field.X.field.set() which ramps with block=True.

    Args:
        name: Instrument name
        station: Real instrument station that is used to get the endpoint
            parameters.
        aliases: A mapping from name of a delegate parameter to the sequence of
            endpoints that it connects to.
        initial_values: Default values to set on the delegate instrument's
            parameters. Defaults to None.
        set_initial_values_on_load: Flag to set initial values when the
            instrument is loaded. Defaults to False.
        setters: Optional setter methods to use instead of calling the .set()
            method on the endpoint parameters. Defaults to None.
        metadata: Optional metadata to pass to instrument. Defaults to None.
    """
    param_cls = DelegateGroupParameter

    def __init__(
        self,
        name: str,
        station: Station,
        aliases: Dict[str, List[str]],
        initial_values: Dict[str, Any] = None,
        set_initial_values_on_load: bool = False,
        setters: Dict[str, Dict[str, Any]] = None,
        units: Dict[str, Dict[str, str]] = None,
        metadata: Optional[Dict[Any, Any]] = None):
        super().__init__(name=name, metadata=metadata)
        self._create_and_add_parameters(
            station=station,
            aliases=aliases,
            setters=setters or {},
            units=units or {}
        )
        self._initial_values = initial_values or {}
        if set_initial_values_on_load:
            self.set_initial_values()
    
    @staticmethod
    def parse_instrument_path(station: Station, path: Union[str, List[str]]):
        """Parse a string path and return the object relative to the station,
        e.g. "my_instrument.my_param" returns station.my_instrument.my_param

        Args:
            station: Measurement station
            path: Relative path to parse
        """
        def _parse_path(parent, elem):
            child = getattr(parent, elem[0])
            if len(elem) == 1:
                return child
            return _parse_path(child, elem[1:])

        return _parse_path(station, path.split("."))

    def set_initial_values(self, dry_run: bool = False):
        """Set parameter initial values on meta instrument

        Args:
            dry_run: Dry run to test if defaults are set correctly.
                Defaults to False.
        """
        _log.debug(f"Setting default values: {self._initial_values}")
        for path, value in self._initial_values.items():
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

    def _create_and_add_parameters(
        self,
        station: Station,
        aliases: Dict[str, List[str]],
        setters: Dict[str, Dict[str, Any]],
        units: Dict[str, Dict[str, str]]
    ):
        """Add parameters to meta instrument based on specified aliases,
        endpoints and setter methods"""
        for param_name, paths in aliases.items():
            self._create_and_add_parameter(
                param_name=param_name,
                station=station,
                paths=paths,
                setter=setters.get(param_name),
                unit=units.get(param_name)
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

    def _create_and_add_parameter(
        self,
        param_name: str,
        station: Station,
        paths: List[str],
        setter: Dict[str, Any],
        unit: str,
        **kwargs
    ):
        """Create meta parameter that links to a given set of paths
        (e.g. my_instrument.my_param) on the station"""
        endpoints = tuple(
            self.parse_instrument_path(station, path) for path in paths
        )
        endpoint_names = self._endpoint_names(endpoints)

        if setter is not None:
           setter_fn = self.parse_instrument_path(station, setter.pop("method"))
           setter = partial(setter_fn, **setter)

        self.add_parameter(
            name=param_name,
            parameter_class=self.param_cls,
            endpoints=endpoints,
            endpoint_names=endpoint_names,
            setter=setter,
            unit=unit,
            **kwargs
        )
        if len(endpoints) > 1:
            self.parameters[param_name]._group = DelegateGroup(
                name=param_name,
                parameters=endpoints,
                parameter_names=endpoint_names
            )

    def __repr__(self):
        params = ", ".join(self.parameters.keys())
        return f"DelegateInstrument(name={self.name}, parameters={params})"
