from typing import Callable, List, Dict, Union, Any, Optional, Sequence

from functools import partial

from qcodes.instrument.delegate.grouped_parameter import (
    DelegateGroup,
    DelegateGroupParameter,
    GroupedParameter
)
from qcodes.instrument.parameter import DelegateParameter, Parameter
from qcodes.instrument.base import InstrumentBase
from qcodes.station import Station

import logging

_log = logging.getLogger(__name__)


class DelegateInstrument(InstrumentBase):
    """DelegateInstrument is an instrument driver with one or more
    parameters that connect to instrument parameters.

    Example usage in instrument YAML:

    field:
        type: qcodes.instrument.delegate.DelegateInstrument
        init:
        parameters:
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

    this will generate an instrument named "field" with methods:
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
        parameters: A mapping from the name of a parameter to the sequence
            of source parameters that it points to.
        initial_values: Default values to set on the delegate instrument's
            parameters. Defaults to None.
        set_initial_values_on_load: Flag to set initial values when the
            instrument is loaded. Defaults to False.
        setters: Optional setter methods to use instead of calling the .set()
            method on the endpoint parameters. Defaults to None.
        units: Optional units to set for parameters.
        metadata: Optional metadata to pass to instrument. Defaults to None.
    """
    param_cls = DelegateGroupParameter

    def __init__(
        self,
        name: str,
        station: Station,
        parameters: Dict[str, List[str]],
        initial_values: Optional[Dict[str, Any]] = None,
        set_initial_values_on_load: bool = False,
        setters: Optional[Dict[str, Dict[str, Any]]] = None,
        units: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[Any, Any]] = None
    ):
        super().__init__(name=name, metadata=metadata)
        self._create_and_add_parameters(
            station=station,
            parameters=parameters,
            setters=setters or {},
            units=units or {}
        )
        self._initial_values = initial_values or {}
        if set_initial_values_on_load:
            self.set_initial_values()

    @staticmethod
    def parse_instrument_path(parent: Union[Station, InstrumentBase], path: str) -> Any:
        """Parse a string path and return the object relative to a station or
        instrument, e.g. "my_instrument.my_param" returns
        station.my_instrument.my_param

        Args:
            station: Measurement station
            path: Relative path to parse
        """
        def _parse_path(parent: Any, elem: Sequence[str]) -> Any:
            child = getattr(parent, elem[0])
            if len(elem) == 1:
                return child
            return _parse_path(child, elem[1:])

        return _parse_path(parent, path.split("."))

    def set_initial_values(self, dry_run: bool = False) -> None:
        """Set parameter initial values on meta instrument

        Args:
            dry_run: Dry run to test if defaults are set correctly.
                Defaults to False.
        """
        _log.debug(f"Setting default values: {self._initial_values}")
        for path, value in self._initial_values.items():
            param = self.parse_instrument_path(parent=self, path=path)
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
                        parent = self.parse_instrument_path(
                            parent=self,
                            path=parent_path
                        )
                    else:
                        parent, name = self, path
                    print(parent, name, value)
                    setattr(parent, name, value)
            else:
                print(f"Dry run: {msg}")

    def _create_and_add_parameters(
        self,
        station: Station,
        parameters: Dict[str, List[str]],
        setters: Dict[str, Dict[str, Any]],
        units: Dict[str, str]
    ) -> None:
        """Add parameters to meta instrument based on specified aliases,
        endpoints and setter methods"""
        for param_name, paths in parameters.items():
            self._create_and_add_parameter(
                group_name=param_name,
                station=station,
                paths=paths,
                setter=setters.get(param_name),
                unit=units.get(param_name)
            )

    @staticmethod
    def _parameter_names(parameters: List[Parameter]) -> List[str]:
        """Get the endpoint names"""
        parameter_names = [_e.name for _e in parameters]
        if len(parameter_names) != len(set(parameter_names)):
            parameter_names = [
                f"{_e}{n}" for n, _e in enumerate(parameter_names)
            ]
        return parameter_names
    

    def _add_parameter(
        self,
        group_name: str,
        name: str,
        source: Parameter,
    ) -> DelegateGroupParameter:
        param = self.param_cls(
            name=f"{group_name}_{name}",
            instrument=self,
            source=source
        )
        if param.name in self.parameters:
            raise KeyError(f'Duplicate parameter name {param.name}')
        self.parameters[param.name] = param
        return param

    def _create_and_add_parameter(
        self,
        group_name: str,
        station: Station,
        paths: List[str],
        setter: Optional[Dict[str, Any]] = None,
        getter: Optional[Callable[..., Any]] = None,
        formatter: Optional[Callable[..., Any]] = None,
        unit: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Create meta parameter that links to a given set of paths
        (e.g. my_instrument.my_param) on the station"""
        source_parameters = [
            self.parse_instrument_path(station, path) for path in paths
        ]
        parameter_names = self._parameter_names(source_parameters)

        setter_fn = None
        if setter is not None:
            setter_method = self.parse_instrument_path(
                station, setter.pop("method")
            )
            setter_fn = partial(setter_method, **setter)

        params = [
            self._add_parameter(group_name, name, source) 
            for name, source in zip(parameter_names, source_parameters)
        ]

        group = DelegateGroup(
            name=group_name,
            parameters=params,
            parameter_names=parameter_names,
            setter=setter_fn,
            getter=getter,
            formatter=formatter
        )

        self.add_parameter(
            name=group_name,
            parameter_class=GroupedParameter,
            group=group,
            unit=unit,
            **kwargs
        )

    def __repr__(self) -> str:
        params = ", ".join(self.parameters.keys())
        return f"DelegateInstrument(name={self.name}, parameters={params})"
