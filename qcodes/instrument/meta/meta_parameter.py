from qcodes.instrument.group_parameter import Group
from typing import Any, Dict, Optional, Tuple, Union
from collections import namedtuple
from qcodes.instrument.parameter import (
    ParamDataType,
    ParamRawDataType,
    Parameter,
    _BaseParameter
)

import logging

_log = logging.getLogger(__name__)


class MetaGroup(Group):
    """Group of Meta parameters"""
    def __init__(
        self,
        name: str,
        parameters: Tuple['MetaParameter'],
        parameter_names: Tuple[str] = None,
        **kwargs
    ):
        super().__init__(parameters=list(parameters), single_instrument=False, **kwargs)
        parameter_names = parameter_names or [_e.name for _e in parameters]
        self._namedtuple = namedtuple(name, parameter_names)

    def get_parameters(self) -> Any:
        return self._namedtuple(*[_p.get() for _p in self.parameters.values()])

    def _set_from_dict(self, calling_dict: Dict[str, ParamRawDataType]) -> None:
        for name, p in list(self.parameters.items()):
            p.set(calling_dict[name])


class MetaParameter(_BaseParameter):
    """
    Meta parameter that returns one or more endpoint parameter values
    
    Args:
        name: Meta parameter name
        endpoints: One or more endpoint parameters to alias
        endpoint_names: One or more endpoint names
        unit: The unit of measure. Use ``''`` for unitless.
        setter: Optional setter function to override
            endpoint parameter setter. Defaults to None.
    """
    def __init__(
        self,
        name: str,
        endpoints: Tuple[Parameter],
        endpoint_names: Tuple[str],
        unit: str = None,
        setter: callable = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self._endpoints = endpoints
        self._setter = setter
        self.unit = unit if unit is not None else ''
        endpoint_names = endpoint_names or [_e.name for _e in endpoints]
        self._group = None

        # Generate sub parameters if >1 endpoints,
        # e.g. my_meta_param.X, my_meta_param.Y
        if len(endpoints) > 1:
            self._parameters = self._sub_parameters(endpoints, endpoint_names)
        self._set = self.set
        self.set = self._set_override

    @property
    def group(self) -> Optional['MetaGroup']:
        """
        The group that this parameter belongs to.
        """
        return self._group

    def _sub_parameters(self, endpoints: Tuple[Parameter], endpoint_names: Tuple[str]):
        """Generate sub parameters if there are > 1 endpoints."""
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
        return tuple(parameters)

    def add_parameter(
        self,
        cls,
        name: str,
        endpoint: Parameter
    ) -> Parameter:
        """Add a sub parameter to the meta parameter

        Args:
            name: Sub parameter name
            endpoint: Endpoint parameter object

        Returns:
            Parameter: Parameter instance
        """
        assert not hasattr(
            self, name
        ), f"Duplicate parameter name {name}."
        parameter = cls(
            name=name,
            endpoints=(endpoint,),
            endpoint_names=(endpoint.name,),
            instrument=None
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
        if self.group is not None:
            return self.group.get_parameters()
        return self.endpoints[0]()

    def set_raw(self, value: Union[float, Dict[str, ParamDataType]]):
        """Set parameter raw value

        Args:
            value: Parameter value to set

        Returns:
            float: Returns the parameter value
        """
        if self._setter is not None:
            return self._setter(value)
        elif self.group is not None:
            self.group.set_parameters(value)
        else:
            return self._endpoints[0](value)
    
    def _set_override(self, value: ParamDataType = None, **kwargs):
        """Overide set method such that values can optionally be set using only kwargs"""
        if value is None:
            self._set(kwargs)
        else:
            self._set(value, **kwargs)

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
