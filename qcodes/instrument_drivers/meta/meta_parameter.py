from typing import Tuple

from collections import namedtuple

from qcodes.instrument.parameter import Parameter, _BaseParameter

import logging

_log = logging.getLogger(__name__)


class MetaParameter(_BaseParameter):
    """Meta parameter that returns one or more endpoint parameter values"""
    def __init__(
        self,
        name: str,
        endpoints: Tuple[Parameter],
        endpoint_names: Tuple[str],
        unit: str = None,
        setter: callable = None,
        **kwargs
    ):
        """Meta parameter that acts as an alias to a real instrument parameter

        Args:
            name: Meta parameter name
            endpoints: One or more endpoint parameters to alias
            endpoint_names: One or more endpoint names
            unit: The unit of measure. Use ``''`` for unitless.
            setter: Optional setter function to override
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
        self.unit = unit if unit is not None else ''

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
